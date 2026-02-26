"""
Canonical Article ID System - Thesis-Grade Provenance
======================================================

Ensures stable, verifiable article identity across:
- Multiple scrapes
- Different input file orders
- Platform changes
- Pipeline runs

Key Principles:
1. IDs derived from immutable content (url, title, source, date)
2. Deterministic collision handling
3. Hash-based verification
4. Explicit ordering (never rely on list position)
"""

import hashlib
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np


def compute_canonical_uid(
    url: str,
    title: str,
    source: str,
    published_date: str,
    collision_suffix: Optional[int] = None
) -> str:
    """
    Compute canonical UID from immutable article fields.
    
    Parameters
    ----------
    url : str
        Article URL (primary identifier)
    title : str
        Article title
    source : str
        Publisher/source domain
    published_date : str
        Publication date (any format, will be stringified)
    collision_suffix : int, optional
        If set, append deterministic suffix for collision resolution
    
    Returns
    -------
    str
        16-character hex UID (or longer if collision suffix added)
    
    Examples
    --------
    >>> compute_canonical_uid(
    ...     "https://example.com/article",
    ...     "Test Article",
    ...     "example.com",
    ...     "2024-01-01"
    ... )
    'a3f9b2c8d1e4f567'
    """
    # Build canonical string from immutable fields
    base = f"{url}|{title}|{source}|{published_date}"
    
    # Hash to 16-char hex
    uid = hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:16]
    
    # Add collision suffix if needed
    if collision_suffix is not None:
        uid = f"{uid}_{collision_suffix}"
    
    return uid


def assign_canonical_uids(articles: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Assign canonical UIDs to all articles with collision handling.
    
    Modifies articles in-place by adding 'bt_uid' field.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles with 'url', 'title', 'source', 'published_date' (or similar)
    
    Returns
    -------
    articles : List[Dict]
        Same list, now with 'bt_uid' field added
    stats : Dict
        Statistics about UID assignment:
        - n_articles: total articles
        - n_unique_uids: unique base UIDs
        - n_collisions: number of collisions
        - collision_examples: list of collided UIDs
    """
    seen = {}
    collision_examples = []
    
    for i, article in enumerate(articles):
        # Extract fields (with fallbacks)
        url = article.get('url', article.get('canonical_id', ''))
        title = article.get('title', '')
        source = article.get('source', article.get('publisher', 'unknown'))
        date = str(article.get('published_date', article.get('published_at', article.get('date', ''))))
        
        # Compute base UID
        base_uid = compute_canonical_uid(url, title, source, date)
        
        # Handle collisions
        if base_uid in seen:
            k = seen[base_uid]
            seen[base_uid] += 1
            uid = f"{base_uid}_{k}"
            
            if len(collision_examples) < 5:
                collision_examples.append({
                    'base_uid': base_uid,
                    'collision_count': k,
                    'url1': articles[seen[base_uid + '_first_idx']].get('url', ''),
                    'url2': url
                })
        else:
            seen[base_uid] = 1
            seen[base_uid + '_first_idx'] = i
            uid = base_uid
        
        # Assign UID
        article['bt_uid'] = uid
    
    stats = {
        'n_articles': len(articles),
        'n_unique_base_uids': len([k for k in seen.keys() if not k.endswith('_first_idx')]),
        'n_collisions': sum(v - 1 for k, v in seen.items() if not k.endswith('_first_idx')),
        'collision_examples': collision_examples
    }
    
    return articles, stats


def canonical_sort(articles: List[Dict]) -> List[Dict]:
    """
    Sort articles by canonical UID (lexicographic).
    
    This is the CANONICAL ORDER - all arrays must follow this.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles with 'bt_uid' field
    
    Returns
    -------
    List[Dict]
        Sorted articles
    """
    if not articles:
        return articles
    
    if 'bt_uid' not in articles[0]:
        raise ValueError("Articles must have 'bt_uid' field! Call assign_canonical_uids() first.")
    
    return sorted(articles, key=lambda a: a['bt_uid'])


def compute_corpus_hash(articles: List[Dict]) -> str:
    """
    Compute SHA256 hash of all UIDs concatenated.
    
    This is the FINGERPRINT of your corpus - use it to verify consistency.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles with 'bt_uid' field, in canonical order
    
    Returns
    -------
    str
        SHA256 hex digest
    """
    if not articles:
        return hashlib.sha256(b"").hexdigest()
    
    uids = [a['bt_uid'] for a in articles]
    concatenated = "".join(uids)
    return hashlib.sha256(concatenated.encode('utf-8')).hexdigest()


def create_manifest(articles: List[Dict], output_path: Path) -> Dict:
    """
    Create manifest JSON with canonical UIDs and verification hashes.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles in canonical order with 'bt_uid' field
    output_path : Path
        Where to save manifest.json
    
    Returns
    -------
    Dict
        Manifest contents
    """
    uids = [a['bt_uid'] for a in articles]
    timestamps = [a.get('published_ts', a.get('published_date', None)) for a in articles]
    
    # Filter valid timestamps
    valid_timestamps = [t for t in timestamps if t is not None]
    
    manifest = {
        'n_articles': len(articles),
        'canonical_uids': uids,
        'canonical_hash': compute_corpus_hash(articles),
        'first_uid': uids[0] if uids else None,
        'last_uid': uids[-1] if uids else None,
        'timestamp_min': min(valid_timestamps) if valid_timestamps else None,
        'timestamp_max': max(valid_timestamps) if valid_timestamps else None,
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"âœ… Manifest saved: {output_path}")
    print(f"   Articles: {manifest['n_articles']}")
    print(f"   Hash: {manifest['canonical_hash'][:16]}...")
    
    return manifest


def verify_corpus(articles: List[Dict], manifest: Dict) -> bool:
    """
    Verify that articles match manifest fingerprint.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles to verify
    manifest : Dict
        Previously saved manifest
    
    Returns
    -------
    bool
        True if match, False otherwise (raises AssertionError in strict mode)
    
    Raises
    ------
    AssertionError
        If verification fails in strict mode
    """
    calculated_hash = compute_corpus_hash(articles)
    expected_hash = manifest['canonical_hash']
    
    if calculated_hash != expected_hash:
        print(f"âŒ CORPUS VERIFICATION FAILED!")
        print(f"   Expected: {expected_hash}")
        print(f"   Got:      {calculated_hash}")
        print(f"   This means article order or membership has changed!")
        return False
    
    print(f"âœ… Corpus verified: {calculated_hash[:16]}...")
    return True


def create_fingerprint(articles: List[Dict], output_path: Path) -> Dict:
    """
    Create lightweight fingerprint file for quick verification.
    
    Parameters
    ----------
    articles : List[Dict]
        Articles in canonical order
    output_path : Path
        Where to save fingerprint.json
    
    Returns
    -------
    Dict
        Fingerprint contents
    """
    uids = [a['bt_uid'] for a in articles]
    timestamps = [a.get('published_ts') for a in articles if a.get('published_ts')]
    
    # Temporal hash (for timeline verification)
    temporal_str = "".join(str(t) for t in timestamps if t is not None)
    temporal_hash = hashlib.sha256(temporal_str.encode()).hexdigest() if temporal_str else None
    
    fingerprint = {
        'n_articles': len(articles),
        'uids_hash': compute_corpus_hash(articles),
        'temporal_hash': temporal_hash,
        'first_uid': uids[0] if uids else None,
        'last_uid': uids[-1] if uids else None,
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    print(f"âœ… Fingerprint: {output_path}")
    
    return fingerprint


def verify_array_alignment(arrays: Dict[str, np.ndarray], uids: List[str]) -> bool:
    """
    Verify that all arrays have correct length matching UIDs.
    
    Parameters
    ----------
    arrays : Dict[str, np.ndarray]
        Named arrays to check (e.g., {'features': ..., 'embeddings': ...})
    uids : List[str]
        Canonical UIDs
    
    Returns
    -------
    bool
        True if all arrays match UID count
    
    Raises
    ------
    AssertionError
        If any array has wrong length
    """
    n_uids = len(uids)
    
    for name, array in arrays.items():
        if len(array) != n_uids:
            raise AssertionError(
                f"Array '{name}' has {len(array)} rows but corpus has {n_uids} UIDs!"
            )
    
    print(f"âœ… All {len(arrays)} arrays aligned with {n_uids} UIDs")
    return True


def create_provenance_log(
    output_path: Path,
    uids_hash: str,
    order_strategy: str = "canonical_bt_uid",
    temporal_sort_applied: bool = False,
    additional_info: Optional[Dict] = None
) -> Dict:
    """
    Create provenance log for pipeline output.
    
    This is your "chain of custody" - proves which corpus version was used.
    
    Parameters
    ----------
    output_path : Path
        Where to save provenance.json
    uids_hash : str
        SHA256 hash of canonical UIDs
    order_strategy : str
        How articles were ordered
    temporal_sort_applied : bool
        Whether temporal sorting was done (for GRU)
    additional_info : Dict, optional
        Any additional provenance metadata
    
    Returns
    -------
    Dict
        Provenance log contents
    """
    log = {
        'uids_sha256': uids_hash,
        'order_strategy': order_strategy,
        'temporal_sort_applied': temporal_sort_applied,
    }
    
    if additional_info:
        log.update(additional_info)
    
    with open(output_path, 'w') as f:
        json.dump(log, f, indent=2)
    
    return log


# ============================================================================
# Pipeline Integration Helpers
# ============================================================================

def prepare_articles_with_ids(articles: List[Dict], sort_canonical: bool = True) -> Tuple[List[Dict], List[str], Dict]:
    """
    One-shot preparation of articles with canonical IDs for pipeline use.
    
    This is the recommended entry point for pipeline integration.
    
    Args:
        articles: Raw articles
        sort_canonical: If True, sort by canonical UID
        
    Returns:
        articles: Articles with bt_uid assigned (potentially reordered)
        canonical_ids: List of canonical UIDs in order
        stats: Statistics about assignment
    """
    # Assign UIDs
    articles, stats = assign_canonical_uids(articles)
    
    # Optional canonical sort
    if sort_canonical:
        articles = canonical_sort(articles)
    
    # Extract IDs in current order
    canonical_ids = [a['bt_uid'] for a in articles]
    
    return articles, canonical_ids, stats


def verify_artifact_ids(artifact: Dict) -> Dict:
    """
    Verify that a saved artifact has valid canonical IDs.
    
    Args:
        artifact: Loaded .pt artifact dictionary
        
    Returns:
        Verification results
    """
    results = {
        'has_ids': False,
        'n_ids': 0,
        'ids_aligned': False,
        'issues': [],
    }
    
    # Check for IDs
    ids = artifact.get('ids') or artifact.get('canonical_ids') or artifact.get('bt_uid_list')
    if ids is not None and len(ids) > 0:
        results['has_ids'] = True
        results['n_ids'] = len(ids)
        
        # Check alignment with features
        for key in ['fused', 'embeddings', 'features']:
            tensor = artifact.get(key)
            if tensor is not None:
                import torch
                if torch.is_tensor(tensor):
                    n_features = tensor.shape[0]
                    results['ids_aligned'] = (len(ids) == n_features)
                    if not results['ids_aligned']:
                        results['issues'].append(f"ID count ({len(ids)}) != {key} count ({n_features})")
                break
    else:
        results['issues'].append("No canonical IDs found!")
    
    return results


def cross_verify_artifacts(artifacts: List[Dict]) -> Dict:
    """
    Verify that multiple artifacts have matching canonical IDs.
    
    CRITICAL for comparing observers - mismatched IDs mean
    comparing different articles!
    
    Args:
        artifacts: List of loaded artifact dictionaries
        
    Returns:
        Cross-verification results
    """
    results = {
        'n_artifacts': len(artifacts),
        'all_have_ids': True,
        'all_ids_match': True,
        'common_count': 0,
        'issues': [],
    }
    
    id_lists = []
    for i, art in enumerate(artifacts):
        ids = art.get('ids') or art.get('canonical_ids') or art.get('bt_uid_list')
        if ids is None or len(ids) == 0:
            results['all_have_ids'] = False
            results['issues'].append(f"Artifact {i}: No IDs")
            id_lists.append(None)
        else:
            id_lists.append(list(ids))
    
    if results['all_have_ids'] and len(id_lists) >= 2:
        first_ids = id_lists[0]
        results['common_count'] = len(first_ids)
        
        for i, ids in enumerate(id_lists[1:], 1):
            if ids != first_ids:
                results['all_ids_match'] = False
                # Find common
                common = set(first_ids) & set(ids)
                results['common_count'] = min(results['common_count'], len(common))
                results['issues'].append(f"Artifact {i}: {len(common)}/{len(first_ids)} IDs match")
    
    return results


if __name__ == "__main__":
    # Test
    print("="*70)
    print("Testing Canonical ID System")
    print("="*70)
    
    # Mock articles
    articles = [
        {
            'url': 'https://example.com/article1',
            'title': 'Test Article 1',
            'source': 'example.com',
            'published_date': '2024-01-01'
        },
        {
            'url': 'https://example.com/article2',
            'title': 'Test Article 2',
            'source': 'example.com',
            'published_date': '2024-01-02'
        },
        # Duplicate (will collide)
        {
            'url': 'https://example.com/article1',
            'title': 'Test Article 1',
            'source': 'example.com',
            'published_date': '2024-01-01'
        }
    ]
    
    # Assign UIDs
    articles, stats = assign_canonical_uids(articles)
    print(f"\nâœ… Assigned UIDs:")
    print(f"   Articles: {stats['n_articles']}")
    print(f"   Unique base UIDs: {stats['n_unique_base_uids']}")
    print(f"   Collisions: {stats['n_collisions']}")
    
    for a in articles:
        print(f"   {a['bt_uid']}: {a['url']}")
    
    # Canonical sort
    articles = canonical_sort(articles)
    print(f"\nâœ… Canonical sort applied")
    
    # Compute hash
    corpus_hash = compute_corpus_hash(articles)
    print(f"\nâœ… Corpus hash: {corpus_hash}")
    
    # Create manifest
    manifest = create_manifest(articles, Path("test_manifest.json"))
    
    # Verify
    verify_corpus(articles, manifest)
    
    print("\n" + "="*70)
    print("âœ… Canonical ID system working!")
    print("="*70)