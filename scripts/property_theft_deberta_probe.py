#!/usr/bin/env python3
"""Run a DeBERTa-backed "property is theft" microprobe.

This is the model-facing counterpart to ``property_theft_microprobe.py``.  It
creates a small controlled corpus where the same events are written through
property-rights and theft/dispossession frames, hides those labels from the
extractor, runs the actual DeBERTa NLI observer bank, and compares whether model
geometry separates cross-frame pairs more strongly than same-frame pairs.

The probe is not a substitute for full-corpus validation.  It is a narrow
instrument check for one paper claim: topical similarity and perspectival
traversability are not the same thing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.property_theft_microprobe import MicroArticle, _articles  # noqa: E402


def _article_text(article: MicroArticle) -> str:
    return (
        f"{article.title}\n\n"
        f"{article.content}\n\n"
        "Background: local officials, residents, courts, and mediators all "
        "describe the same event using competing vocabularies."
    )


def _write_jsonl(path: Path, articles: Sequence[MicroArticle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for article in articles:
            payload = {
                "title": article.title,
                "content": _article_text(article),
                "text": _article_text(article),
                "source": "property_theft_microprobe",
                "publication": "property_theft_microprobe",
                "author": "synthetic_controlled",
                "event_id": article.hidden_label,
                "perspective_tag": article.frame,
                "frame_label": article.frame,
                "label": article.frame,
                "hidden_label": article.hidden_label,
                "article_index": article.index,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_hidden_groups(path: Path, articles: Sequence[MicroArticle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["article_id", "group_topic", "frame", "zone"],
        )
        writer.writeheader()
        for article in articles:
            writer.writerow(
                {
                    "article_id": article.index,
                    "group_topic": article.hidden_label,
                    "frame": article.frame,
                    "zone": article.zone,
                }
            )


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(av) * np.linalg.norm(bv)), 1e-12)
    return float(1.0 - (float(np.dot(av, bv)) / denom))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _std(values: Iterable[float]) -> Optional[float]:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.std(vals)) if vals.size else None


def _pair_contrast(
    *,
    articles: Sequence[MicroArticle],
    features: np.ndarray,
    metric: str,
) -> Dict[str, Any]:
    distance_fn = _cosine_distance if metric == "cosine" else _euclidean_distance
    same_frame: List[float] = []
    cross_frame: List[float] = []
    bridge_to_property: List[float] = []
    bridge_to_theft: List[float] = []
    rows: List[Dict[str, Any]] = []
    index_to_row = {article.index: pos for pos, article in enumerate(articles)}

    for left, right in combinations(articles, 2):
        if left.hidden_label != right.hidden_label:
            continue
        left_vec = features[index_to_row[left.index]]
        right_vec = features[index_to_row[right.index]]
        dist = distance_fn(left_vec, right_vec)
        frames = {left.frame, right.frame}
        relation = "ignored"
        if frames <= {"property", "theft"}:
            if left.frame == right.frame:
                same_frame.append(dist)
                relation = "same_frame"
            else:
                cross_frame.append(dist)
                relation = "cross_frame"
        elif "bridge" in frames and "property" in frames:
            bridge_to_property.append(dist)
            relation = "bridge_to_property"
        elif "bridge" in frames and "theft" in frames:
            bridge_to_theft.append(dist)
            relation = "bridge_to_theft"
        rows.append(
            {
                "hidden_label": left.hidden_label,
                "left_article_idx": left.index,
                "right_article_idx": right.index,
                "left_frame": left.frame,
                "right_frame": right.frame,
                "relation": relation,
                "distance": dist,
            }
        )

    same_mean = _mean(same_frame)
    cross_mean = _mean(cross_frame)
    return {
        "metric": metric,
        "same_frame_pair_count": len(same_frame),
        "cross_frame_pair_count": len(cross_frame),
        "mean_same_frame_distance": same_mean,
        "mean_cross_frame_distance": cross_mean,
        "cross_minus_same_distance": (
            float(cross_mean - same_mean)
            if same_mean is not None and cross_mean is not None
            else None
        ),
        "cross_over_same_distance_ratio": (
            float(cross_mean / max(same_mean, 1e-12))
            if same_mean is not None and cross_mean is not None
            else None
        ),
        "mean_bridge_to_property_distance": _mean(bridge_to_property),
        "mean_bridge_to_theft_distance": _mean(bridge_to_theft),
        "pair_rows": rows,
    }


def _cluster_scores(features: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception as exc:
        return {"status": "SKIPPED", "reason": f"sklearn unavailable: {exc}"}
    unique = sorted(set(labels))
    if len(unique) < 2 or len(labels) < len(unique):
        return {"status": "INVALID", "reason": "not enough labels"}
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int32)
    kmeans = KMeans(n_clusters=len(unique), random_state=42, n_init=20)
    pred = kmeans.fit_predict(features)
    return {
        "status": "OK",
        "label_set": unique,
        "n_clusters": len(unique),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ari": float(adjusted_rand_score(y, pred)),
    }


def _tfidf_features(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=1024,
        stop_words="english",
    )
    return vectorizer.fit_transform(texts).toarray().astype(np.float32)


def _extract_deberta_features(
    articles: Sequence[MicroArticle],
    *,
    model_name: str,
    max_length: int,
    cache_path: Path,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    import torch
    from core.nli_extraction import ExtractionConfig, UnifiedNLIExtractor

    config = ExtractionConfig(
        model_name=model_name,
        max_length=max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cls_tokens=True,
        extract_logits=True,
        record_heads=False,
        apply_pca_to_cls=False,
    )
    extractor = UnifiedNLIExtractor(config)
    batch = [{"text": _article_text(article), "title": article.title} for article in articles]
    extracted = extractor.extract_batch(batch, show_progress=True)
    logits_flat = extracted["logits_flat"].detach().cpu().numpy().astype(np.float32)
    cls_per_bot = extracted["cls_per_bot"].detach().cpu().numpy().astype(np.float32)
    cls_stacked = extracted["cls_stacked"].detach().cpu().numpy().astype(np.float32)
    bot_norms = np.linalg.norm(cls_per_bot, axis=2).astype(np.float32)
    bot_signed_first_pc = _first_pc_scores(cls_per_bot.reshape(cls_per_bot.shape[0], -1)).astype(np.float32)
    payload = {
        "logits_flat": logits_flat,
        "cls_stacked": cls_stacked,
        "bot_norms": bot_norms,
        "spectral_pc1": bot_signed_first_pc.reshape(-1, 1),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def _first_pc_scores(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if min(x.shape) <= 1:
        return np.zeros((x.shape[0],), dtype=np.float64)
    u, s, _vh = np.linalg.svd(x, full_matrices=False)
    return u[:, 0] * s[0]


def run_property_theft_deberta_probe(
    output_dir: Path,
    *,
    model_name: str = "microsoft/deberta-v2-xlarge-mnli",
    max_length: int = 256,
    force_extract: bool = False,
    pipeline_artifact: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = _articles()
    texts = [_article_text(article) for article in articles]
    frame_labels = [article.frame for article in articles]
    binary_frame_labels = [
        article.frame if article.frame in {"property", "theft"} else "other"
        for article in articles
    ]
    event_labels = [article.hidden_label for article in articles]

    corpus_path = output_dir / "property_theft_corpus.jsonl"
    _write_jsonl(corpus_path, articles)
    _write_hidden_groups(output_dir / "labels" / "hidden_groups.csv", articles)

    tfidf = _tfidf_features(texts)
    deberta = _extract_deberta_features(
        articles,
        model_name=model_name,
        max_length=max_length,
        cache_path=output_dir / "deberta_features.npz",
        force=force_extract,
    )
    feature_spaces: Dict[str, np.ndarray] = {
        "tfidf": tfidf,
        "deberta_logits": deberta["logits_flat"],
        "deberta_observer_delta": deberta["cls_stacked"],
        "deberta_observer_norms": deberta["bot_norms"],
        "deberta_spectral_pc1": deberta["spectral_pc1"],
    }

    feature_reports: Dict[str, Any] = {}
    for name, features in feature_spaces.items():
        feature_reports[name] = {
            "shape": list(features.shape),
            "cosine_pair_contrast": _pair_contrast(
                articles=articles,
                features=features,
                metric="cosine",
            ),
            "euclidean_pair_contrast": _pair_contrast(
                articles=articles,
                features=features,
                metric="euclidean",
            ),
            "frame_cluster_scores": _cluster_scores(features, frame_labels),
            "binary_frame_cluster_scores": _cluster_scores(features, binary_frame_labels),
            "event_cluster_scores": _cluster_scores(features, event_labels),
        }

    pipeline_report = (
        _summarize_pipeline_artifact(
            Path(pipeline_artifact),
            articles=articles,
            frame_labels=frame_labels,
            binary_frame_labels=binary_frame_labels,
            event_labels=event_labels,
        )
        if pipeline_artifact is not None
        else {"status": "NOT_REQUESTED"}
    )

    primary = feature_reports["deberta_observer_delta"]["cosine_pair_contrast"]
    baseline = feature_reports["tfidf"]["cosine_pair_contrast"]
    primary_lift = primary.get("cross_minus_same_distance")
    baseline_lift = baseline.get("cross_minus_same_distance")
    lift_over_tfidf = (
        float(primary_lift - baseline_lift)
        if primary_lift is not None and baseline_lift is not None
        else None
    )
    summary = {
        "schema_version": "1.0",
        "probe_name": "property_is_theft_deberta_microprobe",
        "model_name": model_name,
        "max_length": max_length,
        "output_dir": str(output_dir),
        "corpus_path": str(corpus_path),
        "article_count": len(articles),
        "articles": [asdict(article) for article in articles],
        "feature_reports": feature_reports,
        "pipeline_artifact_report": pipeline_report,
        "primary_feature_space": "deberta_observer_delta",
        "baseline_feature_space": "tfidf",
        "primary_cross_minus_same_cosine": primary_lift,
        "baseline_cross_minus_same_cosine": baseline_lift,
        "primary_lift_over_tfidf": lift_over_tfidf,
        "passes_microprobe": bool(
            primary_lift is not None
            and primary_lift > 0.0
            and lift_over_tfidf is not None
            and lift_over_tfidf >= -0.02
        ),
        "interpretation": (
            "A pass means DeBERTa observer-delta geometry treats property/theft "
            "cross-frame same-event pairs as farther apart than same-frame pairs. "
            "It does not prove real-corpus validity; it only supports the targeted "
            "capability under a controlled micro-environment."
        ),
    }
    (output_dir / "property_theft_deberta_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _summarize_pipeline_artifact(
    artifact_path: Path,
    *,
    articles: Sequence[MicroArticle],
    frame_labels: Sequence[str],
    binary_frame_labels: Sequence[str],
    event_labels: Sequence[str],
) -> Dict[str, Any]:
    if not artifact_path.exists():
        return {"status": "MISSING", "artifact_path": str(artifact_path)}
    try:
        import torch

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"status": "LOAD_FAILED", "artifact_path": str(artifact_path), "error": str(exc)}
    if not isinstance(artifact, dict):
        return {"status": "INVALID", "artifact_path": str(artifact_path), "error": "artifact is not a dict"}

    feature_sources: Dict[str, np.ndarray] = {}
    for key in ("features", "integrated_vectors", "embeddings"):
        value = artifact.get(key)
        if value is None:
            continue
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if arr.ndim >= 2 and arr.shape[0] == len(articles):
            feature_sources[key] = arr.reshape(arr.shape[0], -1).astype(np.float32)
    cls = artifact.get("cls_per_bot")
    if cls is not None:
        arr = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)
        if arr.ndim >= 3 and arr.shape[0] == len(articles):
            feature_sources["cls_per_bot_flat"] = arr.reshape(arr.shape[0], -1).astype(np.float32)
            feature_sources["cls_per_bot_norms"] = np.linalg.norm(arr, axis=2).astype(np.float32)
    logits = artifact.get("logits_raw")
    if logits is not None:
        arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        if arr.ndim >= 2 and arr.shape[0] == len(articles):
            feature_sources["logits_raw_flat"] = arr.reshape(arr.shape[0], -1).astype(np.float32)

    feature_reports: Dict[str, Any] = {}
    for name, features in feature_sources.items():
        feature_reports[name] = {
            "shape": list(features.shape),
            "cosine_pair_contrast": _pair_contrast(
                articles=articles,
                features=features,
                metric="cosine",
            ),
            "euclidean_pair_contrast": _pair_contrast(
                articles=articles,
                features=features,
                metric="euclidean",
            ),
            "frame_cluster_scores": _cluster_scores(features, frame_labels),
            "binary_frame_cluster_scores": _cluster_scores(features, binary_frame_labels),
            "event_cluster_scores": _cluster_scores(features, event_labels),
        }

    work_values = artifact.get("walker_work_integrals")
    work_report: Dict[str, Any] = {"status": "MISSING"}
    if work_values is not None:
        work = work_values.detach().cpu().numpy() if hasattr(work_values, "detach") else np.asarray(work_values)
        work = work.reshape(-1).astype(np.float64)
        if work.shape[0] == len(articles):
            work_features = work.reshape(-1, 1)
            work_report = {
                "status": "OK",
                "mean_work": _mean(work.tolist()),
                "std_work": _std(work.tolist()),
                "cosine_pair_contrast": _pair_contrast(
                    articles=articles,
                    features=work_features,
                    metric="cosine",
                ),
                "euclidean_pair_contrast": _pair_contrast(
                    articles=articles,
                    features=work_features,
                    metric="euclidean",
                ),
            }

    validation_metrics = artifact.get("validation_metrics") or artifact.get("metrics") or {}
    spectral_evr = artifact.get("spectral_evr")
    if hasattr(spectral_evr, "detach"):
        spectral_evr = float(spectral_evr.detach().cpu().reshape(-1)[0].item())
    elif isinstance(spectral_evr, (list, tuple, np.ndarray)):
        arr = np.asarray(spectral_evr).reshape(-1)
        spectral_evr = float(arr[0]) if arr.size else None
    elif spectral_evr is not None:
        try:
            spectral_evr = float(spectral_evr)
        except Exception:
            spectral_evr = None

    return {
        "status": "OK",
        "artifact_path": str(artifact_path),
        "artifact_keys": sorted(str(key) for key in artifact.keys()),
        "feature_reports": feature_reports,
        "work_report": work_report,
        "validation_metrics": validation_metrics,
        "spectral_evr": spectral_evr,
        "track5_mode": (
            (artifact.get("provenance") or {}).get("track5_assembly_mode")
            if isinstance(artifact.get("provenance"), dict)
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/microprobes/property_theft/deberta_latest",
    )
    parser.add_argument(
        "--model",
        default="microsoft/deberta-v2-xlarge-mnli",
        help="Hugging Face NLI model. Default matches the repo extractor.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--pipeline-artifact",
        default=None,
        help="Optional observer_*.pt artifact from run_experiments.py to summarize beside the direct extractor result.",
    )
    args = parser.parse_args()

    summary = run_property_theft_deberta_probe(
        Path(args.output_dir),
        model_name=args.model,
        max_length=args.max_length,
        force_extract=args.force_extract,
        pipeline_artifact=Path(args.pipeline_artifact) if args.pipeline_artifact else None,
    )
    print("Property/Theft DeBERTa microprobe complete")
    print(f"- output_dir: {summary['output_dir']}")
    print(f"- model: {summary['model_name']}")
    print(f"- passes_microprobe: {summary['passes_microprobe']}")
    print(f"- primary_cross_minus_same_cosine: {summary['primary_cross_minus_same_cosine']}")
    print(f"- baseline_cross_minus_same_cosine: {summary['baseline_cross_minus_same_cosine']}")
    print(f"- primary_lift_over_tfidf: {summary['primary_lift_over_tfidf']}")
    return 0 if summary["passes_microprobe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
