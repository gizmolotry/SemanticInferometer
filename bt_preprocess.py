# D:\belief-transformer\V3\bt_preprocess.py
# ============================================================
# Corpus Preprocessor (JSONL -> JSONL) for belief-transformer
#
# Goals:
# - Load large JSONL corpora robustly (handles UTF-8 BOM, bad lines)
# - Validate and normalize fields to a consistent schema
# - Parse/require timestamps
# - Generate fast, stable canonical IDs for dedupe across scrapes
# - Output preprocessed JSONL + manifest JSON report
#
# This file is a FULL REWRITE to match your actual data schema:
#   - uses "publisher" (domain) + "source_type" + "published_at"
#   - synthesizes "source" field for downstream compatibility
#   - avoids URL normalization pathological slowdown/hangs
#
# ============================================================

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, unquote, quote


# -----------------------------
# Config / Constants
# -----------------------------

PRINT_WIDTH = 70

# Drop known tracking/noise params that destroy stable identity.
TRACKING_PARAM_EXACT = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_name", "utm_id",
    "gclid", "fbclid", "msclkid",
    "igshid", "mc_cid", "mc_eid",
    "ref", "ref_src", "ref_url", "referrer",
    "spm", "mkt_tok",
    "_ga", "_gl",
    "cmpid", "cmp", "campaign",
    "ocid", "cid", "ncid",
    "s", "smid", "source",
}

TRACKING_PARAM_PREFIXES = (
    "utm_",  # anything utm_*
)

DEFAULT_MAX_QUERY_PARAMS = 25
DEFAULT_MAX_URL_LEN = 2048

# If query params exceed this, we drop query entirely (fast + stable).
HARD_QUERY_PARAM_CAP = 1500

# Cache sizes: big enough to benefit repeated URLs, not so big to explode RAM.
MAX_URL_NORM_CACHE = 400_000
MAX_CANON_CACHE = 400_000


# -----------------------------
# Small utilities
# -----------------------------

def hr(title: str) -> str:
    return "\n".join([
        "=" * PRINT_WIDTH,
        title,
        "=" * PRINT_WIDTH,
    ])


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_strip_bom(s: str) -> str:
    # Some files contain BOM not just at file start but on first line string itself.
    if s and s[0] == "\ufeff":
        return s.lstrip("\ufeff")
    return s


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def stable_hash16(text: str) -> str:
    # 64-bit digest -> 16 hex chars
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8)
    return h.hexdigest()


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_title(title: str) -> str:
    t = normalize_whitespace(title).lower()
    # kill silly unicode quotes spacing, etc.
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[^\w\s'\"-]+", "", t)  # keep words + basic punctuation
    t = normalize_whitespace(t)
    return t


def extract_domain_from_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or ""
    except Exception:
        return ""


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Timestamp parsing
# -----------------------------

_TS_PATTERNS = (
    # ISO formats
    ("%Y-%m-%d", "date"),
    ("%Y-%m-%dT%H:%M:%S", "naive"),
    ("%Y-%m-%d %H:%M:%S", "naive"),
    ("%Y-%m-%dT%H:%M:%S.%f", "naive"),
    ("%Y-%m-%d %H:%M:%S.%f", "naive"),
)

def parse_published_at(value: Any) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Returns:
      (published_ts_utc, published_date_yyyy_mm_dd, error_str)
    Accepts:
      - "YYYY-MM-DD" (your corpus)
      - ISO datetime strings
      - epoch seconds / ms
    """
    if value is None:
        return None, None, "missing"

    # epoch int/float
    if isinstance(value, (int, float)):
        v = int(value)
        # heuristics: ms vs sec
        if v > 10_000_000_000:  # ms
            v = v // 1000
        try:
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
            return int(dt.timestamp()), dt.date().isoformat(), None
        except Exception as e:
            return None, None, f"invalid_epoch:{e}"

    if not isinstance(value, str):
        return None, None, "not_a_string"

    s = value.strip()
    if s == "":
        return None, None, "empty"

    # common "Z"
    s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s

    # try python's fromisoformat when possible
    try:
        # date only
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s2):
            d = date.fromisoformat(s2)
            dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
            return int(dt.timestamp()), d.isoformat(), None

        # datetime with offset
        if re.match(r"^\d{4}-\d{2}-\d{2}T", s2) and ("+" in s2 or "-" in s2[10:]):
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            return int(dt.timestamp()), dt.date().isoformat(), None

        # naive datetime
        if re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", s2):
            # strip offset if weird
            base = s2
            # If there's a timezone but not parseable, cut at seconds/micros.
            base = re.sub(r"([0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?).*", r"\1", base)
            for fmt, _kind in _TS_PATTERNS:
                try:
                    dt = datetime.strptime(base, fmt).replace(tzinfo=timezone.utc)
                    return int(dt.timestamp()), dt.date().isoformat(), None
                except Exception:
                    pass
    except Exception:
        pass

    return None, None, "unparseable"


# -----------------------------
# URL Normalization (FAST)
# -----------------------------

@dataclass
class UrlNormConfig:
    max_query_params: int = DEFAULT_MAX_QUERY_PARAMS
    max_url_len: int = DEFAULT_MAX_URL_LEN
    drop_query_if_too_many: bool = True
    strip_www: bool = True


_URL_NORM_CACHE: Dict[str, str] = {}

def _cache_put(cache: Dict[str, str], key: str, value: str, max_size: int) -> None:
    # Cheap cap: if too big, clear (brutal but effective)
    if len(cache) >= max_size:
        cache.clear()
    cache[key] = value


def normalize_url(url: str, cfg: UrlNormConfig) -> str:
    """
    Normalizes URLs for stable identity.
    Key performance choices:
      - parse_qsl can be expensive on monstrous queries -> hard cap
      - if too many params -> drop query entirely
      - cache normalization results
    """
    if not is_nonempty_str(url):
        return ""

    url = url.strip()
    if url in _URL_NORM_CACHE:
        return _URL_NORM_CACHE[url]

    try:
        parts = urlsplit(url)
    except Exception:
        # not parseable
        _cache_put(_URL_NORM_CACHE, url, url, MAX_URL_NORM_CACHE)
        return url

    scheme = (parts.scheme or "https").lower()
    netloc = (parts.netloc or "").strip().lower()

    # If urlsplit got confused (e.g., missing scheme), try to recover
    if not netloc and parts.path and parts.path.startswith("www."):
        # e.g. "www.example.com/path"
        netloc = parts.path.split("/", 1)[0].lower()
        rest = "/" + parts.path.split("/", 1)[1] if "/" in parts.path else ""
        path = rest
    else:
        path = parts.path or ""

    if cfg.strip_www and netloc.startswith("www."):
        netloc = netloc[4:]

    # Remove default ports
    netloc = re.sub(r":(80|443)$", "", netloc)

    # Path normalization
    path = unquote(path)
    path = re.sub(r"/{2,}", "/", path)  # collapse multiple slashes
    if path != "/":
        path = path.rstrip("/")
    if not path:
        path = "/"
    # Re-quote conservatively
    path = quote(path, safe="/:@-._~%!$&'()*+,;=")

    # Fragment always dropped
    fragment = ""

    query = parts.query or ""
    norm_query = ""

    # Fast decision: if query is huge, drop it.
    if query:
        if len(query) > cfg.max_url_len:
            norm_query = ""
        else:
            # Hard cap on separators to avoid parse_qsl death spirals
            amp_count = query.count("&")
            if amp_count > HARD_QUERY_PARAM_CAP:
                norm_query = ""
            else:
                try:
                    pairs = parse_qsl(query, keep_blank_values=False)
                except Exception:
                    pairs = []

                if cfg.drop_query_if_too_many and len(pairs) > cfg.max_query_params:
                    norm_query = ""
                else:
                    filtered: List[Tuple[str, str]] = []
                    for k, v in pairs:
                        if not k:
                            continue
                        kk = k.lower()
                        if kk in TRACKING_PARAM_EXACT:
                            continue
                        if any(kk.startswith(pfx) for pfx in TRACKING_PARAM_PREFIXES):
                            continue
                        # normalize values lightly
                        vv = v.strip()
                        if vv == "":
                            continue
                        filtered.append((kk, vv))

                    if not filtered:
                        norm_query = ""
                    else:
                        # Sorting a small list is fine; we prevented huge lists above.
                        filtered.sort(key=lambda kv: (kv[0], kv[1]))
                        norm_query = urlencode(filtered, doseq=True)

    normalized = urlunsplit((scheme, netloc, path, norm_query, fragment))

    # Final clamp: if still absurdly long, drop query.
    if len(normalized) > cfg.max_url_len and norm_query:
        normalized = urlunsplit((scheme, netloc, path, "", fragment))

    _cache_put(_URL_NORM_CACHE, url, normalized, MAX_URL_NORM_CACHE)
    return normalized


# -----------------------------
# Canonical ID Generation (FAST + stable)
# -----------------------------

_CANON_CACHE: Dict[str, Tuple[str, Dict[str, Any]]] = {}

@dataclass
class CanonConfig:
    url_norm_cfg: UrlNormConfig


def synthesize_source(article: Dict[str, Any]) -> str:
    # Your data uses "publisher": "abc10.com"
    # Some sources may carry "source" already; keep it if present.
    if is_nonempty_str(article.get("source")):
        return str(article["source"]).strip()
    if is_nonempty_str(article.get("publisher")):
        return str(article["publisher"]).strip()
    # fallback: from URL
    url = article.get("url")
    if is_nonempty_str(url):
        dom = extract_domain_from_url(str(url))
        if dom:
            return dom
    # last fallback
    if is_nonempty_str(article.get("source_type")):
        return str(article["source_type"]).strip()
    return "unknown"


def generate_canonical_id(article: Dict[str, Any], cfg: CanonConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Generates a stable canonical ID for dedupe.
    Priority:
      1) normalized URL if present
      2) else hash(source + title + published_date)
    """
    url = article.get("url")
    source = synthesize_source(article)
    title = article.get("title") or ""
    published_date = article.get("published_date") or article.get("published_at") or ""

    # Cache key: for repeated identical records
    cache_key = f"{url}|{source}|{title}|{published_date}"
    if cache_key in _CANON_CACHE:
        return _CANON_CACHE[cache_key]

    url_norm = normalize_url(str(url), cfg.url_norm_cfg) if is_nonempty_str(url) else ""
    title_norm = normalize_title(str(title)) if is_nonempty_str(title) else ""

    # Build identity string
    if url_norm:
        # Use domain + path (query stripped/normalized already)
        # plus published_date when available to reduce edge collisions across re-used URLs
        ident = f"u::{url_norm}||d::{published_date}"
        basis = "url+date"
    else:
        ident = f"s::{source.lower()}||t::{title_norm}||d::{published_date}"
        basis = "source+title+date"

    cid = stable_hash16(ident)

    info = {
        "basis": basis,
        "ident": ident,
        "url_norm": url_norm,
        "title_norm": title_norm,
        "source": source,
        "published_date": published_date,
    }

    # cache
    if len(_CANON_CACHE) >= MAX_CANON_CACHE:
        _CANON_CACHE.clear()
    _CANON_CACHE[cache_key] = (cid, info)
    return cid, info


# -----------------------------
# Validation / Normalization
# -----------------------------

@dataclass
class ValidationRules:
    min_title_len: int = 10
    require_url: bool = True


def validate_article(article: Dict[str, Any], rules: ValidationRules) -> Tuple[bool, Dict[str, int]]:
    """
    Returns (is_valid, issue_counts_delta)
    Note: we treat "issues" as warnings unless they break required fields.
    """
    issues: Dict[str, int] = {}

    # Required URL
    url = article.get("url")
    if rules.require_url and not is_nonempty_str(url):
        issues["missing_url"] = issues.get("missing_url", 0) + 1
        return False, issues

    # Source presence (your old code flagged missing_source because it looked for "source")
    src = synthesize_source(article)
    if not is_nonempty_str(src) or src == "unknown":
        issues["missing_source"] = issues.get("missing_source", 0) + 1

    # Title checks
    title = article.get("title")
    if not is_nonempty_str(title):
        issues["missing_title"] = issues.get("missing_title", 0) + 1
    else:
        if len(str(title).strip()) < rules.min_title_len:
            issues["title_too_short"] = issues.get("title_too_short", 0) + 1

    # Content checks (warning only)
    content = article.get("content")
    snippets = article.get("snippets")
    if not is_nonempty_str(content) and not (isinstance(snippets, list) and len(snippets) > 0):
        issues["missing_text"] = issues.get("missing_text", 0) + 1

    return True, issues


def normalize_article_fields(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure downstream fields exist without destroying original schema.
    We keep your original keys (publisher, source_type, published_at, etc.)
    and add:
      - source
      - published_ts (UTC epoch seconds)
      - published_date (YYYY-MM-DD)
    """
    out = dict(article)

    # Ensure 'source'
    out["source"] = synthesize_source(out)

    # Timestamp handling
    ts, dstr, err = parse_published_at(out.get("published_at"))
    if ts is not None:
        out["published_ts"] = ts
    if dstr is not None:
        out["published_date"] = dstr
    if err is not None:
        out["_published_parse_error"] = err

    # Optional: normalize word_count/char_count if missing
    if "char_count" not in out:
        c = out.get("content")
        if is_nonempty_str(c):
            out["char_count"] = len(str(c))
    if "word_count" not in out:
        c = out.get("content")
        if is_nonempty_str(c):
            out["word_count"] = len(re.findall(r"\S+", str(c)))

    return out


# -----------------------------
# Dedupe strategy
# -----------------------------

@dataclass
class DedupeConfig:
    strategy: str = "first"  # first | last | longest_content
    keep_all: bool = False   # if True, do not dedupe, still compute canonical_id


def choose_better(existing: Dict[str, Any], candidate: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    if strategy == "first":
        return existing
    if strategy == "last":
        return candidate
    if strategy == "longest_content":
        ex_len = len(existing.get("content") or "")
        ca_len = len(candidate.get("content") or "")
        if ca_len > ex_len:
            return candidate
        return existing
    # default fallback
    return existing


# -----------------------------
# Main preprocess
# -----------------------------

@dataclass
class PreprocessReport:
    input_path: str
    output_path: str
    manifest_path: Optional[str]
    started_at: str
    finished_at: Optional[str] = None

    loaded_lines: int = 0
    parsed_articles: int = 0
    json_errors: int = 0

    valid_articles: int = 0
    dropped_articles: int = 0
    issues: Dict[str, int] = dataclasses.field(default_factory=dict)

    require_timestamp: bool = False
    timestamp_valid: int = 0
    timestamp_missing: int = 0
    timestamp_invalid: int = 0

    dedupe_strategy: str = "first"
    canonical_generated: int = 0
    unique_canonical: int = 0
    duplicates_collapsed: int = 0

    time_min_date: Optional[str] = None
    time_max_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def preprocess_corpus(
    input_path: str,
    output_path: str,
    dedupe_cfg: DedupeConfig,
    require_timestamp: bool,
    manifest_path: Optional[str],
    url_norm_cfg: UrlNormConfig,
    validation_rules: ValidationRules,
    progress_every: int = 50_000,
) -> PreprocessReport:
    report = PreprocessReport(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        started_at=now_utc_iso(),
        require_timestamp=require_timestamp,
        dedupe_strategy=dedupe_cfg.strategy,
    )

    print(hr("CORPUS PREPROCESSING"))
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if manifest_path:
        print(f"Manifest: {manifest_path}")
    print()

    # 1) Load JSONL robustly
    print("📂 Loading articles...")
    raw_articles: List[Dict[str, Any]] = []

    # Use utf-8-sig to eat BOM at file start.
    with open(input_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            report.loaded_lines += 1
            line = safe_strip_bom(line).strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    raw_articles.append(obj)
                    report.parsed_articles += 1
                else:
                    report.json_errors += 1
            except json.JSONDecodeError as e:
                report.json_errors += 1
                # Print only a few to avoid log spam
                if report.json_errors <= 5:
                    print(f"  ⚠️ Line {i}: JSON parse error: {e}")
                elif report.json_errors == 6:
                    print("  ⚠️ Further JSON errors suppressed...")
                continue

            if progress_every and (i % progress_every == 0):
                print(f"  … read {i:,} lines, parsed {report.parsed_articles:,} articles")

    print(f"  ✅ Loaded {len(raw_articles):,} articles")
    if report.json_errors:
        print(f"  ⚠️ JSON errors: {report.json_errors:,}")
    print()

    # 2) Validate + normalize fields + timestamps
    print("🔍 Validating articles...")
    normalized: List[Dict[str, Any]] = []
    for art in raw_articles:
        art2 = normalize_article_fields(art)
        ok, issues_delta = validate_article(art2, validation_rules)
        for k, v in issues_delta.items():
            report.issues[k] = report.issues.get(k, 0) + v
        if not ok:
            report.dropped_articles += 1
            continue
        report.valid_articles += 1

        # Timestamp accounting
        ts = art2.get("published_ts")
        if ts is None:
            # parse_published_at may have failed or missing
            if art2.get("published_at") is None:
                report.timestamp_missing += 1
            else:
                report.timestamp_invalid += 1
        else:
            report.timestamp_valid += 1

            dstr = art2.get("published_date")
            if isinstance(dstr, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", dstr):
                if report.time_min_date is None or dstr < report.time_min_date:
                    report.time_min_date = dstr
                if report.time_max_date is None or dstr > report.time_max_date:
                    report.time_max_date = dstr

        normalized.append(art2)

    print(f"  ✅ {report.valid_articles:,} valid articles")
    if report.dropped_articles:
        print(f"  ❌ Dropped: {report.dropped_articles:,}")
    if report.issues:
        print("  ⚠️ Issues found:")
        for k in sorted(report.issues.keys()):
            print(f"      {k}: {report.issues[k]:,}")
    print()

    if require_timestamp:
        print("⏰ Processing timestamps...")
        print(f"  ✅ Valid timestamps: {report.timestamp_valid:,}")
        print(f"  ⚠️ Missing: {report.timestamp_missing:,}")
        print(f"  ❌ Invalid: {report.timestamp_invalid:,}")
        # filter out missing/invalid if require_timestamp is set
        before = len(normalized)
        normalized = [a for a in normalized if a.get("published_ts") is not None]
        after = len(normalized)
        print(f"  🔸 Filtered to {after:,} articles with timestamps (from {before:,})")
        print()
    else:
        # no filter; still reporting
        pass

    # 3) Canonical IDs + dedupe
    print("🔑 Generating canonical IDs...")
    canon_cfg = CanonConfig(url_norm_cfg=url_norm_cfg)

    # If keep_all: still output with canonical_id but no dedupe.
    if dedupe_cfg.keep_all:
        output_rows: List[Dict[str, Any]] = []
        seen = set()
        for idx, a in enumerate(normalized, start=1):
            cid, id_info = generate_canonical_id(a, canon_cfg)
            report.canonical_generated += 1
            a["canonical_id"] = cid
            a["url_normalized"] = id_info.get("url_norm") or ""
            output_rows.append(a)
            seen.add(cid)
            if idx % 100_000 == 0:
                print(f"  … canonicalized {idx:,}")
        report.unique_canonical = len(seen)
        report.duplicates_collapsed = 0
    else:
        by_cid: Dict[str, Dict[str, Any]] = {}
        dupes = 0
        for idx, a in enumerate(normalized, start=1):
            cid, id_info = generate_canonical_id(a, canon_cfg)
            report.canonical_generated += 1
            a["canonical_id"] = cid
            a["url_normalized"] = id_info.get("url_norm") or ""

            if cid not in by_cid:
                by_cid[cid] = a
            else:
                dupes += 1
                by_cid[cid] = choose_better(by_cid[cid], a, dedupe_cfg.strategy)

            if idx % 100_000 == 0:
                print(f"  … canonicalized {idx:,}")

        output_rows = list(by_cid.values())
        report.unique_canonical = len(by_cid)
        report.duplicates_collapsed = dupes

    print(f"  ✅ Canonical IDs generated: {report.canonical_generated:,}")
    print(f"  ✅ Unique canonical IDs: {report.unique_canonical:,}")
    print(f"  🔁 Duplicates collapsed: {report.duplicates_collapsed:,} (strategy={dedupe_cfg.strategy})")
    print()

    # 4) Write output JSONL
    ensure_dir_for_file(output_path)
    print("💾 Writing output JSONL...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for a in output_rows:
            out_f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"  ✅ Wrote {len(output_rows):,} records to {output_path}")
    print()

    # 5) Manifest
    report.finished_at = now_utc_iso()
    if manifest_path:
        ensure_dir_for_file(manifest_path)
        manifest = report.to_dict()
        manifest["notes"] = {
            "schema": "Original fields preserved; added source, published_ts, published_date, canonical_id, url_normalized.",
            "dedupe": {
                "strategy": dedupe_cfg.strategy,
                "keep_all": dedupe_cfg.keep_all,
            },
            "url_normalization": dataclasses.asdict(url_norm_cfg),
            "generated_at": report.finished_at,
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        print("🧾 Manifest written.")
        print(f"  ✅ {manifest_path}")
        print()

    return report


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess a JSONL corpus: validate, normalize timestamps, generate canonical IDs, dedupe, output JSONL + manifest."
    )
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--output", required=True, help="Output JSONL path")

    p.add_argument(
        "--dedupe-strategy",
        default="first",
        choices=["first", "last", "longest_content"],
        help="How to choose representative record among duplicates by canonical_id."
    )
    p.add_argument(
        "--keep-all",
        action="store_true",
        help="Do not dedupe; still compute canonical_id for every record."
    )

    p.add_argument(
        "--require-timestamp",
        action="store_true",
        help="If set, drops any record without a valid published_at timestamp."
    )

    p.add_argument(
        "--manifest",
        default=None,
        help="If set, writes a JSON manifest report to this path."
    )

    # URL normalization tuning (performance knobs)
    p.add_argument("--max-query-params", type=int, default=DEFAULT_MAX_QUERY_PARAMS,
                   help="Max query params to keep; if exceeded, query is dropped.")
    p.add_argument("--max-url-len", type=int, default=DEFAULT_MAX_URL_LEN,
                   help="Max normalized URL length; if exceeded, query is dropped.")
    p.add_argument("--keep-www", action="store_true",
                   help="Do not strip www. from hostname during normalization.")

    # Validation knobs
    p.add_argument("--min-title-len", type=int, default=10,
                   help="Warn if title shorter than this length.")
    p.add_argument("--allow-missing-url", action="store_true",
                   help="If set, do not drop records missing url (not recommended).")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    url_cfg = UrlNormConfig(
        max_query_params=clamp_int(int(args.max_query_params), 0, 500),
        max_url_len=clamp_int(int(args.max_url_len), 256, 100_000),
        drop_query_if_too_many=True,
        strip_www=(not args.keep_www),
    )

    validation_rules = ValidationRules(
        min_title_len=clamp_int(int(args.min_title_len), 0, 10_000),
        require_url=(not args.allow_missing_url),
    )

    dedupe_cfg = DedupeConfig(
        strategy=args.dedupe_strategy,
        keep_all=bool(args.keep_all),
    )

    # Run
    try:
        preprocess_corpus(
            input_path=args.input,
            output_path=args.output,
            dedupe_cfg=dedupe_cfg,
            require_timestamp=bool(args.require_timestamp),
            manifest_path=args.manifest,
            url_norm_cfg=url_cfg,
            validation_rules=validation_rules,
        )
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user (KeyboardInterrupt).", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
