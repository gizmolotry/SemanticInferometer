#!/usr/bin/env python3
"""Compare Track 1.5 forward deltas against gradient sensitivity vectors.

This is a contained ablation probe. It does not alter the production pipeline.
It asks whether gradients through a frozen NLI model provide a cleaner observer
shear basis than the current A-minus-B mean-pooled embedding deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data_utils import extract_article_text  # noqa: E402


DEFAULT_CORPORA = {
    "real": REPO_ROOT / "data" / "real_corpus.jsonl",
    "constant": REPO_ROOT / "data" / "control_constant.jsonl",
    "shuffled": REPO_ROOT / "data" / "control_shuffled.jsonl",
    "random": REPO_ROOT / "data" / "control_random.jsonl",
}


@dataclass(frozen=True)
class ProbeConfig:
    model_name: str
    queries_config: str
    limit: int
    max_length: int
    device: str
    seed: int
    corpora: Tuple[str, ...]
    methods: Tuple[str, ...]


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                break
    return rows


def _load_hypothesis_pairs(path: Path) -> List[Tuple[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    for row in (payload or {}).get("queries", {}).values():
        if isinstance(row, dict) and row.get("A") and row.get("B"):
            pairs.append((str(row["A"]), str(row["B"])))
    if not pairs:
        raise ValueError(f"No hypothesis pairs found in {path}")
    return pairs


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return (last_hidden * mask).sum(dim=1) / denom


def _pool_token_gradient(grad_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=grad_hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return (grad_hidden * mask).sum(dim=1) / denom


def _as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [x for x in (_as_float(v) for v in values) if x is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = np.asarray([x for x in (_as_float(v) for v in values) if x is not None], dtype=np.float64)
    return float(np.std(vals)) if vals.size else None


def _cosine_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x = x / norms
    return x @ x.T


def _pairwise_cosine_values(features: np.ndarray) -> List[float]:
    if features.shape[0] < 2:
        return []
    sim = _cosine_matrix(features)
    return [float(sim[i, j]) for i, j in combinations(range(features.shape[0]), 2)]


def _pairwise_cosine_distance_values(features: np.ndarray) -> List[float]:
    return [1.0 - v for v in _pairwise_cosine_values(features)]


def _effective_rank(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    energy = singular**2
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    p = energy / total
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
    return float(np.exp(entropy))


def _feature_metrics(features: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    pairwise_dist = _pairwise_cosine_distance_values(flat)
    observer_cosines: List[float] = []
    observer_abs_cosines: List[float] = []
    observer_norms: List[float] = []
    if x.ndim == 3:
        for article_features in x:
            vals = _pairwise_cosine_values(article_features)
            observer_cosines.extend(vals)
            observer_abs_cosines.extend(abs(v) for v in vals)
            observer_norms.extend(float(np.linalg.norm(v)) for v in article_features)
    return {
        "shape": list(x.shape),
        "flat_simple_variance": float(np.var(flat)),
        "flat_mean_l2_norm": float(np.mean(np.linalg.norm(flat, axis=1))),
        "flat_pairwise_cosine_distance_mean": _mean(pairwise_dist),
        "flat_pairwise_cosine_distance_std": _std(pairwise_dist),
        "flat_effective_rank": _effective_rank(flat),
        "observer_pairwise_cosine_mean": _mean(observer_cosines),
        "observer_pairwise_abs_cosine_mean": _mean(observer_abs_cosines),
        "observer_norm_mean": _mean(observer_norms),
        "observer_norm_std": _std(observer_norms),
    }


def _safe_ratio(real_value: Optional[float], control_value: Optional[float]) -> Optional[float]:
    if real_value is None or control_value is None or abs(control_value) <= 1e-12:
        return None
    return float(real_value / control_value)


def _safe_abs_log_ratio(real_value: Optional[float], control_value: Optional[float]) -> Optional[float]:
    if real_value is None or control_value is None or real_value <= 0 or control_value <= 0:
        return None
    return float(abs(math.log(real_value / control_value)))


class Track15GradientProbe:
    def __init__(self, config: ProbeConfig):
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.config = config
        requested = torch.device(config.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entailment_idx = self._infer_entailment_index()
        self.hypothesis_pairs = _load_hypothesis_pairs(REPO_ROOT / config.queries_config)

    def _infer_entailment_index(self) -> int:
        for idx, label in self.id2label.items():
            if "entail" in label:
                return int(idx)
        return 2

    def _encode_inputs(self, premise: str, hypothesis: str) -> Mapping[str, torch.Tensor]:
        return self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.config.max_length,
            padding=True,
        ).to(self.device)

    def _anchor_vector(self, hypothesis: str) -> torch.Tensor:
        inputs = self.tokenizer(
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.config.max_length, 128),
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            pooled = _mean_pool(outputs.hidden_states[-1], inputs["attention_mask"])
            return F.normalize(pooled, p=2, dim=-1).detach()

    def _encode_forward_vector(self, premise: str, hypothesis: str) -> torch.Tensor:
        with torch.no_grad():
            inputs = self._encode_inputs(premise, hypothesis)
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            return _mean_pool(outputs.hidden_states[-1], inputs["attention_mask"]).squeeze(0).detach().cpu()

    def _logit_gradient_vector(self, premise: str, hypothesis: str) -> torch.Tensor:
        inputs = self._encode_inputs(premise, hypothesis)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        score = outputs.logits[:, self.entailment_idx].sum()
        grad_hidden = torch.autograd.grad(score, last_hidden, retain_graph=False, create_graph=False)[0]
        pooled_grad = _pool_token_gradient(grad_hidden, inputs["attention_mask"])
        return pooled_grad.squeeze(0).detach().cpu()

    def _anchor_gradient_vector(self, premise: str, hypothesis: str, anchor: torch.Tensor) -> torch.Tensor:
        inputs = self._encode_inputs(premise, hypothesis)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        pooled = _mean_pool(outputs.hidden_states[-1], inputs["attention_mask"])
        pooled_norm = F.normalize(pooled, p=2, dim=-1)
        score = F.cosine_similarity(pooled_norm, anchor.to(self.device), dim=-1).sum()
        grad_pooled = torch.autograd.grad(score, pooled, retain_graph=False, create_graph=False)[0]
        return grad_pooled.squeeze(0).detach().cpu()

    def extract_article_features(self, text: str) -> Dict[str, np.ndarray]:
        forward_rows: List[torch.Tensor] = []
        logit_grad_rows: List[torch.Tensor] = []
        anchor_grad_rows: List[torch.Tensor] = []

        for hyp_a, hyp_b in self.hypothesis_pairs:
            if "forward_delta" in self.config.methods:
                forward_a = self._encode_forward_vector(text, hyp_a)
                forward_b = self._encode_forward_vector(text, hyp_b)
                forward_rows.append(forward_a - forward_b)
            if "logit_gradient_delta" in self.config.methods:
                grad_a = self._logit_gradient_vector(text, hyp_a)
                grad_b = self._logit_gradient_vector(text, hyp_b)
                logit_grad_rows.append(grad_a - grad_b)
            if "anchor_gradient_delta" in self.config.methods:
                anchor_a = self._anchor_vector(hyp_a)
                anchor_b = self._anchor_vector(hyp_b)
                anchor_grad_a = self._anchor_gradient_vector(text, hyp_a, anchor_a)
                anchor_grad_b = self._anchor_gradient_vector(text, hyp_b, anchor_b)
                anchor_grad_rows.append(anchor_grad_a - anchor_grad_b)

        payload: Dict[str, np.ndarray] = {}
        if forward_rows:
            payload["forward_delta"] = torch.stack(forward_rows, dim=0).numpy().astype(np.float32)
        if logit_grad_rows:
            payload["logit_gradient_delta"] = torch.stack(logit_grad_rows, dim=0).numpy().astype(np.float32)
        if anchor_grad_rows:
            payload["anchor_gradient_delta"] = torch.stack(anchor_grad_rows, dim=0).numpy().astype(np.float32)
        return payload


def run_probe(config: ProbeConfig, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = Track15GradientProbe(config)
    corpora = {name: DEFAULT_CORPORA[name] for name in config.corpora}
    article_rows: Dict[str, List[Dict[str, Any]]] = {}
    features_by_method: Dict[str, Dict[str, np.ndarray]] = {method: {} for method in config.methods}
    per_article_rows: List[Dict[str, Any]] = []
    start = time.time()

    for corpus_name, corpus_path in corpora.items():
        rows = _load_jsonl(corpus_path, config.limit)
        article_rows[corpus_name] = rows
        method_features: Dict[str, List[np.ndarray]] = {method: [] for method in config.methods}
        for idx, row in enumerate(rows):
            text = extract_article_text(row)
            if not text.strip():
                text = str(row.get("content") or row.get("text") or row.get("title") or "")
            print(f"[{corpus_name}] {idx + 1}/{len(rows)} chars={len(text)}")
            extracted = probe.extract_article_features(text)
            for method, feature in extracted.items():
                method_features[method].append(feature)
                per_article_rows.append(
                    {
                        "corpus": corpus_name,
                        "article_index": idx,
                        "method": method,
                        "feature_l2_norm": float(np.linalg.norm(feature.reshape(-1))),
                        "observer_norm_mean": float(np.mean(np.linalg.norm(feature, axis=1))),
                    }
                )
        for method, rows_for_method in method_features.items():
            if rows_for_method:
                features_by_method[method][corpus_name] = np.stack(rows_for_method, axis=0)

    corpus_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    cartography_rows: List[Dict[str, Any]] = []
    metric_names = [
        "flat_simple_variance",
        "flat_mean_l2_norm",
        "flat_pairwise_cosine_distance_mean",
        "flat_effective_rank",
        "observer_pairwise_abs_cosine_mean",
        "observer_norm_mean",
    ]
    for method, by_corpus in features_by_method.items():
        corpus_metrics[method] = {}
        for corpus_name, features in by_corpus.items():
            corpus_metrics[method][corpus_name] = _feature_metrics(features)
        real_metrics = corpus_metrics[method].get("real", {})
        for corpus_name, metrics in corpus_metrics[method].items():
            if corpus_name == "real":
                continue
            for metric_name in metric_names:
                real_value = _as_float(real_metrics.get(metric_name))
                control_value = _as_float(metrics.get(metric_name))
                cartography_rows.append(
                    {
                        "method": method,
                        "metric": metric_name,
                        "control": corpus_name,
                        "real_value": real_value,
                        "control_value": control_value,
                        "ratio_real_over_control": _safe_ratio(real_value, control_value),
                        "abs_log_ratio": _safe_abs_log_ratio(real_value, control_value),
                    }
                )

    method_scores: Dict[str, Dict[str, Any]] = {}
    for method in config.methods:
        rows = [r for r in cartography_rows if r["method"] == method]
        stochastic_rows = [r for r in rows if r["control"] in {"shuffled", "random"}]
        method_scores[method] = {
            "mean_abs_log_ratio_all_controls": _mean(r["abs_log_ratio"] for r in rows),
            "mean_abs_log_ratio_stochastic_controls": _mean(r["abs_log_ratio"] for r in stochastic_rows),
            "real_vs_random_variance_ratio": next(
                (
                    r["ratio_real_over_control"]
                    for r in rows
                    if r["control"] == "random" and r["metric"] == "flat_simple_variance"
                ),
                None,
            ),
            "real_vs_shuffled_variance_ratio": next(
                (
                    r["ratio_real_over_control"]
                    for r in rows
                    if r["control"] == "shuffled" and r["metric"] == "flat_simple_variance"
                ),
                None,
            ),
            "observer_abs_cosine_real": corpus_metrics.get(method, {}).get("real", {}).get(
                "observer_pairwise_abs_cosine_mean"
            ),
        }

    output_npz = output_dir / "track15_gradient_probe_features.npz"
    npz_payload: Dict[str, np.ndarray] = {}
    for method, by_corpus in features_by_method.items():
        for corpus_name, features in by_corpus.items():
            npz_payload[f"{method}__{corpus_name}"] = features
    np.savez_compressed(output_npz, **npz_payload)

    with (output_dir / "metric_cartography.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "metric",
            "control",
            "real_value",
            "control_value",
            "ratio_real_over_control",
            "abs_log_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cartography_rows)

    with (output_dir / "per_article_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["corpus", "article_index", "method", "feature_l2_norm", "observer_norm_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_article_rows)

    summary = {
        "schema_version": "1.0",
        "summary_type": "track15_gradient_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.time() - start),
        "config": asdict(config),
        "device_used": str(probe.device),
        "model_id2label": probe.id2label,
        "entailment_idx": probe.entailment_idx,
        "corpus_sizes": {name: len(rows) for name, rows in article_rows.items()},
        "corpus_metrics": corpus_metrics,
        "metric_cartography": cartography_rows,
        "method_scores": method_scores,
        "interpretation": {
            "primary_question": "Do gradient sensitivity vectors improve observer shear separation without control chaos?",
            "readout": "Higher stochastic abs-log-ratio means stronger real/control displacement; lower real observer_abs_cosine means more orthogonal observer pulls.",
            "caveat": "This is a small ablation probe, not a production Track 1.5 replacement.",
        },
        "artifacts": {
            "features_npz": str(output_npz),
            "metric_cartography_csv": str(output_dir / "metric_cartography.csv"),
            "per_article_metrics_csv": str(output_dir / "per_article_metrics.csv"),
        },
    }
    (output_dir / "track15_gradient_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--queries-config", default="config/framing_queries.yaml")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=["real", "constant", "shuffled", "random"],
        choices=sorted(DEFAULT_CORPORA),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["forward_delta", "logit_gradient_delta", "anchor_gradient_delta"],
        choices=["forward_delta", "logit_gradient_delta", "anchor_gradient_delta"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "track15_gradient_probe" / f"probe_{_utc_slug()}_n{args.limit}"
    )
    config = ProbeConfig(
        model_name=args.model,
        queries_config=args.queries_config,
        limit=args.limit,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
        corpora=tuple(args.corpora),
        methods=tuple(args.methods),
    )
    summary = run_probe(config, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "runtime_seconds": summary["runtime_seconds"],
        "method_scores": summary["method_scores"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
