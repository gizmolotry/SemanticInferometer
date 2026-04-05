#!/usr/bin/env python3
"""
nli_probe.py

Minimal-pair probe harness for NLI models to measure compositional / order sensitivity.

This is designed to be called as a sidecar from run_full_experiment_suite.py.

CLI compatibility (do not change lightly):
  --model <hf_model>
  --hypotheses <json_file_list_of_strings>
  --entities <comma_list>                 (optional)
  --corpus-jsonl <path>                   (optional)
  --corpus-field <fieldname>              (default: text)
  --corpus-limit <int>                    (default: 200)
  --n-synth <int>                         (default: 50)
  --out <output.json>
  --max-length <int>                      (default: 256)
  --batch-size <int>                      (default: 16)

Outputs:
  JSON file with:
    config
    report (global + per transform x hypothesis aggregates)
    results (per-pair raw records)

Windows-safe:
  Forces UTF-8 stdout/stderr to prevent UnicodeEncodeError on cp1252 consoles.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------------------------------------------------------
# Unicode hardening
# -----------------------------------------------------------------------------

def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (getattr(__import__("sys"), "stdout"), getattr(__import__("sys"), "stderr")):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: For pure inference this is typically enough; we avoid extra knobs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Text perturbations
# -----------------------------------------------------------------------------

def safe_sent_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]

def sentence_shuffle(text: str, rng: random.Random) -> str:
    sents = safe_sent_split(text)
    if len(sents) <= 1:
        return text
    rng.shuffle(sents)
    return " ".join(sents)

def word_shuffle(text: str, rng: random.Random) -> str:
    # Tokenize into word tokens + punctuation tokens
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    word_idxs = [i for i, t in enumerate(tokens) if re.match(r"^\w+$", t)]
    word_tokens = [tokens[i] for i in word_idxs]
    rng.shuffle(word_tokens)
    for j, i in enumerate(word_idxs):
        tokens[i] = word_tokens[j]

    out = " ".join(tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# -----------------------------------------------------------------------------
# Probe definitions
# -----------------------------------------------------------------------------

@dataclass
class ProbePair:
    id: str
    transform: str
    premise_a: str
    premise_b: str
    expected_sign: Optional[int] = None  # optional directional expectation

def gen_synthetic_pairs(
    entities: List[str],
    rng: random.Random,
    n_per_type: int = 50
) -> List[ProbePair]:
    verbs = ["attacked", "bombed", "arrested", "accused", "rescued", "killed", "detained"]
    aux = ["did", "does", "will", "can", "should", "has", "have", "is", "are", "was", "were"]
    base_verbs = ["target", "harm", "support", "condemn", "approve", "deny", "strike"]
    quant_pairs = [("some", "all"), ("many", "few")]
    time_pairs = [("before", "after")]
    cause_pairs = [("because", "therefore")]

    def pick_two() -> Tuple[str, str]:
        if len(entities) < 2:
            raise ValueError("Need at least 2 entities for synthetic probes")
        a, b = rng.sample(entities, 2)
        return a, b

    pairs: List[ProbePair] = []

    # Agent/patient swap
    for i in range(n_per_type):
        A, B = pick_two()
        v = rng.choice(verbs)
        s1 = f"{A} {v} {B}."
        s2 = f"{B} {v} {A}."
        pairs.append(ProbePair(f"agent_swap_{i}", "agent_swap", s1, s2))

    # Negation flip (adds/removes "not")
    for i in range(n_per_type):
        A, _ = pick_two()
        a = rng.choice(aux)
        v = rng.choice(base_verbs)
        obj = rng.choice(["civilians", "the operation", "the strike", "the ceasefire", "the claim"])
        s1 = f"{A} {a} {v} {obj}."
        s2 = f"{A} {a} not {v} {obj}."
        pairs.append(ProbePair(f"negation_{i}", "negation_flip", s1, s2))

    # Quantifier flip
    for i in range(n_per_type):
        A, B = pick_two()
        q1, q2 = rng.choice(quant_pairs)
        outcome = rng.choice(["were harmed", "were displaced", "were arrested", "were injured"])
        s1 = f"{q1.capitalize()} people in {B} {outcome} after actions by {A}."
        s2 = f"{q2.capitalize()} people in {B} {outcome} after actions by {A}."
        pairs.append(ProbePair(f"quant_{i}", "quantifier_flip", s1, s2))

    # Temporal flip
    for i in range(n_per_type):
        _, B = pick_two()
        w1, w2 = rng.choice(time_pairs)
        e1 = rng.choice(["The raid", "The strike", "The statement", "The vote"])
        e2 = rng.choice(["the protests", "the evacuation", "the ceasefire", "the negotiations"])
        s1 = f"{e1} happened {w1} {e2} in {B}."
        s2 = f"{e1} happened {w2} {e2} in {B}."
        pairs.append(ProbePair(f"time_{i}", "temporal_flip", s1, s2))

    # Causal polarity (crude)
    for i in range(n_per_type):
        A, B = pick_two()
        w1, w2 = rng.choice(cause_pairs)
        e1 = rng.choice(["Protests erupted", "Negotiations stalled", "Violence escalated"])
        e2 = rng.choice(["the raid", "the strike", "the announcement"])
        s1 = f"{e1} {w1} of {e2} by {A} in {B}."
        s2 = f"{e1} {w2} of {e2} by {A} in {B}."
        pairs.append(ProbePair(f"cause_{i}", "causal_flip", s1, s2))

    return pairs

def gen_corpus_perturbation_pairs(
    premises: List[str],
    rng: random.Random,
    n: int = 200
) -> List[ProbePair]:
    pairs: List[ProbePair] = []
    sample = premises if len(premises) <= n else rng.sample(premises, n)

    for i, p in enumerate(sample):
        pairs.append(ProbePair(f"sent_shuffle_{i}", "sentence_shuffle", p, sentence_shuffle(p, rng)))
        pairs.append(ProbePair(f"word_shuffle_{i}", "word_shuffle", p, word_shuffle(p, rng)))
    return pairs


# -----------------------------------------------------------------------------
# NLI execution
# -----------------------------------------------------------------------------

def infer_label_map(id2label: Dict[int, str]) -> Dict[str, int]:
    # Try to infer from config strings; fallback to MNLI common ordering.
    norm = {i: str(lbl).lower() for i, lbl in (id2label or {}).items()}
    ent = neu = con = None
    for i, lbl in norm.items():
        if "entail" in lbl:
            ent = i
        elif "neutral" in lbl:
            neu = i
        elif "contrad" in lbl:
            con = i
    if ent is not None and con is not None:
        return {"contradiction": con, "neutral": neu if neu is not None else 1, "entailment": ent}
    # default MNLI: [contradiction, neutral, entailment]
    return {"contradiction": 0, "neutral": 1, "entailment": 2}

def run_nli_batch(
    model,
    tokenizer,
    premises: List[str],
    hypotheses: List[str],
    device: str,
    max_length: int,
    batch_size: int
) -> torch.Tensor:
    """
    Returns probabilities with shape [len(premises), len(hypotheses), num_labels]
    """
    model.eval()
    num_labels = model.config.num_labels
    out = torch.empty((len(premises), len(hypotheses), num_labels), dtype=torch.float32)

    with torch.no_grad():
        for h_idx, hyp in enumerate(hypotheses):
            for start in range(0, len(premises), batch_size):
                p_chunk = premises[start:start + batch_size]
                h_chunk = [hyp] * len(p_chunk)

                enc = tokenizer(
                    p_chunk,
                    h_chunk,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                out[start:start + len(p_chunk), h_idx, :] = probs

    return out


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

@dataclass
class PairResult:
    pair_id: str
    transform: str
    hypothesis: str
    probs_a: List[float]
    probs_b: List[float]
    pred_a: int
    pred_b: int
    margin_a: float
    margin_b: float
    delta_margin: float
    flipped: bool

def score_pairs(
    pairs: List[ProbePair],
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    hypotheses: List[str],
    label_map: Dict[str, int],
) -> List[PairResult]:
    ent_i = label_map["entailment"]
    con_i = label_map["contradiction"]

    results: List[PairResult] = []
    for i, pair in enumerate(pairs):
        for h_idx, hyp in enumerate(hypotheses):
            pa = probs_a[i, h_idx, :]
            pb = probs_b[i, h_idx, :]

            pred_a = int(torch.argmax(pa).item())
            pred_b = int(torch.argmax(pb).item())

            margin_a = float(pa[ent_i] - pa[con_i])
            margin_b = float(pb[ent_i] - pb[con_i])
            delta = margin_b - margin_a

            results.append(PairResult(
                pair_id=pair.id,
                transform=pair.transform,
                hypothesis=hyp,
                probs_a=[float(x) for x in pa.tolist()],
                probs_b=[float(x) for x in pb.tolist()],
                pred_a=pred_a,
                pred_b=pred_b,
                margin_a=margin_a,
                margin_b=margin_b,
                delta_margin=delta,
                flipped=(pred_a != pred_b),
            ))
    return results

def aggregate_report(results: List[PairResult]) -> Dict:
    from collections import defaultdict
    import statistics

    def mean(xs: List[float]) -> float:
        return float(statistics.mean(xs)) if xs else float("nan")

    def median(xs: List[float]) -> float:
        return float(statistics.median(xs)) if xs else float("nan")

    bucket = defaultdict(list)
    for r in results:
        bucket[(r.transform, r.hypothesis)].append(r)

    all_deltas = [abs(r.delta_margin) for r in results]
    all_flips = [1.0 if r.flipped else 0.0 for r in results]

    report = {
        "global": {
            "n_results": len(results),
            "overall_flip_rate": mean(all_flips),
            "overall_median_abs_delta_margin": median(all_deltas),
            "overall_mean_abs_delta_margin": mean(all_deltas),
            "overall_frac_large_shift_0.2": mean([1.0 if d > 0.2 else 0.0 for d in all_deltas]),
        },
        "by_transform_hypothesis": {}
    }

    for (t, h), rs in bucket.items():
        deltas = [abs(r.delta_margin) for r in rs]
        flips = [1.0 if r.flipped else 0.0 for r in rs]
        report["by_transform_hypothesis"][f"{t}|||{h}"] = {
            "transform": t,
            "hypothesis": h,
            "n": len(rs),
            "flip_rate": mean(flips),
            "median_abs_delta_margin": median(deltas),
            "mean_abs_delta_margin": mean(deltas),
            "frac_large_shift_0.2": mean([1.0 if d > 0.2 else 0.0 for d in deltas]),
        }

    return report


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_jsonl_texts(path: str, field: str = "text", limit: int = 200) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            val = obj.get(field, "")
            if val:
                texts.append(str(val))
            if limit and len(texts) >= limit:
                break
    return texts


# -----------------------------------------------------------------------------
# Public callable API (if you ever import it)
# -----------------------------------------------------------------------------

def run_probe(
    model_name: str,
    hypotheses: List[str],
    entities: Optional[List[str]] = None,
    corpus_premises: Optional[List[str]] = None,
    n_synth_per_type: int = 50,
    n_corpus: int = 200,
    seed: int = 42,
    device: Optional[str] = None,
    max_length: int = 256,
    batch_size: int = 16,
) -> Dict:
    set_deterministic(seed)
    rng = random.Random(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    label_map = infer_label_map(getattr(model.config, "id2label", {}))

    pairs: List[ProbePair] = []
    if entities:
        pairs.extend(gen_synthetic_pairs(entities, rng, n_per_type=n_synth_per_type))
    if corpus_premises:
        pairs.extend(gen_corpus_perturbation_pairs(corpus_premises, rng, n=n_corpus))

    if not pairs:
        raise ValueError("No pairs generated: provide --entities and/or --corpus-jsonl")

    premises_a = [p.premise_a for p in pairs]
    premises_b = [p.premise_b for p in pairs]

    probs_a = run_nli_batch(model, tokenizer, premises_a, hypotheses, device, max_length, batch_size)
    probs_b = run_nli_batch(model, tokenizer, premises_b, hypotheses, device, max_length, batch_size)

    results = score_pairs(pairs, probs_a, probs_b, hypotheses, label_map)
    report = aggregate_report(results)

    return {
        "config": {
            "model_name": model_name,
            "device": device,
            "seed": seed,
            "max_length": max_length,
            "batch_size": batch_size,
            "label_map": label_map,
            "n_pairs": len(pairs),
            "n_hypotheses": len(hypotheses),
        },
        "report": report,
        "results": [asdict(r) for r in results],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    _force_utf8_stdio()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF NLI model name (e.g., roberta-large-mnli)")
    ap.add_argument("--hypotheses", required=True, help="JSON file containing a list of hypothesis strings")
    ap.add_argument("--entities", default="", help="Comma-separated entities for synthetic probes (optional)")
    ap.add_argument("--corpus-jsonl", default="", help="Optional JSONL file of premises for shuffle probes")
    ap.add_argument("--corpus-field", default="text", help="Field in JSONL (default: text)")
    ap.add_argument("--corpus-limit", type=int, default=200, help="Premises to sample for shuffle probes (default: 200)")
    ap.add_argument("--n-synth", type=int, default=50, help="Pairs per synthetic transform (default: 50)")
    ap.add_argument("--out", default="nli_probe_results.json", help="Output JSON file")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.hypotheses, "r", encoding="utf-8") as f:
        hypotheses = json.load(f)
    if not isinstance(hypotheses, list) or not all(isinstance(x, str) for x in hypotheses):
        raise ValueError("--hypotheses must be a JSON list of strings")

    entities = [e.strip() for e in args.entities.split(",") if e.strip()] if args.entities else None
    corpus = load_jsonl_texts(args.corpus_jsonl, field=args.corpus_field, limit=args.corpus_limit) if args.corpus_jsonl else None

    out = run_probe(
        model_name=args.model,
        hypotheses=hypotheses,
        entities=entities,
        corpus_premises=corpus,
        n_synth_per_type=args.n_synth,
        n_corpus=args.corpus_limit,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Compact summary (ASCII only)
    g = out["report"]["global"]
    print("GLOBAL")
    for k in ("n_results", "overall_flip_rate", "overall_median_abs_delta_margin", "overall_mean_abs_delta_margin", "overall_frac_large_shift_0.2"):
        print(f"  {k}: {g.get(k)}")

    print("\nTOP signals (median_abs_delta_margin)")
    items = list(out["report"]["by_transform_hypothesis"].values())
    items.sort(key=lambda x: x["median_abs_delta_margin"], reverse=True)
    for it in items[:12]:
        hyp = it["hypothesis"]
        print(f'  {it["transform"]:<16} median|dMargin|={it["median_abs_delta_margin"]:.4f} flip={it["flip_rate"]:.3f} n={it["n"]} hyp="{hyp[:60]}..."')


if __name__ == "__main__":
    main()
