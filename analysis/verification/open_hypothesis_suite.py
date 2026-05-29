from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from core.ablation_dag import AblationTaskGraph, build_orchestration_contract


OPEN_HYPOTHESIS_SCHEMA_VERSION = "1.0"

SUPPORTED = "supported"
PARTIAL = "partially_supported"
BLOCKED = "blocked"
UNSUPPORTED = "unsupported"
UNPROVEN = "unproven"

HYPOTHESIS_IDS = (
    "observer_recentering_robustness",
    "simple_baseline_superiority",
    "real_corpus_ideological_accuracy",
    "terrain_semantics_software",
    "track4_publishable_core",
    "property_theft_real_scale",
    "prompt_set_invariance",
    "track3_density_semantic_interpretation",
    "visualization_human_readability",
)

HYPOTHESIS_LABELS = {
    "observer_recentering_robustness": "Observer recentering is robust across kernels/seeds",
    "simple_baseline_superiority": "Observer-local recompute beats simpler geometry",
    "real_corpus_ideological_accuracy": "Real-corpus ideological labels are correct",
    "terrain_semantics_software": "Bridge/Swamp/Tightrope/Void has software-level terrain meaning",
    "track4_publishable_core": "Track 4 is ready as a core publishable mechanism",
    "property_theft_real_scale": "Property/theft holonomy survives beyond the toy probe",
    "prompt_set_invariance": "Results survive alternate V-observer prompt sets",
    "track3_density_semantic_interpretation": "Track 3 density has independent semantic interpretation",
    "visualization_human_readability": "The Dash/atlas view is human-interpretable",
}


@dataclass(frozen=True)
class OpenHypothesisConfig:
    repo_root: Path
    output_dir: Path
    claim_matrix_path: Optional[Path] = None
    recentering_summary_path: Optional[Path] = None
    track4_observer_state_summary_path: Optional[Path] = None
    property_theft_transport_summary_path: Optional[Path] = None
    independent_label_summary_path: Optional[Path] = None
    prompt_invariance_summary_path: Optional[Path] = None
    track3_density_validation_path: Optional[Path] = None
    visualization_validation_path: Optional[Path] = None
    min_real_scale_property_records: int = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _latest_existing(paths: Iterable[Path]) -> Optional[Path]:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _latest_glob(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    return _latest_existing(root.glob(pattern))


def _default_artifact_paths(repo_root: Path) -> Dict[str, Optional[Path]]:
    outputs = repo_root / "outputs"
    return {
        "claim_matrix_path": _latest_glob(outputs / "thesis_validation", "**/claim_matrix.json"),
        "recentering_summary_path": _latest_existing(
            [
                outputs
                / "observer_recenter_robustness_suite"
                / "variant_sweep_20260528"
                / "observer_recenter_robustness_suite.json",
                outputs
                / "observer_recenter_robustness_suite"
                / "robustness_3x3_baselines_20260528"
                / "observer_recenter_robustness_suite.json",
            ]
        )
        or _latest_glob(
            outputs / "observer_recenter_robustness_suite",
            "**/observer_recenter_robustness_suite.json",
        ),
        "track4_observer_state_summary_path": _latest_existing(
            [
                outputs
                / "thesis_validation"
                / "observer_state_action_3seed_20260521"
                / "track4_observer_state_action_summary.json",
                outputs
                / "thesis_validation"
                / "focused_paper_evidence_20260522"
                / "track4_observer_state_action_summary.json",
            ]
        )
        or _latest_glob(outputs / "thesis_validation", "**/track4_observer_state_action_summary.json"),
        "property_theft_transport_summary_path": _latest_glob(
            outputs / "track4_observer_slice_transport_probe",
            "**/observer_slice_transport_summary.json",
        ),
    }


def _claim_by_id(claim_matrix: Mapping[str, Any], claim_id: str) -> Dict[str, Any]:
    for row in claim_matrix.get("claims", []) or []:
        if isinstance(row, Mapping) and str(row.get("claim_id")) == claim_id:
            return dict(row)
    return {}


def _summary_path(payload: Mapping[str, Any], key: str) -> Optional[str]:
    paths = payload.get("artifact_paths") or {}
    if isinstance(paths, Mapping):
        value = paths.get(key)
        return str(value) if value else None
    return None


def _evidence_row(
    *,
    hypothesis_id: str,
    status: str,
    answer: str,
    claim_scope: str,
    publication_safe: bool,
    evidence: Optional[Mapping[str, Any]] = None,
    blockers: Optional[Sequence[str]] = None,
    next_tests: Optional[Sequence[str]] = None,
    artifact_paths: Optional[Sequence[str | Path]] = None,
) -> Dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "label": HYPOTHESIS_LABELS[hypothesis_id],
        "status": status,
        "publication_safe": bool(publication_safe),
        "claim_scope": claim_scope,
        "answer": answer,
        "evidence": _jsonable(dict(evidence or {})),
        "blockers": list(blockers or []),
        "next_tests": list(next_tests or []),
        "artifact_paths": [str(Path(path)) for path in (artifact_paths or []) if path],
    }


def _aggregate_entries(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    aggregate = summary.get("aggregate")
    if isinstance(aggregate, Mapping):
        return {str(k): dict(v) for k, v in aggregate.items() if isinstance(v, Mapping)}
    return {}


def _pass_count(row: Mapping[str, Any]) -> int:
    return _to_int(row.get("pass_count"), default=0)


def _cell_count(row: Mapping[str, Any], fallback: int) -> int:
    value = _to_int(row.get("cell_count"), default=0)
    return value if value > 0 else fallback


def _mean_primary_gain(row: Mapping[str, Any]) -> Optional[float]:
    return _to_float(
        row.get("mean_primary_label_gain")
        if row.get("mean_primary_label_gain") is not None
        else row.get("mean_primary_gain")
    )


def _evaluate_observer_recentering_robustness(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _load_json(payload.get("recentering_summary_path"))
    path = _summary_path(payload, "recentering_summary_path")
    if not summary:
        return _evidence_row(
            hypothesis_id="observer_recentering_robustness",
            status=UNPROVEN,
            claim_scope="missing_robustness_matrix",
            publication_safe=False,
            answer="No observer-recentering robustness artifact was found.",
            blockers=["missing observer_recenter_robustness_suite.json"],
            next_tests=["Run the 3x3 recentering robustness suite with simple baselines."],
        )
    aggregate = _aggregate_entries(summary)
    cell_count = _to_int(summary.get("cell_count"), default=0)
    local_rows = {
        name: row for name, row in aggregate.items() if name.startswith("local_track_recompute")
    }
    best_name = None
    best_row: Dict[str, Any] = {}
    if local_rows:
        best_name, best_row = max(
            local_rows.items(),
            key=lambda item: (_pass_count(item[1]), _mean_primary_gain(item[1]) or -1.0),
        )
    best_pass_count = _pass_count(best_row)
    best_cell_count = _cell_count(best_row, cell_count)
    artifact_pass = _pass_count(aggregate.get("artifact_view", {}))
    translation_pass = _pass_count(aggregate.get("translation_only", {}))
    if best_cell_count > 0 and best_pass_count == best_cell_count:
        status = SUPPORTED
        answer = "Observer-local recompute is robust in every registered cell."
        safe = True
    elif best_pass_count > 0 and artifact_pass == 0 and translation_pass == 0:
        status = PARTIAL
        answer = (
            "Observer-local recompute is real and beats visual/translation controls, "
            "but it is not universal across the current robustness grid."
        )
        safe = True
    else:
        status = UNSUPPORTED
        answer = "Observer-local recompute does not clear the robustness controls."
        safe = False
    return _evidence_row(
        hypothesis_id="observer_recentering_robustness",
        status=status,
        publication_safe=safe,
        claim_scope="controlled_synthetic_recenter_robustness",
        answer=answer,
        evidence={
            "cell_count": cell_count,
            "best_local_baseline": best_name,
            "best_local_pass_count": best_pass_count,
            "best_local_cell_count": best_cell_count,
            "translation_only_pass_count": translation_pass,
            "artifact_view_pass_count": artifact_pass,
            "best_local_mean_primary_gain": _mean_primary_gain(best_row),
        },
        blockers=[] if status == SUPPORTED else ["not all kernel/seed cells pass"],
        next_tests=[
            "Rerun uniform-weighted local recompute on a larger synthetic corpus.",
            "Inspect weak anchors in failed IMQ/RBF cells.",
        ],
        artifact_paths=[path] if path else [],
    )


def _evaluate_simple_baseline_superiority(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _load_json(payload.get("recentering_summary_path"))
    path = _summary_path(payload, "recentering_summary_path")
    aggregate = _aggregate_entries(summary)
    if not aggregate:
        return _evidence_row(
            hypothesis_id="simple_baseline_superiority",
            status=UNPROVEN,
            publication_safe=False,
            claim_scope="missing_baseline_matrix",
            answer="No baseline comparison matrix was found.",
            blockers=["missing simple baseline rows"],
            next_tests=["Run local recompute against raw Track 2, CLS/PCA, translation, and artifact baselines."],
        )
    local_rows = {
        name: row for name, row in aggregate.items() if name.startswith("local_track_recompute")
    }
    simple_rows = {
        name: row
        for name, row in aggregate.items()
        if name not in local_rows
        and name not in {"translation_only", "artifact_view"}
    }
    if not local_rows or not simple_rows:
        return _evidence_row(
            hypothesis_id="simple_baseline_superiority",
            status=UNPROVEN,
            publication_safe=False,
            claim_scope="incomplete_baseline_matrix",
            answer="The matrix does not contain both local recompute and serious simple baselines.",
            blockers=["missing local or simple baselines"],
            next_tests=["Add raw_track2_pca and cls_mean_pca rows to the robustness suite."],
            artifact_paths=[path] if path else [],
        )
    best_local_name, best_local = max(
        local_rows.items(),
        key=lambda item: (_pass_count(item[1]), _mean_primary_gain(item[1]) or -1.0),
    )
    best_simple_name, best_simple = max(
        simple_rows.items(),
        key=lambda item: (_pass_count(item[1]), _mean_primary_gain(item[1]) or -1.0),
    )
    local_pass = _pass_count(best_local)
    simple_pass = _pass_count(best_simple)
    local_gain = _mean_primary_gain(best_local)
    simple_gain = _mean_primary_gain(best_simple)
    if local_pass > simple_pass:
        status = SUPPORTED
        answer = "The best local recompute branch beats the best simple baseline on strict pass count."
        safe = True
    elif local_pass == simple_pass and (local_gain or -1.0) >= (simple_gain or -1.0):
        status = PARTIAL
        answer = (
            "The best local recompute branch matches, rather than dominates, the strongest simple baseline. "
            "This proves mechanism, not architectural supremacy."
        )
        safe = True
    else:
        status = UNSUPPORTED
        answer = "A simple geometry baseline currently beats or matches local recompute without a compensating gain."
        safe = False
    return _evidence_row(
        hypothesis_id="simple_baseline_superiority",
        status=status,
        publication_safe=safe,
        claim_scope="controlled_baseline_comparison",
        answer=answer,
        evidence={
            "best_local_baseline": best_local_name,
            "best_local_pass_count": local_pass,
            "best_local_mean_primary_gain": local_gain,
            "best_simple_baseline": best_simple_name,
            "best_simple_pass_count": simple_pass,
            "best_simple_mean_primary_gain": simple_gain,
        },
        blockers=[] if status == SUPPORTED else ["simple PCA baselines remain serious competitors"],
        next_tests=[
            "Promote only the mechanism claim unless larger runs show clear dominance.",
            "Run the same baseline matrix on a larger planted synthetic corpus.",
        ],
        artifact_paths=[path] if path else [],
    )


def _evaluate_real_corpus_ideological_accuracy(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _load_json(payload.get("independent_label_summary_path"))
    path = _summary_path(payload, "independent_label_summary_path")
    validated = bool(summary.get("independent_labels_validated"))
    unit = str(summary.get("unit_of_analysis") or "")
    metric_pass = bool(summary.get("accuracy_gate_pass") or summary.get("agreement_gate_pass"))
    if validated and unit == "article" and metric_pass:
        return _evidence_row(
            hypothesis_id="real_corpus_ideological_accuracy",
            status=SUPPORTED,
            publication_safe=True,
            claim_scope="real_corpus_article_level_labels",
            answer="Real-corpus ideological accuracy is supported against independent article-level labels.",
            evidence=summary,
            artifact_paths=[path] if path else [],
        )
    return _evidence_row(
        hypothesis_id="real_corpus_ideological_accuracy",
        status=BLOCKED,
        publication_safe=True,
        claim_scope="claim_boundary",
        answer=(
            "Real corpus results can support geometry/control/provenance, not true ideological accuracy, "
            "until independent article-level labels are attached."
        ),
        evidence=summary or {"independent_label_artifact_present": False},
        blockers=[
            "missing independently validated article-level labels",
            "source/proxy labels are exploratory metadata, not ground truth",
        ],
        next_tests=[
            "Attach independent article-level labels or a defended gold/proxy target.",
            "Report source/proxy agreement only as exploratory alignment.",
        ],
        artifact_paths=[path] if path else [],
    )


def _evaluate_terrain_semantics_software(payload: Mapping[str, Any]) -> Dict[str, Any]:
    claim_matrix = _load_json(payload.get("claim_matrix_path"))
    path = _summary_path(payload, "claim_matrix_path")
    terrain = _claim_by_id(claim_matrix, "terrain_incremental_signal")
    traversal = _claim_by_id(claim_matrix, "track4_traversal_validity")
    if bool(terrain.get("thesis_safe")):
        status = SUPPORTED
        answer = (
            "The terrain abstraction has software-level meaning: within planted labels, cross-terrain "
            "pairs carry extra traversal signal. The Bridge/Swamp/Tightrope/Void words remain interpretive labels."
        )
        safe = True
        blockers: list[str] = []
    elif terrain:
        status = PARTIAL if traversal else UNSUPPORTED
        answer = (
            "Terrain exists as a computable diagnostic, but the current evidence does not prove the full "
            "Bridge/Swamp/Tightrope/Void semantic mapping. Treat the names as UI/metaphor, not hard logic."
        )
        safe = True
        blockers = terrain.get("failure_reasons") or ["terrain incremental signal is not thesis-safe"]
    else:
        status = UNPROVEN
        answer = "No terrain incremental signal claim was found in the current evidence matrix."
        safe = False
        blockers = ["missing terrain_incremental_signal claim"]
    return _evidence_row(
        hypothesis_id="terrain_semantics_software",
        status=status,
        publication_safe=safe,
        claim_scope="software_abstraction_not_literal_metaphor",
        answer=answer,
        evidence={
            "terrain_incremental_signal": terrain,
            "track4_traversal_validity": traversal,
        },
        blockers=blockers,
        next_tests=[
            "Keep zone names as visual shorthand and validate only density/stress/work contracts.",
            "Use within-label cross-terrain work gaps as the semantic test.",
        ],
        artifact_paths=[path] if path else [],
    )


def _evaluate_track4_publishable_core(payload: Mapping[str, Any]) -> Dict[str, Any]:
    claim_matrix = _load_json(payload.get("claim_matrix_path"))
    matrix_path = _summary_path(payload, "claim_matrix_path")
    summary = _load_json(payload.get("track4_observer_state_summary_path"))
    summary_path = _summary_path(payload, "track4_observer_state_summary_path")
    action_only = _claim_by_id(claim_matrix, "track4_observer_state_action_only_separation")
    full = _claim_by_id(claim_matrix, "track4_observer_state_action_separation")
    direct_claim = summary.get("claim_evaluation") if isinstance(summary.get("claim_evaluation"), Mapping) else {}
    action_supported = bool(action_only.get("thesis_safe") or direct_claim.get("action_only_robustness_pass"))
    full_supported = bool(full.get("thesis_safe") or direct_claim.get("thesis_safe"))
    direct_failures = list(direct_claim.get("failure_reasons") or [])
    if full_supported:
        status = SUPPORTED
        answer = "Track 4 observer-state action and hysteresis are currently supportable as a bounded mechanism claim."
        safe = True
    elif action_supported:
        status = PARTIAL
        answer = (
            "Track 4 has a publishable action-separation core, but full observer-state/hysteresis robustness "
            "is not yet safe across every registered basis/seed gate."
        )
        safe = True
    else:
        status = UNSUPPORTED if (action_only or full or direct_claim) else UNPROVEN
        answer = "Track 4 is not ready as a core publishable mechanism under the current gates."
        safe = False
    return _evidence_row(
        hypothesis_id="track4_publishable_core",
        status=status,
        publication_safe=safe,
        claim_scope="track4_action_separation_not_terrain_metaphor",
        answer=answer,
        evidence={
            "claim_matrix_action_only": action_only,
            "claim_matrix_full": full,
            "direct_claim_evaluation": direct_claim,
        },
        blockers=[] if status == SUPPORTED else direct_failures or ["full Track 4 robustness is not yet universal"],
        next_tests=[
            "Separate action-only from hysteresis/terrain claims in the paper.",
            "Run null-calibrated action and observer-state ablations on larger real/synthetic cells.",
        ],
        artifact_paths=[path for path in (matrix_path, summary_path) if path],
    )


def _evaluate_property_theft_real_scale(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _load_json(payload.get("property_theft_transport_summary_path"))
    path = _summary_path(payload, "property_theft_transport_summary_path")
    if not summary:
        return _evidence_row(
            hypothesis_id="property_theft_real_scale",
            status=UNPROVEN,
            publication_safe=False,
            claim_scope="missing_holonomy_probe",
            answer="No property/theft observer-slice transport artifact was found.",
            blockers=["missing observer_slice_transport_summary.json"],
            next_tests=["Run observer-slice transport on a larger property/theft corpus slice."],
        )
    records = _to_int(summary.get("record_count"), default=0)
    excess = _to_float(summary.get("mean_excess_holonomy_action"))
    null_mean = _to_float(summary.get("mean_null_holonomy_action"))
    min_records = _to_int(payload.get("min_real_scale_property_records"), default=40)
    positive = excess is not None and excess > 0.0 and (null_mean is None or null_mean <= excess)
    if positive and records >= min_records:
        status = SUPPORTED
        answer = "Property/theft holonomy survives at registered real-scale record count."
        safe = True
        blockers: list[str] = []
    elif positive:
        status = PARTIAL
        answer = (
            "The property/theft result proves the mechanism can express chart-dependent path cost, "
            "but it is still a toy/small-probe result rather than real-scale evidence."
        )
        safe = True
        blockers = ["record count below real-scale threshold"]
    else:
        status = UNSUPPORTED
        answer = "The property/theft holonomy probe does not separate from its null."
        safe = False
        blockers = ["no positive excess holonomy over null"]
    return _evidence_row(
        hypothesis_id="property_theft_real_scale",
        status=status,
        publication_safe=safe,
        claim_scope="observer_slice_holonomy",
        answer=answer,
        evidence={
            "record_count": records,
            "min_real_scale_records": min_records,
            "mean_excess_holonomy_action": excess,
            "mean_null_holonomy_action": null_mean,
        },
        blockers=blockers,
        next_tests=[
            "Run property/theft over many article/concept pairs and multiple observer charts.",
            "Register a shuffled/translation null per edge, not only per toy loop.",
        ],
        artifact_paths=[path] if path else [],
    )


def _generic_external_gate(
    payload: Mapping[str, Any],
    *,
    hypothesis_id: str,
    artifact_key: str,
    required_bool_key: str,
    claim_scope: str,
    missing_answer: str,
    supported_answer: str,
    blocked_answer: str,
    next_tests: Sequence[str],
) -> Dict[str, Any]:
    summary = _load_json(payload.get(artifact_key))
    path = _summary_path(payload, artifact_key)
    if not summary:
        return _evidence_row(
            hypothesis_id=hypothesis_id,
            status=UNPROVEN,
            publication_safe=False,
            claim_scope=claim_scope,
            answer=missing_answer,
            blockers=[f"missing {artifact_key}"],
            next_tests=next_tests,
        )
    if bool(summary.get(required_bool_key)):
        return _evidence_row(
            hypothesis_id=hypothesis_id,
            status=SUPPORTED,
            publication_safe=True,
            claim_scope=claim_scope,
            answer=supported_answer,
            evidence=summary,
            artifact_paths=[path] if path else [],
        )
    return _evidence_row(
        hypothesis_id=hypothesis_id,
        status=BLOCKED,
        publication_safe=True,
        claim_scope=claim_scope,
        answer=blocked_answer,
        evidence=summary,
        blockers=list(summary.get("failure_reasons") or [f"{required_bool_key} is false"]),
        next_tests=next_tests,
        artifact_paths=[path] if path else [],
    )


def _evaluate_prompt_set_invariance(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _generic_external_gate(
        payload,
        hypothesis_id="prompt_set_invariance",
        artifact_key="prompt_invariance_summary_path",
        required_bool_key="prompt_invariance_pass",
        claim_scope="alternate_v_observer_prompt_sets",
        missing_answer="No alternate-prompt invariance artifact was found.",
        supported_answer="The result survives multiple registered V-observer prompt sets.",
        blocked_answer="Prompt-set invariance was tested but did not pass.",
        next_tests=[
            "Run at least three semantically equivalent V-observer prompt banks.",
            "Require preserved claim direction and bounded effect-size drift.",
        ],
    )


def _evaluate_track3_density_semantic_interpretation(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _generic_external_gate(
        payload,
        hypothesis_id="track3_density_semantic_interpretation",
        artifact_key="track3_density_validation_path",
        required_bool_key="track3_density_semantic_pass",
        claim_scope="independent_track3_density_validation",
        missing_answer="No independent Track 3 density validation artifact was found.",
        supported_answer="Track 3 density correlates with an independently defined semantic/stability target.",
        blocked_answer="Track 3 density validation was attempted but did not pass.",
        next_tests=[
            "Compare rho/thermal work to held-out disagreement, label entropy, or path stability.",
            "Separate Track 3 as density/stability from Track 4 action semantics.",
        ],
    )


def _evaluate_visualization_human_readability(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _generic_external_gate(
        payload,
        hypothesis_id="visualization_human_readability",
        artifact_key="visualization_validation_path",
        required_bool_key="human_readability_pass",
        claim_scope="human_factors_validation",
        missing_answer="No human-readability validation artifact was found for the Dash/atlas view.",
        supported_answer="Human readers can interpret the atlas/path visualization above the registered gate.",
        blocked_answer="The visualization was tested but did not clear the human-readability gate.",
        next_tests=[
            "Run a small blinded task: users identify observer switch, high-action path, and terrain contract.",
            "Measure accuracy/time/confidence against static PCA and single-chart views.",
        ],
    )


EVALUATORS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
    "observer_recentering_robustness": _evaluate_observer_recentering_robustness,
    "simple_baseline_superiority": _evaluate_simple_baseline_superiority,
    "real_corpus_ideological_accuracy": _evaluate_real_corpus_ideological_accuracy,
    "terrain_semantics_software": _evaluate_terrain_semantics_software,
    "track4_publishable_core": _evaluate_track4_publishable_core,
    "property_theft_real_scale": _evaluate_property_theft_real_scale,
    "prompt_set_invariance": _evaluate_prompt_set_invariance,
    "track3_density_semantic_interpretation": _evaluate_track3_density_semantic_interpretation,
    "visualization_human_readability": _evaluate_visualization_human_readability,
}


class MockHypothesisOrchestrator:
    """Small DAG-backed orchestrator for claim-boundary validation.

    This deliberately reuses the repository's node-manifest pattern while
    avoiding heavyweight model inference. Each node reads existing ledger
    artifacts and emits a bounded answer for one open hypothesis.
    """

    def __init__(self, config: OpenHypothesisConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.graph = AblationTaskGraph(
            self.output_dir / "_hypothesis_dag",
            config_hash="open_hypothesis_suite_v1",
            graph_name="open_hypothesis_suite",
            branch_metadata={"schema_version": OPEN_HYPOTHESIS_SCHEMA_VERSION},
        )

    def _input_payload(self) -> Dict[str, Any]:
        defaults = _default_artifact_paths(self.config.repo_root)
        payload: Dict[str, Any] = {
            "repo_root": str(self.config.repo_root),
            "output_dir": str(self.output_dir),
            "min_real_scale_property_records": int(self.config.min_real_scale_property_records),
            "claim_matrix_path": self.config.claim_matrix_path or defaults.get("claim_matrix_path"),
            "recentering_summary_path": self.config.recentering_summary_path
            or defaults.get("recentering_summary_path"),
            "track4_observer_state_summary_path": self.config.track4_observer_state_summary_path
            or defaults.get("track4_observer_state_summary_path"),
            "property_theft_transport_summary_path": self.config.property_theft_transport_summary_path
            or defaults.get("property_theft_transport_summary_path"),
            "independent_label_summary_path": self.config.independent_label_summary_path,
            "prompt_invariance_summary_path": self.config.prompt_invariance_summary_path,
            "track3_density_validation_path": self.config.track3_density_validation_path,
            "visualization_validation_path": self.config.visualization_validation_path,
        }
        payload["artifact_paths"] = {
            key: str(value) if value else None
            for key, value in payload.items()
            if key.endswith("_path")
        }
        return _jsonable(payload)

    def run(self) -> Dict[str, Any]:
        input_result = self.graph.run_node(
            "discover_existing_artifacts",
            params={"repo_root": str(self.config.repo_root)},
            action=lambda: {"outputs": self._input_payload()},
        )
        discovered = dict(input_result.outputs)
        claim_rows: list[Dict[str, Any]] = []
        for hypothesis_id in HYPOTHESIS_IDS:
            result = self.graph.run_node(
                hypothesis_id,
                dependencies=["discover_existing_artifacts"],
                params={
                    "hypothesis_id": hypothesis_id,
                    "artifact_paths": discovered.get("artifact_paths", {}),
                },
                action=lambda hid=hypothesis_id: {
                    "outputs": EVALUATORS[hid](discovered),
                },
            )
            claim_rows.append(dict(result.outputs))
        status_counts: Dict[str, int] = {}
        for row in claim_rows:
            status = str(row.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
        contract = build_orchestration_contract(
            dag_id="open_hypothesis_suite",
            executor="mock_orchestrator",
            nodes=[
                {"node_id": "discover_existing_artifacts", "status": "completed"},
                *({"node_id": hypothesis_id, "status": "completed"} for hypothesis_id in HYPOTHESIS_IDS),
            ],
            dependencies={hypothesis_id: ["discover_existing_artifacts"] for hypothesis_id in HYPOTHESIS_IDS},
            manifest_paths=[Path(result.manifest_path or "") for result in self.graph.results.values()],
            metadata={
                "schema_version": OPEN_HYPOTHESIS_SCHEMA_VERSION,
                "hypothesis_count": len(HYPOTHESIS_IDS),
            },
        )
        payload = {
            "schema_version": OPEN_HYPOTHESIS_SCHEMA_VERSION,
            "summary_type": "open_hypothesis_suite",
            "generated_at_utc": _utc_now(),
            "hypothesis_count": len(claim_rows),
            "status_counts": status_counts,
            "claims": claim_rows,
            "artifact_paths": discovered.get("artifact_paths", {}),
            "orchestration_contract": contract,
        }
        self._write_outputs(payload)
        return payload

    def _write_outputs(self, payload: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "open_hypothesis_matrix.json"
        csv_path = self.output_dir / "open_hypothesis_matrix.csv"
        json_path.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "hypothesis_id",
                "status",
                "publication_safe",
                "claim_scope",
                "answer",
                "blockers",
                "artifact_paths",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in payload.get("claims", []) or []:
                writer.writerow(
                    {
                        "hypothesis_id": row.get("hypothesis_id"),
                        "status": row.get("status"),
                        "publication_safe": row.get("publication_safe"),
                        "claim_scope": row.get("claim_scope"),
                        "answer": row.get("answer"),
                        "blockers": "; ".join(str(v) for v in row.get("blockers", []) or []),
                        "artifact_paths": "; ".join(str(v) for v in row.get("artifact_paths", []) or []),
                    }
                )


def run_open_hypothesis_suite(config: OpenHypothesisConfig) -> Dict[str, Any]:
    return MockHypothesisOrchestrator(config).run()
