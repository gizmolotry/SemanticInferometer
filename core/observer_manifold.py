from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
BUNDLE_TYPE = "observer_manifold_bundle"
EDGE_LEDGER_TYPE = "observer_manifold_edge_action_ledger"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _clean_label(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "missing"}:
        return ""
    return text


def article_idx(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("idx", "index", "article_idx"):
        if key not in row:
            continue
        try:
            return int(row[key])
        except Exception:
            return None
    return None


def xy(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    try:
        x = float(row.get("x", row.get("observer_x")))
        y = float(row.get("y", row.get("observer_y")))
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return np.asarray([x, y], dtype=np.float64)


def xyz(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    point = xy(row)
    if point is None:
        return None
    try:
        z = float(row.get("z", row.get("observer_z", row.get("z_height", 0.0))))
    except Exception:
        z = 0.0
    if not math.isfinite(z):
        z = 0.0
    return np.asarray([float(point[0]), float(point[1]), z], dtype=np.float64)


def article_map(view_state: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = view_state.get("articles") if isinstance(view_state.get("articles"), list) else []
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = article_idx(row)
        if idx is not None:
            out[idx] = dict(row)
    return out


def metadata_by_idx(run_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    for candidate in (Path(run_dir) / "MONOLITH_DATA.csv", Path(run_dir) / "article_metadata.csv"):
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        if "index" not in df.columns:
            df = df.reset_index().rename(columns={"index": "index"})
        rows: Dict[int, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            rows[idx] = {str(key): row.get(key) for key in df.columns}
        return rows, str(candidate)
    json_path = Path(run_dir) / "article_metadata.json"
    if json_path.exists():
        payload = _load_json(json_path)
        raw_rows = payload.get("articles") or payload.get("rows") or payload.get("metadata")
        if isinstance(raw_rows, list):
            rows: Dict[int, Dict[str, Any]] = {}
            for pos, row in enumerate(raw_rows):
                if not isinstance(row, dict):
                    continue
                idx = article_idx(row)
                rows[int(idx if idx is not None else pos)] = dict(row)
            return rows, str(json_path)
    return {}, None


def _infer_observer_idx(observer_dir: Path, view_state: Mapping[str, Any]) -> Optional[int]:
    focus = view_state.get("observer_focus") if isinstance(view_state.get("observer_focus"), dict) else {}
    if "idx" in focus:
        try:
            return int(focus["idx"])
        except Exception:
            pass
    try:
        return int(observer_dir.name.split("_")[-1])
    except Exception:
        return None


def _label_for(metadata: Mapping[int, Mapping[str, Any]], idx: int) -> Dict[str, str]:
    row = metadata.get(idx, {})
    return {
        key: _clean_label(row.get(key))
        for key in ("source", "publication", "author", "label", "perspective_tag", "verdict", "zone")
        if _clean_label(row.get(key))
    }


@dataclass(frozen=True)
class ObserverNode:
    idx: int
    global_xyz: Tuple[float, float, float]
    translation_null_xyz: Tuple[float, float, float]
    observer_xyz: Tuple[float, float, float]
    nontranslation_shift: float
    density: Optional[float] = None
    stress: Optional[float] = None
    work_actual: Optional[float] = None
    zone: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    observer_simplex: Optional[Tuple[float, ...]] = None
    spectral_probe_magnitudes: Optional[Tuple[float, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(self.__dict__)


@dataclass(frozen=True)
class ObserverEdgeAction:
    source_idx: int
    target_idx: int
    path_id: int
    step_id: int
    observer_distance: float
    translation_null_distance: float
    excess_distance: float
    positive_excess_distance: float
    stress_penalty: float
    density_penalty: float
    observer_action: float
    translation_null_action: float
    excess_action: float
    positive_excess_action: float
    source_zone: str = ""
    target_zone: str = ""
    source_label: str = ""
    target_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(self.__dict__)


@dataclass(frozen=True)
class ObserverManifoldBundle:
    run_dir: str
    observer_dir: str
    observer_idx: int
    focus_idx: int
    metadata_path: Optional[str]
    global_view_state_path: str
    observer_view_state_path: str
    nodes: Dict[int, ObserverNode]
    edges: List[ObserverEdgeAction]
    path_count: int
    provenance: Dict[str, Any]
    source_artifacts: Dict[str, str] = field(default_factory=dict)
    coordinate_frame: Dict[str, Any] = field(default_factory=dict)
    paths: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def focus_xy_centered(self) -> bool:
        node = self.nodes.get(int(self.focus_idx))
        return bool(node and np.linalg.norm(np.asarray(node.observer_xyz[:2], dtype=float)) <= 1e-9)

    @property
    def mean_nontranslation_shift(self) -> Optional[float]:
        values = [node.nontranslation_shift for node in self.nodes.values() if math.isfinite(node.nontranslation_shift)]
        return float(np.mean(values)) if values else None

    def edge_summary(self) -> Dict[str, Any]:
        finite_edges = [edge for edge in self.edges if math.isfinite(edge.observer_action)]
        return {
            "edge_count": len(self.edges),
            "mean_observer_action": _mean([edge.observer_action for edge in finite_edges]),
            "mean_translation_null_action": _mean([edge.translation_null_action for edge in finite_edges]),
            "mean_excess_action": _mean([edge.excess_action for edge in finite_edges]),
            "mean_positive_excess_action": _mean([edge.positive_excess_action for edge in finite_edges]),
            "mean_excess_distance": _mean([edge.excess_distance for edge in finite_edges]),
            "positive_excess_edge_count": sum(1 for edge in finite_edges if edge.positive_excess_action > 0.0),
        }

    def to_dict(self, *, include_nodes: bool = True, include_edges: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "bundle_type": BUNDLE_TYPE,
            "run_dir": self.run_dir,
            "observer_dir": self.observer_dir,
            "observer_idx": self.observer_idx,
            "focus_idx": self.focus_idx,
            "metadata_path": self.metadata_path,
            "source_artifacts": dict(self.source_artifacts),
            "coordinate_frame": dict(self.coordinate_frame),
            "global_view_state_path": self.global_view_state_path,
            "observer_view_state_path": self.observer_view_state_path,
            "node_count": len(self.nodes),
            "path_count": self.path_count,
            "edge_count": len(self.edges),
            "focus_xy_centered": self.focus_xy_centered,
            "mean_nontranslation_shift": self.mean_nontranslation_shift,
            "edge_action_summary": self.edge_summary(),
            "provenance": dict(self.provenance),
        }
        if include_nodes:
            payload["nodes"] = [node.to_dict() for _, node in sorted(self.nodes.items())]
        payload["paths"] = _json_safe(self.paths)
        if include_edges:
            payload["edge_action_ledger"] = [edge.to_dict() for edge in self.edges]
        return _json_safe(payload)


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _edge_penalties(source: ObserverNode, target: ObserverNode) -> Tuple[float, float]:
    stress_values = [value for value in (source.stress, target.stress) if value is not None and math.isfinite(value)]
    density_values = [value for value in (source.density, target.density) if value is not None and math.isfinite(value)]
    stress_penalty = float(np.mean(stress_values)) if stress_values else 0.0
    density_penalty = float(1.0 - np.mean(density_values)) if density_values else 0.0
    return max(stress_penalty, 0.0), max(density_penalty, 0.0)


def _path_indices_from_walker_paths(view_state: Mapping[str, Any]) -> List[List[int]]:
    paths = view_state.get("walker_paths") if isinstance(view_state.get("walker_paths"), list) else []
    out: List[List[int]] = []
    for row in paths:
        if not isinstance(row, dict):
            continue
        raw_indices = row.get("path_indices")
        if isinstance(raw_indices, list):
            indices = []
            for value in raw_indices:
                try:
                    indices.append(int(value))
                except Exception:
                    pass
            if len(indices) >= 2:
                out.append(indices)
                continue
        idx = article_idx(row)
        if idx is not None:
            out.append([idx, idx])
    return out


def _path_rows_from_walker_paths(view_state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_paths = view_state.get("walker_paths") if isinstance(view_state.get("walker_paths"), list) else []
    rows: List[Dict[str, Any]] = []
    for path_id, row in enumerate(raw_paths):
        if not isinstance(row, dict):
            continue
        indices: List[int] = []
        raw_indices = row.get("path_indices")
        if isinstance(raw_indices, list):
            for value in raw_indices:
                try:
                    indices.append(int(value))
                except Exception:
                    pass
        elif article_idx(row) is not None:
            idx = int(article_idx(row) or 0)
            indices = [idx, idx]
        rows.append(
            {
                "path_id": path_id,
                "article_idx": article_idx(row),
                "path_indices": indices,
                "path_space": row.get("path_space"),
                "path_geometry_role": row.get("path_geometry_role"),
                "focused_observer_replay": bool(row.get("focused_observer_replay")),
                "fresh_focused_observer_replay": bool(row.get("fresh_focused_observer_replay")),
                "legacy_projected_path": bool(row.get("legacy_projected_path")),
                "start_xyz": [
                    _float_or_none(row.get("start_x")),
                    _float_or_none(row.get("start_y")),
                    _float_or_none(row.get("start_z")),
                ],
                "end_xyz": [
                    _float_or_none(row.get("end_x")),
                    _float_or_none(row.get("end_y")),
                    _float_or_none(row.get("end_z")),
                ],
                "n_points": row.get("n_points"),
                "work_integral": _float_or_none(row.get("work_integral")),
            }
        )
    return rows


def build_edge_action_ledger(
    nodes: Mapping[int, ObserverNode],
    paths: Sequence[Sequence[int]],
    *,
    label_key: str = "source",
) -> List[ObserverEdgeAction]:
    edges: List[ObserverEdgeAction] = []
    for path_id, path in enumerate(paths):
        cleaned = [int(idx) for idx in path if int(idx) in nodes]
        for step_id, (source_idx, target_idx) in enumerate(zip(cleaned, cleaned[1:])):
            source = nodes[source_idx]
            target = nodes[target_idx]
            observer_distance = float(
                np.linalg.norm(np.asarray(target.observer_xyz) - np.asarray(source.observer_xyz))
            )
            translation_distance = float(
                np.linalg.norm(np.asarray(target.translation_null_xyz) - np.asarray(source.translation_null_xyz))
            )
            stress_penalty, density_penalty = _edge_penalties(source, target)
            observer_action = observer_distance + stress_penalty + density_penalty
            translation_action = translation_distance + stress_penalty + density_penalty
            excess_distance = observer_distance - translation_distance
            excess_action = observer_action - translation_action
            edges.append(
                ObserverEdgeAction(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    path_id=path_id,
                    step_id=step_id,
                    observer_distance=observer_distance,
                    translation_null_distance=translation_distance,
                    excess_distance=excess_distance,
                    positive_excess_distance=max(excess_distance, 0.0),
                    stress_penalty=stress_penalty,
                    density_penalty=density_penalty,
                    observer_action=observer_action,
                    translation_null_action=translation_action,
                    excess_action=excess_action,
                    positive_excess_action=max(excess_action, 0.0),
                    source_zone=source.zone,
                    target_zone=target.zone,
                    source_label=source.labels.get(label_key, ""),
                    target_label=target.labels.get(label_key, ""),
                )
            )
    return edges


def load_observer_manifold_bundle(
    run_dir: Path,
    observer_dir: Path,
    *,
    label_key: str = "source",
) -> ObserverManifoldBundle:
    run_dir = Path(run_dir)
    observer_dir = Path(observer_dir)
    global_view_path = run_dir / "MONOLITH.view_state.json"
    observer_view_path = observer_dir / "MONOLITH.view_state.json"
    global_state = _load_json(global_view_path)
    observer_state = _load_json(observer_view_path)
    global_articles = article_map(global_state)
    observer_articles = article_map(observer_state)
    metadata, metadata_path = metadata_by_idx(run_dir)
    observer_idx = _infer_observer_idx(observer_dir, observer_state)
    if observer_idx is None:
        raise ValueError(f"Could not infer observer index from {observer_dir}")
    focus_payload = (
        observer_state.get("observer_focus")
        if isinstance(observer_state.get("observer_focus"), dict)
        else {}
    )
    focus_idx = int(observer_idx)
    if int(focus_payload.get("idx", focus_idx)) in observer_articles:
        focus_idx = int(focus_payload.get("idx", focus_idx))
    global_focus = xy(global_articles.get(focus_idx, {}))
    if global_focus is None:
        global_focus = np.zeros(2, dtype=np.float64)

    nodes: Dict[int, ObserverNode] = {}
    for idx in sorted(set(global_articles).intersection(observer_articles)):
        global_point = xyz(global_articles[idx])
        observer_point = xyz(observer_articles[idx])
        if global_point is None or observer_point is None:
            continue
        translation_xyz = np.asarray(global_point, dtype=np.float64).copy()
        translation_xyz[:2] = translation_xyz[:2] - global_focus
        shift = float(np.linalg.norm(np.asarray(observer_point[:2]) - np.asarray(translation_xyz[:2])))
        meta_row = metadata.get(idx, {})
        density = _float_or_none(meta_row.get("density", observer_articles[idx].get("density")))
        stress = _float_or_none(meta_row.get("stress", observer_articles[idx].get("stress")))
        work_actual = _float_or_none(meta_row.get("w_actual", observer_articles[idx].get("w_actual")))
        zone = _clean_label(meta_row.get("zone", observer_articles[idx].get("zone")))
        observer_simplex = _first_sequence(
            observer_articles[idx].get("observer_simplex"),
            observer_articles[idx].get("observer_simplex_weights"),
            meta_row.get("observer_simplex"),
            meta_row.get("observer_simplex_weights"),
        )
        spectral_probe_magnitudes = _first_sequence(
            observer_articles[idx].get("spectral_probe_magnitudes"),
            observer_articles[idx].get("probe_magnitudes"),
            meta_row.get("spectral_probe_magnitudes"),
            meta_row.get("probe_magnitudes"),
        )
        nodes[idx] = ObserverNode(
            idx=idx,
            global_xyz=tuple(float(v) for v in global_point),
            translation_null_xyz=tuple(float(v) for v in translation_xyz),
            observer_xyz=tuple(float(v) for v in observer_point),
            nontranslation_shift=shift,
            density=density,
            stress=stress,
            work_actual=work_actual,
            zone=zone,
            labels=_label_for(metadata, idx),
            observer_simplex=observer_simplex,
            spectral_probe_magnitudes=spectral_probe_magnitudes,
        )
    paths = _path_indices_from_walker_paths(observer_state)
    path_rows = _path_rows_from_walker_paths(observer_state)
    edges = build_edge_action_ledger(nodes, paths, label_key=label_key)
    coordinate_frame = {
        "global_space": "MONOLITH.view_state.json/articles[x,y,z]",
        "observer_space": "observer_<idx>/MONOLITH.view_state.json/articles[x,y,z]",
        "xy_origin_policy": "selected_article_at_observer_xy_origin",
        "z_origin_policy": "xy_origin_preserve_canonical_z",
        "translation_null_definition": "global_xyz_minus_global_focus_xy_without_observer_recompute",
        "path_space": "observer_view_state.walker_paths",
    }
    source_artifacts = {
        "global_view_state": str(global_view_path),
        "observer_view_state": str(observer_view_path),
    }
    if metadata_path:
        source_artifacts["metadata"] = str(metadata_path)
    provenance = {
        "coordinate_source": "global_and_observer_view_state",
        "edge_source": "observer_view_state.walker_paths.path_indices",
        "label_key": label_key,
        "recenter_mode": focus_payload.get("recenter_mode"),
        "local_track_recompute_active": bool(focus_payload.get("local_track_recompute_active")),
        "accepted_sidecar_mode": bool(focus_payload.get("accepted_sidecar_mode")),
        "legacy_hydrated": bool(
            (observer_state.get("path_ledger_provenance") or {}).get("legacy_path_ledger_hydrated")
            if isinstance(observer_state.get("path_ledger_provenance"), dict)
            else False
        ),
        "focused_observer_replay": any(
            bool(row.get("focused_observer_replay"))
            for row in observer_state.get("walker_paths", [])
            if isinstance(row, dict)
        ),
        "fresh_focused_observer_replay": any(
            bool(row.get("fresh_focused_observer_replay"))
            for row in observer_state.get("walker_paths", [])
            if isinstance(row, dict)
        ),
    }
    provenance["provenance_hash"] = _stable_hash(
        {
            "run_dir": str(run_dir),
            "observer_dir": str(observer_dir),
            "focus_idx": focus_idx,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "paths": paths,
        }
    )
    return ObserverManifoldBundle(
        run_dir=str(run_dir),
        observer_dir=str(observer_dir),
        observer_idx=int(observer_idx),
        focus_idx=int(focus_idx),
        metadata_path=metadata_path,
        global_view_state_path=str(global_view_path),
        observer_view_state_path=str(observer_view_path),
        nodes=nodes,
        edges=edges,
        path_count=len(paths),
        provenance=provenance,
        source_artifacts=source_artifacts,
        coordinate_frame=coordinate_frame,
        paths=path_rows,
    )


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _coerce_optional_tuple(value: Any) -> Optional[Tuple[float, ...]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception:
            return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    out: List[float] = []
    for item in list(value):
        try:
            item_f = float(item)
        except Exception:
            return None
        if not math.isfinite(item_f):
            return None
        out.append(item_f)
    return tuple(out) if out else None


def _first_sequence(*values: Any) -> Optional[Tuple[float, ...]]:
    for value in values:
        coerced = _coerce_optional_tuple(value)
        if coerced is not None:
            return coerced
    return None


def write_observer_manifold_bundle(bundle: ObserverManifoldBundle, output_dir: Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"observer_{bundle.observer_idx}_manifold_bundle.json"
    csv_path = output_dir / f"observer_{bundle.observer_idx}_edge_action_ledger.csv"
    json_path.write_text(
        json.dumps(bundle.to_dict(include_nodes=True, include_edges=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(ObserverEdgeAction.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for edge in bundle.edges:
            writer.writerow(edge.to_dict())
    return {"bundle_json": str(json_path), "edge_ledger_csv": str(csv_path)}


def _coerce_tuple(values: Any, *, length: int, default: float = 0.0) -> Tuple[float, ...]:
    raw = list(values) if isinstance(values, (list, tuple)) else []
    out: List[float] = []
    for idx in range(length):
        value = raw[idx] if idx < len(raw) else default
        try:
            value = float(value)
        except Exception:
            value = default
        out.append(value if math.isfinite(value) else default)
    return tuple(out)


def _node_from_payload(raw: Mapping[str, Any]) -> ObserverNode:
    observer_simplex = raw.get("observer_simplex")
    spectral_probe_magnitudes = raw.get("spectral_probe_magnitudes")
    return ObserverNode(
        idx=int(raw.get("idx", 0)),
        global_xyz=_coerce_tuple(raw.get("global_xyz"), length=3),
        translation_null_xyz=_coerce_tuple(raw.get("translation_null_xyz"), length=3),
        observer_xyz=_coerce_tuple(raw.get("observer_xyz"), length=3),
        nontranslation_shift=float(raw.get("nontranslation_shift") or 0.0),
        density=_float_or_none(raw.get("density")),
        stress=_float_or_none(raw.get("stress")),
        work_actual=_float_or_none(raw.get("work_actual")),
        zone=_clean_label(raw.get("zone")),
        labels=dict(raw.get("labels") if isinstance(raw.get("labels"), dict) else {}),
        observer_simplex=(
            _coerce_tuple(observer_simplex, length=len(observer_simplex))
            if isinstance(observer_simplex, (list, tuple))
            else None
        ),
        spectral_probe_magnitudes=(
            _coerce_tuple(spectral_probe_magnitudes, length=len(spectral_probe_magnitudes))
            if isinstance(spectral_probe_magnitudes, (list, tuple))
            else None
        ),
    )


def _edge_from_payload(raw: Mapping[str, Any]) -> ObserverEdgeAction:
    return ObserverEdgeAction(
        source_idx=int(raw.get("source_idx", 0)),
        target_idx=int(raw.get("target_idx", 0)),
        path_id=int(raw.get("path_id", 0)),
        step_id=int(raw.get("step_id", 0)),
        observer_distance=float(raw.get("observer_distance") or 0.0),
        translation_null_distance=float(raw.get("translation_null_distance") or 0.0),
        excess_distance=float(raw.get("excess_distance") or 0.0),
        positive_excess_distance=float(raw.get("positive_excess_distance") or 0.0),
        stress_penalty=float(raw.get("stress_penalty") or 0.0),
        density_penalty=float(raw.get("density_penalty") or 0.0),
        observer_action=float(raw.get("observer_action") or 0.0),
        translation_null_action=float(raw.get("translation_null_action") or 0.0),
        excess_action=float(raw.get("excess_action") or 0.0),
        positive_excess_action=float(raw.get("positive_excess_action") or 0.0),
        source_zone=_clean_label(raw.get("source_zone")),
        target_zone=_clean_label(raw.get("target_zone")),
        source_label=_clean_label(raw.get("source_label")),
        target_label=_clean_label(raw.get("target_label")),
    )


def read_observer_manifold_bundle(bundle_json: Path) -> ObserverManifoldBundle:
    payload = _load_json(Path(bundle_json))
    if payload.get("bundle_type") != BUNDLE_TYPE:
        raise ValueError(f"Not an observer manifold bundle: {bundle_json}")
    raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    raw_edges = payload.get("edge_action_ledger") if isinstance(payload.get("edge_action_ledger"), list) else []
    nodes = {
        node.idx: node
        for node in (_node_from_payload(row) for row in raw_nodes if isinstance(row, dict))
    }
    edges = [_edge_from_payload(row) for row in raw_edges if isinstance(row, dict)]
    return ObserverManifoldBundle(
        run_dir=str(payload.get("run_dir", "")),
        observer_dir=str(payload.get("observer_dir", "")),
        observer_idx=int(payload.get("observer_idx", 0)),
        focus_idx=int(payload.get("focus_idx", payload.get("observer_idx", 0))),
        metadata_path=payload.get("metadata_path"),
        global_view_state_path=str(payload.get("global_view_state_path", "")),
        observer_view_state_path=str(payload.get("observer_view_state_path", "")),
        nodes=nodes,
        edges=edges,
        path_count=int(payload.get("path_count", 0)),
        provenance=dict(payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}),
        source_artifacts=dict(
            payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), dict) else {}
        ),
        coordinate_frame=dict(
            payload.get("coordinate_frame") if isinstance(payload.get("coordinate_frame"), dict) else {}
        ),
        paths=list(payload.get("paths") if isinstance(payload.get("paths"), list) else []),
    )


def build_observer_manifold_bundle(
    run_dir: Path,
    observer_dir: Path,
    *,
    label_key: str = "source",
) -> ObserverManifoldBundle:
    return load_observer_manifold_bundle(run_dir, observer_dir, label_key=label_key)


def _metadata_rows_from_payload(metadata: Any) -> Dict[int, Dict[str, Any]]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        rows: Dict[int, Dict[str, Any]] = {}
        for key, value in metadata.items():
            if not isinstance(value, Mapping):
                continue
            try:
                idx = int(key)
            except Exception:
                idx = article_idx(value)
            if idx is not None:
                rows[int(idx)] = dict(value)
        return rows
    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        rows = {}
        for pos, value in enumerate(metadata):
            if not isinstance(value, Mapping):
                continue
            idx = article_idx(value)
            rows[int(idx if idx is not None else pos)] = dict(value)
        return rows
    return {}


def build_observer_manifold_bundle_from_view_states(
    global_view_state: Mapping[str, Any],
    observer_view_state: Mapping[str, Any],
    *,
    metadata: Any = None,
    run_dir: str = "",
    observer_dir: str = "",
    observer_idx: Optional[int] = None,
    focus_idx: Optional[int] = None,
    label_key: str = "source",
    metadata_path: Optional[str] = None,
    global_view_state_path: str = "",
    observer_view_state_path: str = "",
) -> ObserverManifoldBundle:
    global_articles = article_map(global_view_state)
    observer_articles = article_map(observer_view_state)
    metadata_rows = _metadata_rows_from_payload(metadata)
    focus_payload = (
        observer_view_state.get("observer_focus")
        if isinstance(observer_view_state.get("observer_focus"), dict)
        else {}
    )
    resolved_observer_idx = observer_idx
    if resolved_observer_idx is None:
        try:
            resolved_observer_idx = int(focus_payload.get("idx"))
        except Exception:
            resolved_observer_idx = 0
    resolved_focus_idx = focus_idx
    if resolved_focus_idx is None:
        try:
            resolved_focus_idx = int(focus_payload.get("idx"))
        except Exception:
            resolved_focus_idx = int(resolved_observer_idx)
    global_focus = xy(global_articles.get(int(resolved_focus_idx), {}))
    if global_focus is None:
        global_focus = np.zeros(2, dtype=np.float64)

    nodes: Dict[int, ObserverNode] = {}
    for idx in sorted(set(global_articles).intersection(observer_articles)):
        global_point = xyz(global_articles[idx])
        observer_point = xyz(observer_articles[idx])
        if global_point is None or observer_point is None:
            continue
        translation_xyz = np.asarray(global_point, dtype=np.float64).copy()
        translation_xyz[:2] = translation_xyz[:2] - global_focus
        shift = float(np.linalg.norm(np.asarray(observer_point[:2]) - np.asarray(translation_xyz[:2])))
        meta_row = metadata_rows.get(idx, {})
        density = _float_or_none(meta_row.get("density", observer_articles[idx].get("density")))
        stress = _float_or_none(meta_row.get("stress", observer_articles[idx].get("stress")))
        work_actual = _float_or_none(meta_row.get("w_actual", observer_articles[idx].get("w_actual")))
        zone = _clean_label(meta_row.get("zone", observer_articles[idx].get("zone")))
        observer_simplex = _first_sequence(
            observer_articles[idx].get("observer_simplex"),
            observer_articles[idx].get("observer_simplex_weights"),
            meta_row.get("observer_simplex"),
            meta_row.get("observer_simplex_weights"),
        )
        spectral_probe_magnitudes = _first_sequence(
            observer_articles[idx].get("spectral_probe_magnitudes"),
            observer_articles[idx].get("probe_magnitudes"),
            meta_row.get("spectral_probe_magnitudes"),
            meta_row.get("probe_magnitudes"),
        )
        nodes[idx] = ObserverNode(
            idx=idx,
            global_xyz=tuple(float(v) for v in global_point),
            translation_null_xyz=tuple(float(v) for v in translation_xyz),
            observer_xyz=tuple(float(v) for v in observer_point),
            nontranslation_shift=shift,
            density=density,
            stress=stress,
            work_actual=work_actual,
            zone=zone,
            labels=_label_for(metadata_rows, idx),
            observer_simplex=observer_simplex,
            spectral_probe_magnitudes=spectral_probe_magnitudes,
        )

    paths = _path_indices_from_walker_paths(observer_view_state)
    path_rows = _path_rows_from_walker_paths(observer_view_state)
    edges = build_edge_action_ledger(nodes, paths, label_key=label_key)
    source_artifacts = {
        "global_view_state": global_view_state_path,
        "observer_view_state": observer_view_state_path,
    }
    if metadata_path:
        source_artifacts["metadata"] = metadata_path
    provenance = {
        "coordinate_source": "in_memory_view_state",
        "edge_source": "observer_view_state.walker_paths.path_indices",
        "label_key": label_key,
        "legacy_hydrated": bool(
            (observer_view_state.get("path_ledger_provenance") or {}).get("legacy_path_ledger_hydrated")
            if isinstance(observer_view_state.get("path_ledger_provenance"), dict)
            else False
        ),
        "focused_observer_replay": any(
            bool(row.get("focused_observer_replay"))
            for row in observer_view_state.get("walker_paths", [])
            if isinstance(row, dict)
        ),
        "fresh_focused_observer_replay": any(
            bool(row.get("fresh_focused_observer_replay"))
            for row in observer_view_state.get("walker_paths", [])
            if isinstance(row, dict)
        ),
    }
    provenance["provenance_hash"] = _stable_hash(
        {
            "run_dir": run_dir,
            "observer_dir": observer_dir,
            "focus_idx": resolved_focus_idx,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "paths": paths,
        }
    )
    return ObserverManifoldBundle(
        run_dir=run_dir,
        observer_dir=observer_dir,
        observer_idx=int(resolved_observer_idx),
        focus_idx=int(resolved_focus_idx),
        metadata_path=metadata_path,
        global_view_state_path=global_view_state_path,
        observer_view_state_path=observer_view_state_path,
        nodes=nodes,
        edges=edges,
        path_count=len(paths),
        provenance=provenance,
        source_artifacts=source_artifacts,
        coordinate_frame={
            "global_space": "in_memory_global_view_state/articles[x,y,z]",
            "observer_space": "in_memory_observer_view_state/articles[x,y,z]",
            "xy_origin_policy": "selected_article_at_observer_xy_origin",
            "translation_null_definition": "global_xyz_minus_global_focus_xy_without_observer_recompute",
        },
        paths=path_rows,
    )


def observer_bundle_to_meaning_probe_inputs(bundle: ObserverManifoldBundle | Mapping[str, Any]) -> Dict[str, Any]:
    payload = bundle.to_dict(include_nodes=True, include_edges=True) if isinstance(bundle, ObserverManifoldBundle) else dict(bundle)
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    global_articles: Dict[int, Dict[str, Any]] = {}
    observer_articles: Dict[int, Dict[str, Any]] = {}
    translation_articles: Dict[int, Dict[str, Any]] = {}
    labels: Dict[int, str] = {}
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        idx = int(raw.get("idx"))
        global_xyz = raw.get("global_xyz") or [0.0, 0.0, 0.0]
        observer_xyz = raw.get("observer_xyz") or [0.0, 0.0, 0.0]
        translation_xyz = raw.get("translation_null_xyz") or [0.0, 0.0, 0.0]
        global_articles[idx] = {"idx": idx, "x": global_xyz[0], "y": global_xyz[1], "z": global_xyz[2]}
        observer_articles[idx] = {"idx": idx, "x": observer_xyz[0], "y": observer_xyz[1], "z": observer_xyz[2]}
        translation_articles[idx] = {"idx": idx, "x": translation_xyz[0], "y": translation_xyz[1], "z": translation_xyz[2]}
        node_labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
        source = _clean_label(node_labels.get("source"))
        if source:
            labels[idx] = source
    return {
        "global_articles": global_articles,
        "observer_articles": observer_articles,
        "translation_articles": translation_articles,
        "labels": labels,
        "anchor_idx": int(payload.get("focus_idx", payload.get("observer_idx", 0))),
    }


def observer_bundle_to_action_graph_inputs(
    bundle: ObserverManifoldBundle | Mapping[str, Any],
    *,
    coordinate_frame: str = "observer_xyz",
) -> Dict[str, Any]:
    payload = bundle.to_dict(include_nodes=True, include_edges=False) if isinstance(bundle, ObserverManifoldBundle) else dict(bundle)
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    ordered = sorted((node for node in nodes if isinstance(node, dict)), key=lambda row: int(row.get("idx", 0)))
    embeddings: List[List[float]] = []
    density: List[float] = []
    stress: List[float] = []
    index_to_row: Dict[int, int] = {}
    node_indices: List[int] = []
    simplex_rows: List[Tuple[float, ...]] = []
    simplex_supported = bool(ordered)
    for raw in ordered:
        idx = int(raw.get("idx"))
        if coordinate_frame not in raw:
            raise ValueError(f"Requested coordinate_frame={coordinate_frame!r} missing for node idx={idx}")
        point = raw.get(coordinate_frame)
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"Node idx={idx} has invalid coordinate_frame={coordinate_frame!r}")
        embeddings.append(
            [
                float(point[0]),
                float(point[1]),
                float(point[2] if len(point) > 2 and point[2] is not None else 0.0),
            ]
        )
        density.append(float(raw.get("density") if raw.get("density") is not None else 1.0))
        stress.append(float(raw.get("stress") if raw.get("stress") is not None else 0.0))
        index_to_row[idx] = len(embeddings) - 1
        node_indices.append(idx)
        simplex = _coerce_optional_tuple(raw.get("observer_simplex"))
        if simplex is None:
            simplex_supported = False
        else:
            simplex_rows.append(simplex)
    out: Dict[str, Any] = {
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "track3_density": np.asarray(density, dtype=np.float32),
        "metric_stress": np.asarray(stress, dtype=np.float32),
        "article_index_to_row": index_to_row,
        "node_indices": node_indices,
        "coordinate_frame": coordinate_frame,
        "observer_simplex_supported": bool(simplex_supported and simplex_rows),
    }
    if simplex_supported and simplex_rows:
        dims = {len(row) for row in simplex_rows}
        if len(dims) != 1:
            raise ValueError("observer_simplex rows must have consistent dimensionality")
        out["observer_simplex"] = np.asarray(simplex_rows, dtype=np.float32)
    return out
