"""Observer Atlas bundle generation for Dash.

The Atlas bundle is a visualization contract, not a new scientific metric.
It joins observer-conditioned manifold slices to observer-slice transport
records so Dash can render semantic-first vs observer-first routes spatially.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.observer_manifold import ObserverManifoldBundle, read_observer_manifold_bundle
from core.observer_slice_transport import summarize_observer_slice_transport


ATLAS_SCHEMA_VERSION = 1
ATLAS_BUNDLE_TYPE = "observer_atlas_bundle"


class ObserverAtlasError(ValueError):
    """Raised when Atlas artifacts cannot be safely constructed."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ObserverAtlasError(f"JSON root must be an object: {path}")
    return loaded


def _bundle_payload(bundle: ObserverManifoldBundle | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(bundle, ObserverManifoldBundle):
        return bundle.to_dict(include_nodes=True, include_edges=True)
    return dict(bundle)


def _finite_xyz(values: Any) -> Tuple[float, float, float]:
    raw = list(values) if isinstance(values, (list, tuple, np.ndarray)) else []
    if len(raw) < 2:
        raise ObserverAtlasError(f"coordinate must contain at least x/y, got {values!r}")
    out = [
        float(raw[0]),
        float(raw[1]),
        float(raw[2] if len(raw) > 2 and raw[2] is not None else 0.0),
    ]
    if not all(math.isfinite(v) for v in out):
        raise ObserverAtlasError(f"coordinate contains non-finite values: {values!r}")
    return tuple(out)  # type: ignore[return-value]


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _source_fingerprints(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        out[path.name] = _file_fingerprint(path)
    return out


def _manifest_actions(observer_manifest: Optional[Mapping[str, Any]]) -> Dict[int, str]:
    if not isinstance(observer_manifest, Mapping):
        return {}
    observers = observer_manifest.get("observers")
    if not isinstance(observers, list):
        return {}
    out: Dict[int, str] = {}
    for row in observers:
        if not isinstance(row, Mapping):
            continue
        try:
            idx = int(row.get("idx"))
        except Exception:
            continue
        out[idx] = str(row.get("action", "")).strip().lower()
    return out


@dataclass(frozen=True)
class _SliceFrame:
    slice_id: str
    label: str
    role: str
    nodes: List[Dict[str, Any]]
    coordinates: np.ndarray


def _node_label(raw: Mapping[str, Any], article_idx: int) -> str:
    labels = raw.get("labels") if isinstance(raw.get("labels"), Mapping) else {}
    for key in ("label", "source", "publication", "perspective_tag", "verdict"):
        value = labels.get(key)
        if value:
            return str(value)
    return f"Article {article_idx}"


def _build_slice(
    *,
    slice_id: str,
    label: str,
    role: str,
    payload: Mapping[str, Any],
    coordinate_frame: str,
    row_article_ids: Sequence[int],
) -> _SliceFrame:
    raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    by_idx: Dict[int, Mapping[str, Any]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            continue
        try:
            by_idx[int(raw.get("idx"))] = raw
        except Exception:
            continue

    nodes: List[Dict[str, Any]] = []
    coords: List[Tuple[float, float, float]] = []
    for row_index, article_idx in enumerate(row_article_ids):
        raw = by_idx.get(int(article_idx))
        if raw is None:
            raise ObserverAtlasError(f"slice {slice_id!r} missing article_idx={article_idx}")
        if coordinate_frame not in raw:
            raise ObserverAtlasError(
                f"slice {slice_id!r} missing coordinate_frame={coordinate_frame!r} for article_idx={article_idx}"
            )
        xyz = _finite_xyz(raw.get(coordinate_frame))
        coords.append(xyz)
        nodes.append(
            {
                "row_index": int(row_index),
                "article_idx": int(article_idx),
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "zone": str(raw.get("zone", "") or ""),
                "density": raw.get("density"),
                "stress": raw.get("stress"),
                "label": _node_label(raw, int(article_idx)),
                "labels": dict(raw.get("labels") if isinstance(raw.get("labels"), Mapping) else {}),
                "nontranslation_shift": raw.get("nontranslation_shift"),
            }
        )
    return _SliceFrame(
        slice_id=slice_id,
        label=label,
        role=role,
        nodes=nodes,
        coordinates=np.asarray(coords, dtype=np.float64),
    )


def _common_article_ids(payloads: Sequence[Mapping[str, Any]]) -> List[int]:
    common: Optional[set[int]] = None
    for payload in payloads:
        raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        ids: set[int] = set()
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                continue
            try:
                ids.add(int(raw.get("idx")))
            except Exception:
                continue
        common = ids if common is None else common.intersection(ids)
    if not common:
        raise ObserverAtlasError("observer manifold bundles have no common article nodes")
    return sorted(common)


def _article_pairs_from_edges(
    payloads: Sequence[Mapping[str, Any]],
    article_to_row: Mapping[int, int],
    *,
    max_pairs: int,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for payload in payloads:
        edges = payload.get("edge_action_ledger") if isinstance(payload.get("edge_action_ledger"), list) else []
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            try:
                source_article = int(edge.get("source_idx"))
                target_article = int(edge.get("target_idx"))
            except Exception:
                continue
            if source_article not in article_to_row or target_article not in article_to_row:
                continue
            pair = (int(article_to_row[source_article]), int(article_to_row[target_article]))
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _fallback_article_pairs(row_count: int, *, max_pairs: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for idx in range(max(0, row_count - 1)):
        pairs.append((idx, idx + 1))
        if len(pairs) >= max_pairs:
            break
    return pairs


def _resolve_legacy_row_index(value: Any, row_article_ids: Sequence[int], article_to_row: Mapping[int, int]) -> int:
    raw = int(value)
    is_row = 0 <= raw < len(row_article_ids)
    is_article = raw in article_to_row
    if is_row and is_article and int(row_article_ids[raw]) != raw:
        raise ObserverAtlasError(
            f"transport record index {raw} is ambiguous; provide explicit row_index or article_idx"
        )
    if is_row:
        return raw
    if is_article:
        return int(article_to_row[raw])
    raise ObserverAtlasError(f"transport record index {raw} is neither row index nor known article_idx")


def _resolve_record_row(
    record: Mapping[str, Any],
    prefix: str,
    row_article_ids: Sequence[int],
    article_to_row: Mapping[int, int],
) -> int:
    row_key = f"{prefix}_row_index"
    article_key = f"{prefix}_article_idx"
    legacy_key = f"{prefix}_idx"
    if row_key in record:
        row = int(record.get(row_key))
        if not (0 <= row < len(row_article_ids)):
            raise ObserverAtlasError(f"{row_key}={row} outside atlas row range")
        return row
    if article_key in record:
        article_idx = int(record.get(article_key))
        if article_idx not in article_to_row:
            raise ObserverAtlasError(f"{article_key}={article_idx} missing from atlas article map")
        return int(article_to_row[article_idx])
    return _resolve_legacy_row_index(record.get(legacy_key), row_article_ids, article_to_row)


def _route_points_for_record(
    record: Mapping[str, Any],
    *,
    slice_frames: Mapping[str, _SliceFrame],
    aliases: Mapping[str, str],
    row_article_ids: Sequence[int],
    article_to_row: Mapping[int, int],
) -> Dict[str, Any]:
    source_slice_raw = str(record.get("source_slice", ""))
    target_slice_raw = str(record.get("target_slice", ""))
    source_slice = aliases.get(source_slice_raw, source_slice_raw)
    target_slice = aliases.get(target_slice_raw, target_slice_raw)
    if source_slice not in slice_frames or target_slice not in slice_frames:
        raise ObserverAtlasError(f"transport record references unknown slices: {source_slice_raw!r}->{target_slice_raw!r}")

    source_row = _resolve_record_row(record, "source", row_article_ids, article_to_row)
    target_row = _resolve_record_row(record, "target", row_article_ids, article_to_row)
    source_article = int(row_article_ids[source_row])
    target_article = int(row_article_ids[target_row])
    source_frame = slice_frames[source_slice]
    target_frame = slice_frames[target_slice]

    a = source_frame.coordinates[source_row]
    b = source_frame.coordinates[target_row]
    c = target_frame.coordinates[target_row]
    d = target_frame.coordinates[source_row]
    semantic_points = [a, b, c]
    observer_points = [a, d, c]
    closed_loop_points = [a, b, c, d, a]
    return {
        "source_row_index": int(source_row),
        "target_row_index": int(target_row),
        "source_article_idx": source_article,
        "target_article_idx": target_article,
        "source_slice": source_slice,
        "target_slice": target_slice,
        "source_slice_original": source_slice_raw,
        "target_slice_original": target_slice_raw,
        "semantic_first": {
            "action": record.get("semantic_first_action"),
            "components": dict(record.get("semantic_first_components") if isinstance(record.get("semantic_first_components"), Mapping) else {}),
            "points": _json_safe(semantic_points),
        },
        "observer_first": {
            "action": record.get("observer_first_action"),
            "components": dict(record.get("observer_first_components") if isinstance(record.get("observer_first_components"), Mapping) else {}),
            "points": _json_safe(observer_points),
        },
        "closed_loop_points": _json_safe(closed_loop_points),
        "commutator_gap": record.get("commutator_gap"),
        "holonomy_action": record.get("holonomy_action"),
        "relative_holonomy": record.get("relative_holonomy"),
    }


def _null_lookup(
    null_records: Any,
    *,
    row_article_ids: Sequence[int],
    article_to_row: Mapping[int, int],
    aliases: Mapping[str, str],
) -> Dict[Tuple[int, int, str, str], Mapping[str, Any]]:
    out: Dict[Tuple[int, int, str, str], Mapping[str, Any]] = {}
    if not isinstance(null_records, list):
        return out
    for row in null_records:
        if not isinstance(row, Mapping):
            continue
        try:
            source_row = _resolve_record_row(row, "source", row_article_ids, article_to_row)
            target_row = _resolve_record_row(row, "target", row_article_ids, article_to_row)
            source_slice_raw = str(row.get("source_slice", ""))
            target_slice_raw = str(row.get("target_slice", ""))
            key = (
                int(source_row),
                int(target_row),
                aliases.get(source_slice_raw, source_slice_raw),
                aliases.get(target_slice_raw, target_slice_raw),
            )
        except Exception:
            continue
        out[key] = row
    return out


def _decorate_routes(
    transport_summary: Mapping[str, Any],
    *,
    slice_frames: Mapping[str, _SliceFrame],
    aliases: Mapping[str, str],
    row_article_ids: Sequence[int],
    article_to_row: Mapping[int, int],
) -> List[Dict[str, Any]]:
    records = transport_summary.get("records") if isinstance(transport_summary.get("records"), list) else []
    nulls = _null_lookup(
        transport_summary.get("null_records"),
        row_article_ids=row_article_ids,
        article_to_row=article_to_row,
        aliases=aliases,
    )
    routes: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        route = _route_points_for_record(
            record,
            slice_frames=slice_frames,
            aliases=aliases,
            row_article_ids=row_article_ids,
            article_to_row=article_to_row,
        )
        null_key = (
            int(route.get("source_row_index")),
            int(route.get("target_row_index")),
            str(route.get("source_slice", "")),
            str(route.get("target_slice", "")),
        )
        null_record = nulls.get(null_key, {})
        null_holonomy = null_record.get("holonomy_action") if isinstance(null_record, Mapping) else None
        try:
            route["null_holonomy_action"] = float(null_holonomy) if null_holonomy is not None else None
        except Exception:
            route["null_holonomy_action"] = None
        try:
            route["excess_holonomy_action"] = float(route.get("holonomy_action") or 0.0) - float(
                route.get("null_holonomy_action") or 0.0
            )
        except Exception:
            route["excess_holonomy_action"] = None
        routes.append(_json_safe(route))
    return routes


def _atlas_metrics(routes: Sequence[Mapping[str, Any]], transport_summary: Mapping[str, Any]) -> Dict[str, Any]:
    holonomies = [
        float(row.get("holonomy_action"))
        for row in routes
        if row.get("holonomy_action") is not None and math.isfinite(float(row.get("holonomy_action")))
    ]
    excess = [
        float(row.get("excess_holonomy_action"))
        for row in routes
        if row.get("excess_holonomy_action") is not None and math.isfinite(float(row.get("excess_holonomy_action")))
    ]
    return {
        "record_count": int(len(routes)),
        "mean_holonomy_action": float(np.mean(holonomies)) if holonomies else transport_summary.get("mean_holonomy_action"),
        "max_holonomy_action": float(np.max(holonomies)) if holonomies else transport_summary.get("max_holonomy_action"),
        "mean_null_holonomy_action": transport_summary.get("mean_null_holonomy_action"),
        "mean_excess_holonomy_action": float(np.mean(excess)) if excess else transport_summary.get("mean_excess_holonomy_action"),
    }


def build_observer_atlas_bundle(
    manifold_bundles: Sequence[ObserverManifoldBundle | Mapping[str, Any]],
    *,
    transport_summary: Optional[Mapping[str, Any]] = None,
    run_dir: str | Path | None = None,
    coordinate_frame: str = "observer_xyz",
    max_article_pairs: int = 32,
    source_artifact_paths: Sequence[str | Path] = (),
    observer_manifest: Optional[Mapping[str, Any]] = None,
    require_focused_observer_artifacts: bool = True,
) -> Dict[str, Any]:
    """Build a run-level Observer Atlas bundle from observer manifold bundles."""

    payloads = [_bundle_payload(bundle) for bundle in manifold_bundles]
    if not payloads:
        raise ObserverAtlasError("at least one observer manifold bundle is required")

    manifest_actions = _manifest_actions(observer_manifest)
    if require_focused_observer_artifacts and manifest_actions:
        for payload in payloads:
            observer_idx = int(payload.get("observer_idx", payload.get("focus_idx", 0)))
            action = manifest_actions.get(observer_idx, "")
            if action in {"link", "copy", "hardlink"}:
                raise ObserverAtlasError(
                    f"observer_{observer_idx} action={action!r} reuses global MONOLITH output; focused artifact required"
                )
            provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
            recenter_mode = str(provenance.get("recenter_mode", "") or "").strip()
            local_active = bool(provenance.get("local_track_recompute_active"))
            accepted_sidecar = bool(provenance.get("accepted_sidecar_mode"))
            if recenter_mode and not (local_active and recenter_mode == "local_track_recompute") and not accepted_sidecar:
                raise ObserverAtlasError(
                    f"observer_{observer_idx} view state is not a focused local recompute artifact "
                    f"(recenter_mode={recenter_mode!r}, local_track_recompute_active={local_active})"
                )

    row_article_ids = _common_article_ids(payloads)
    article_to_row = {int(article_idx): int(pos) for pos, article_idx in enumerate(row_article_ids)}
    reference = payloads[0]
    run_dir_value = str(run_dir or reference.get("run_dir", ""))

    global_frame = _build_slice(
        slice_id="global",
        label="Global",
        role="global",
        payload=reference,
        coordinate_frame="global_xyz",
        row_article_ids=row_article_ids,
    )
    translation_frame = _build_slice(
        slice_id="translation_null",
        label="Translation Null",
        role="null",
        payload=reference,
        coordinate_frame="translation_null_xyz",
        row_article_ids=row_article_ids,
    )
    frames: Dict[str, _SliceFrame] = {
        global_frame.slice_id: global_frame,
        translation_frame.slice_id: translation_frame,
    }
    aliases: Dict[str, str] = {
        "global": "global",
        "translation_null": "translation_null",
    }
    observer_slice_names: List[str] = []
    for pos, payload in enumerate(payloads):
        observer_idx = int(payload.get("observer_idx", payload.get("focus_idx", pos)))
        focus_idx = int(payload.get("focus_idx", observer_idx))
        slice_id = f"observer_{observer_idx}"
        label = f"Observer {observer_idx}"
        frames[slice_id] = _build_slice(
            slice_id=slice_id,
            label=label,
            role="observer",
            payload=payload,
            coordinate_frame=coordinate_frame,
            row_article_ids=row_article_ids,
        )
        observer_slice_names.append(slice_id)
        aliases[f"observer_{focus_idx}"] = slice_id
        aliases[str(observer_idx)] = slice_id
        if len(payloads) == 1:
            aliases["observer_recomputed"] = slice_id

    arrays = {name: frame.coordinates for name, frame in frames.items()}
    transport_was_provided = transport_summary is not None
    article_pair_source = "provided_transport_summary"
    if transport_summary is None:
        pairs = _article_pairs_from_edges(payloads, article_to_row, max_pairs=max_article_pairs)
        article_pair_source = "observer_edge_action_ledger"
        if not pairs:
            pairs = _fallback_article_pairs(len(row_article_ids), max_pairs=max_article_pairs)
            article_pair_source = "fallback_adjacent_row_pairs_no_edge_action_ledger"
        route_slices = ["global", *observer_slice_names]
        slice_pairs = [
            (left, right)
            for left in route_slices
            for right in route_slices
            if left != right
        ]
        null_slices = {name: frames["translation_null"].coordinates for name in route_slices}
        density = [reference_node.get("density", 1.0) for reference_node in frames["global"].nodes]
        stress = [reference_node.get("stress", 0.0) for reference_node in frames["global"].nodes]
        transport_summary = summarize_observer_slice_transport(
            {name: arrays[name] for name in route_slices},
            article_pairs=pairs,
            slice_pairs=slice_pairs,
            density=density,
            stress=stress,
            null_slices=null_slices,
            row_to_article_index=row_article_ids,
        )

    routes = _decorate_routes(
        transport_summary,
        slice_frames=frames,
        aliases=aliases,
        row_article_ids=row_article_ids,
        article_to_row=article_to_row,
    )
    source_paths = [Path(p) for p in source_artifact_paths]
    payload = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "bundle_type": ATLAS_BUNDLE_TYPE,
        "run_dir": run_dir_value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": coordinate_frame,
        "row_count": len(row_article_ids),
        "row_article_ids": [int(idx) for idx in row_article_ids],
        "article_idx_to_row": {str(article_idx): row for article_idx, row in article_to_row.items()},
        "source_artifacts": {
            "paths": [str(path) for path in source_paths],
            "fingerprints": _source_fingerprints(source_paths),
        },
        "slices": [
            {
                "slice_id": frame.slice_id,
                "label": frame.label,
                "role": frame.role,
                "node_count": len(frame.nodes),
                "nodes": frame.nodes,
            }
            for frame in frames.values()
        ],
        "routes": routes,
        "metrics": _atlas_metrics(routes, transport_summary),
        "transport_summary": dict(transport_summary),
        "provenance": {
            "builder": "core.observer_atlas_bundle.build_observer_atlas_bundle",
            "requires_focused_observer_artifacts": bool(require_focused_observer_artifacts),
            "coordinate_source": "observer_manifold_bundle",
            "transport_source": "provided" if transport_was_provided else "generated",
            "article_pair_source": article_pair_source,
        },
    }
    return _json_safe(payload)


def build_observer_atlas_bundle_from_paths(
    bundle_paths: Sequence[str | Path],
    *,
    transport_summary_path: str | Path | None = None,
    observer_manifest_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    coordinate_frame: str = "observer_xyz",
    max_article_pairs: int = 32,
    require_focused_observer_artifacts: bool = True,
) -> Dict[str, Any]:
    bundles = [read_observer_manifold_bundle(Path(path)) for path in bundle_paths]
    transport_summary = _load_json(Path(transport_summary_path)) if transport_summary_path else None
    observer_manifest = _load_json(Path(observer_manifest_path)) if observer_manifest_path else None
    source_paths: List[Path] = [Path(path) for path in bundle_paths]
    if transport_summary_path:
        source_paths.append(Path(transport_summary_path))
    if observer_manifest_path:
        source_paths.append(Path(observer_manifest_path))
    return build_observer_atlas_bundle(
        bundles,
        transport_summary=transport_summary,
        run_dir=run_dir,
        coordinate_frame=coordinate_frame,
        max_article_pairs=max_article_pairs,
        source_artifact_paths=source_paths,
        observer_manifest=observer_manifest,
        require_focused_observer_artifacts=require_focused_observer_artifacts,
    )


def read_observer_atlas_bundle(path: str | Path) -> Dict[str, Any]:
    payload = _load_json(Path(path))
    if payload.get("bundle_type") != ATLAS_BUNDLE_TYPE:
        raise ObserverAtlasError(f"Not an observer Atlas bundle: {path}")
    return payload


def write_observer_atlas_bundle(bundle: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "observer_atlas_bundle.json"
    path.write_text(json.dumps(_json_safe(dict(bundle)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"atlas_bundle": str(path)}
