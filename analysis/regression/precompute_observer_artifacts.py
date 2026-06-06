#!/usr/bin/env python3
"""
Precompute observer/article artifact files for Dash browsing.

Creates observer_N/MONOLITH.html files and writes observer_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_article_indices(monolith_csv: Path) -> List[int]:
    if not monolith_csv.exists():
        raise FileNotFoundError(f"Missing MONOLITH_DATA.csv: {monolith_csv}")
    indices: List[int] = []
    with monolith_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row_i, row in enumerate(reader):
            raw = row.get("index", "")
            try:
                idx = int(raw)
            except Exception:
                idx = row_i
            indices.append(idx)
    return indices


def _load_rows(monolith_csv: Path) -> List[dict]:
    if not monolith_csv.exists():
        raise FileNotFoundError(f"Missing MONOLITH_DATA.csv: {monolith_csv}")
    rows: List[dict] = []
    with monolith_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row_i, row in enumerate(reader):
            materialized = dict(row)
            raw = materialized.get("index", "")
            try:
                materialized["index"] = int(raw)
            except Exception:
                materialized["index"] = row_i
            rows.append(materialized)
    return rows


def _select_article_indices(indices: Sequence[int], selected: Optional[Sequence[int]]) -> List[int]:
    if selected is None:
        return list(indices)
    allowed = {int(idx) for idx in selected}
    return [idx for idx in indices if int(idx) in allowed]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists"
        dst.unlink()
    _ensure_parent(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def _render_focused(run_dir: Path, output_path: Path, observer_idx: int, strict: bool) -> None:
    try:
        from run_full_experiment_suite import _ensure_observer_universes_materialized

        result = _ensure_observer_universes_materialized(run_dir, observer_indices=[int(observer_idx)])
        print(f"[observer-precompute] local_recompute={json.dumps(result, ensure_ascii=True)}")
    except Exception as exc:
        raise RuntimeError(
            f"focused observer render requires local observer materialization for observer_{observer_idx}: {exc}"
        ) from exc
    if str(result.get("status", "")).lower() not in {"success", "already_exists"}:
        raise RuntimeError(
            f"focused observer render requires local observer materialization for observer_{observer_idx}; "
            f"got {json.dumps(result, ensure_ascii=True)}"
        )
    _require_focused_observer_universe(run_dir, observer_idx)
    cmd = [
        sys.executable,
        "-m",
        "analysis.MONOLITH_VIZ",
        str(run_dir),
        "--output",
        str(output_path),
        "--observer-idx",
        str(observer_idx),
    ]
    if strict:
        cmd.append("--strict")
    env = os.environ.copy()
    env["MONOLITH_REQUIRE_LOCAL_OBSERVER_RECOMPUTE"] = "1"
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    _validate_focused_view_state(output_path, observer_idx)


def _require_focused_observer_universe(run_dir: Path, observer_idx: int) -> None:
    obs_dir = Path(run_dir) / "relativity_cache" / f"obs_{int(observer_idx)}"
    features_path = obs_dir / "features.npy"
    if not features_path.exists():
        raise RuntimeError(
            f"focused observer artifact requires local recompute features: {features_path}"
        )


def _validate_focused_view_state(output_path: Path, observer_idx: int) -> dict:
    state_path = Path(output_path).with_suffix(".view_state.json")
    if not state_path.exists():
        raise RuntimeError(f"focused observer render did not emit view state: {state_path}")
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"focused observer view state is unreadable: {state_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"focused observer view state root must be an object: {state_path}")
    focus = payload.get("observer_focus") if isinstance(payload.get("observer_focus"), dict) else {}
    try:
        focus_idx = int(focus.get("idx"))
    except Exception as exc:
        raise RuntimeError(f"focused observer view state missing observer_focus.idx: {state_path}") from exc
    if focus_idx != int(observer_idx):
        raise RuntimeError(
            f"focused observer view state idx mismatch: expected {observer_idx}, got {focus_idx}"
        )
    recenter_mode = str(focus.get("recenter_mode", "")).strip()
    local_active = bool(focus.get("local_track_recompute_active"))
    accepted_sidecar = bool(focus.get("accepted_sidecar_mode"))
    if not (local_active and recenter_mode == "local_track_recompute") and not accepted_sidecar:
        raise RuntimeError(
            "focused observer render degraded to non-local observer geometry: "
            f"recenter_mode={recenter_mode!r}, local_track_recompute_active={local_active}"
        )
    return payload


def _emit_relativity_sidecars(run_dir: Path, observer_indices: Sequence[int]) -> dict:
    try:
        import run_full_experiment_suite as suite
        from analysis.verification.scientific_summaries import write_observer_relativity_summary
    except Exception as exc:
        return {"status": "skipped", "reason": f"relativity emit imports unavailable: {exc}"}

    try:
        rows = _load_rows(run_dir / "MONOLITH_DATA.csv")
    except Exception as exc:
        return {"status": "skipped", "reason": f"failed to read MONOLITH rows: {exc}"}

    selected_ids = {int(idx) for idx in observer_indices}
    selected_positions = [
        pos
        for pos, row in enumerate(rows)
        if int(row.get("index", pos)) in selected_ids
    ]
    row_to_article_id = {
        int(pos): int(rows[pos].get("index", pos))
        for pos in selected_positions
    }
    emitted = suite._emit_relativity_defaults(run_dir, rows, observer_indices=selected_positions)
    if str(emitted.get("mode", "")).strip() == "observer_payload_relativity_v1":
        remap = _remap_payload_relativity_sidecars(run_dir, row_to_article_id)
        emitted = dict(emitted)
        emitted["article_id_sidecar_remap"] = remap
    bundle_path = suite._emit_relativity_deltas_json(run_dir)
    summary_payload = {}
    try:
        summary_path = write_observer_relativity_summary(run_dir)
        try:
            summary_payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            if not isinstance(summary_payload, dict):
                summary_payload = {}
        except Exception as exc:
            emitted = dict(emitted)
            emitted["summary_warning"] = f"failed to read observer relativity summary: {exc}"
    except Exception as exc:
        summary_path = None
        emitted = dict(emitted)
        emitted["summary_warning"] = str(exc)
    summary_status = str(summary_payload.get("status", "UNKNOWN") if summary_payload else "MISSING").upper()
    summary_safe = summary_payload.get("safe_for_thesis_claim") if summary_payload else False
    status = "ok"
    if summary_path is None or summary_status in {"INVALID", "NO_DATA", "MISSING", "ERROR"} or summary_safe is False:
        status = "invalid"
    return {
        "status": status,
        "emitted": emitted,
        "bundle_path": str(bundle_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "summary_status": summary_status,
        "summary_safe_for_thesis_claim": bool(summary_safe is True),
        "summary_failure_reasons": summary_payload.get("failure_reasons", []) if isinstance(summary_payload, dict) else [],
    }


def _rewrite_observer_payload_identity(payload: dict, *, old_id: int, new_id: int) -> dict:
    rewritten = dict(payload)
    rewritten["observer_id"] = int(new_id)
    paths = rewritten.get("paths")
    if isinstance(paths, list):
        rewritten["paths"] = [
            str(path).replace(f"observer_{old_id}/", f"observer_{new_id}/")
            for path in paths
        ]
    provenance = rewritten.get("provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance["observer_row_index"] = int(old_id)
        provenance["observer_article_idx"] = int(new_id)
        rewritten["provenance"] = provenance
    return rewritten


def _remap_payload_relativity_sidecars(run_dir: Path, row_to_article_id: dict[int, int]) -> dict:
    rel_dir = run_dir / "relativity_cache"
    if not rel_dir.exists():
        return {"status": "skipped", "reason": "relativity_cache missing"}
    remapped = 0
    missing = []
    for prefix in ("state", "delta"):
        loaded: dict[int, dict] = {}
        for row_idx in sorted(row_to_article_id):
            src = rel_dir / f"{prefix}_{row_idx}.json"
            if not src.exists():
                missing.append(str(src.relative_to(run_dir)))
                continue
            try:
                payload = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            loaded[row_idx] = payload if isinstance(payload, dict) else {}
        destination_paths = {rel_dir / f"{prefix}_{article_id}.json" for article_id in row_to_article_id.values()}
        for row_idx, payload in loaded.items():
            article_id = int(row_to_article_id[row_idx])
            dst = rel_dir / f"{prefix}_{article_id}.json"
            rewritten = _rewrite_observer_payload_identity(payload, old_id=int(row_idx), new_id=article_id)
            dst.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")
            remapped += int(row_idx != article_id)
        for row_idx in sorted(row_to_article_id):
            src = rel_dir / f"{prefix}_{row_idx}.json"
            if row_idx != int(row_to_article_id[row_idx]) and src.exists() and src not in destination_paths:
                src.unlink()
    return {
        "status": "OK",
        "remapped_files": remapped,
        "row_to_article_id": {str(k): int(v) for k, v in row_to_article_id.items()},
        "missing_sources": missing,
    }


def _emit_observer_atlas_bundle(run_dir: Path, manifest_path: Path, manifest: dict, *, max_pairs: int) -> dict:
    if manifest.get("mode") != "focused":
        return {
            "status": "SKIPPED",
            "reason": "observer atlas requires focused observer artifacts",
        }
    try:
        from core.observer_atlas_bundle import build_observer_atlas_bundle_from_paths, write_observer_atlas_bundle
        from core.observer_manifold import build_observer_manifold_bundle, write_observer_manifold_bundle
        from core.observer_slice_transport import write_observer_slice_transport_summary
    except Exception as exc:
        return {"status": "ERROR", "reason": f"observer atlas imports failed: {exc}"}

    bundle_paths: List[Path] = []
    errors: List[str] = []
    for row in manifest.get("observers", []):
        if not isinstance(row, dict) or not row.get("exists"):
            continue
        action = str(row.get("action", "")).strip().lower()
        if action in {"link", "copy", "hardlink"}:
            continue
        rel_path = row.get("relative_path")
        if not rel_path:
            continue
        observer_dir = (run_dir / str(rel_path)).parent
        try:
            bundle = build_observer_manifold_bundle(run_dir, observer_dir)
            written = write_observer_manifold_bundle(bundle, run_dir)
            bundle_paths.append(Path(written["bundle_json"]))
        except Exception as exc:
            errors.append(f"{observer_dir.name}: {exc}")

    if not bundle_paths:
        return {
            "status": "NO_BUNDLES",
            "reason": "no focused observer manifold bundles could be built",
            "errors": errors,
        }

    try:
        atlas = build_observer_atlas_bundle_from_paths(
            bundle_paths,
            observer_manifest_path=manifest_path,
            run_dir=run_dir,
            max_article_pairs=max_pairs,
            require_focused_observer_artifacts=True,
        )
        written = write_observer_atlas_bundle(atlas, run_dir)
        transport_summary = atlas.get("transport_summary") if isinstance(atlas.get("transport_summary"), dict) else {}
        transport_written = write_observer_slice_transport_summary(transport_summary, run_dir) if transport_summary else {}
    except Exception as exc:
        return {
            "status": "ERROR",
            "reason": f"observer atlas build failed: {exc}",
            "bundle_count": len(bundle_paths),
            "errors": errors,
        }

    return {
        "status": "OK",
        "bundle_count": len(bundle_paths),
        "atlas_bundle": str(Path(written["atlas_bundle"]).relative_to(run_dir)).replace("\\", "/"),
        "transport_summary": str(Path(transport_written["summary"]).relative_to(run_dir)).replace("\\", "/") if transport_written else None,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute observer artifacts and manifest for Dash.")
    parser.add_argument("run_dir", type=Path, help="Path to run folder containing MONOLITH_DATA.csv")
    parser.add_argument("--variant", default="MONOLITH.html", help="Variant filename to materialize")
    parser.add_argument(
        "--mode",
        choices=["link", "copy", "focused"],
        default="link",
        help="link/copy reuses existing global artifact; focused renders one file per observer index.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing observer files.")
    parser.add_argument("--strict", action="store_true", help="Pass strict validation to focused renders.")
    parser.add_argument(
        "--observer-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional article indices to materialize; omitted means all observers.",
    )
    parser.add_argument(
        "--atlas-max-pairs",
        type=int,
        default=32,
        help="Maximum article edge pairs to include when emitting observer_atlas_bundle.json.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    indices = _select_article_indices(_load_article_indices(monolith_csv), args.observer_indices)
    variant_name = str(args.variant)
    global_variant = run_dir / variant_name

    if not global_variant.exists() and args.mode in {"link", "copy"}:
        raise FileNotFoundError(
            f"Global artifact missing: {global_variant}. Generate it first or use --mode focused."
        )

    records = []
    for idx in indices:
        out_path = run_dir / f"observer_{idx}" / variant_name
        if args.mode == "focused":
            _ensure_parent(out_path)
            if out_path.exists() and not args.overwrite:
                action = "exists"
            else:
                if out_path.exists():
                    out_path.unlink()
                _render_focused(run_dir, out_path, idx, args.strict)
                action = "rendered"
        elif args.mode == "copy":
            action = _link_or_copy(global_variant, out_path, args.overwrite)
            if action == "hardlink":
                action = "copy"
        else:
            action = _link_or_copy(global_variant, out_path, args.overwrite)

        records.append(
            {
                "idx": idx,
                "value": f"article:{idx}",
                "relative_path": str(out_path.relative_to(run_dir)).replace("\\", "/"),
                "view_state_relative_path": str(out_path.with_suffix(".view_state.json").relative_to(run_dir)).replace("\\", "/")
                if out_path.with_suffix(".view_state.json").exists()
                else None,
                "exists": out_path.exists(),
                "action": action,
            }
        )

    relativity_result = _emit_relativity_sidecars(run_dir, indices)

    found = sum(1 for r in records if r["exists"])
    total = len(records)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "variant": variant_name,
        "mode": args.mode,
        "n_articles": total,
        "coverage": {
            "found": found,
            "total": total,
            "pct": (float(found) / float(total)) if total else 0.0,
        },
        "relativity": relativity_result,
        "observers": records,
    }

    manifest_path = run_dir / "observer_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    atlas_result = _emit_observer_atlas_bundle(
        run_dir,
        manifest_path,
        manifest,
        max_pairs=max(1, int(args.atlas_max_pairs)),
    )
    if atlas_result:
        manifest["observer_atlas"] = atlas_result
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"[observer-precompute] run={run_dir}")
    print(f"[observer-precompute] mode={args.mode} variant={variant_name}")
    print(f"[observer-precompute] coverage={found}/{total}")
    print(f"[observer-precompute] manifest={manifest_path}")
    if relativity_result:
        print(f"[observer-precompute] relativity={json.dumps(relativity_result, ensure_ascii=True)}")
    if atlas_result:
        print(f"[observer-precompute] observer_atlas={json.dumps(atlas_result, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
