import json
import sys
import types
from pathlib import Path

import pytest
import torch

from core.ablation_dag import (
    AblationDag,
    NodeSpec,
    build_orchestration_contract,
    fingerprint_path,
)
from core.master_ablation import AblationConfig, AblationRunner


def _write_corpus(path: Path, count: int = 2) -> None:
    rows = [
        {"title": f"Article {idx}", "text": f"Body {idx}"}
        for idx in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _install_fake_modules(monkeypatch, counters):
    fake_pipeline = types.ModuleType("core.complete_pipeline")

    def run_multi_observer_experiment_simple(*, seeds, output_dir, **kwargs):
        counters["experiment"] += 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            torch.save(
                {
                    "features": torch.tensor([[float(seed)], [float(seed + 1)]]),
                    "T1_embeddings": [1.0],
                },
                output_dir / f"observer_{seed}.pt",
            )
        checkpoint_dir = output_dir / "checkpoints" / "real_batch"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "manifest.json").write_text(json.dumps({"run": "ok"}), encoding="utf-8")
        (checkpoint_dir / "T1.5_spectral_state.npz").write_bytes(b"npz")
        (checkpoint_dir / "T2_kernel_projections.npz").write_bytes(b"npz")
        (checkpoint_dir / "T2_kernel_meta.json").write_text(json.dumps({"sigma": 1.0}), encoding="utf-8")
        (checkpoint_dir / "T3_topology.npz").write_bytes(b"npz")
        (checkpoint_dir / "T3_topology.json").write_text(json.dumps({"nmi_score": 0.82}), encoding="utf-8")
        (output_dir / "features.npy").write_bytes(b"npy")
        (output_dir / "MONOLITH_DATA.csv").write_text("bt_uid,density,stress\nu0,0.5,0.5\n", encoding="utf-8")
        (output_dir / "phantom_verdicts.json").write_text(json.dumps([{"verdict": "HONEST"}]), encoding="utf-8")
        (output_dir / "walker_work_integrals.npy").write_bytes(b"npy")

    fake_pipeline.run_multi_observer_experiment_simple = run_multi_observer_experiment_simple
    monkeypatch.setitem(sys.modules, "core.complete_pipeline", fake_pipeline)

    fake_suite = types.ModuleType("run_full_experiment_suite")

    def emit_results_json(run_dir, payload):
        out = Path(run_dir) / "ablation_results.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def translate_lab_diagnostics(blob, source_path):
        return {
            "status": "OK",
            "message": "translated from lab_diagnostics.json",
            "metrics": {"stage_3_nmi": blob.get("stage_3_nmi", 0.75)},
            "source_path": str(source_path),
        }

    def normalize_run_provenance(result, run_meta):
        return {"run_meta": run_meta}

    def emit_consumer_contract_bundle(run_dir):
        counters["bundle"] += 1
        run_dir = Path(run_dir)
        (run_dir / "baseline_meta.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        (run_dir / "validation.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return {"status": "success", "bundle_dir": str(run_dir)}

    fake_suite._emit_ablation_results_json = emit_results_json
    fake_suite._translate_lab_diagnostics_to_ablation_summary = translate_lab_diagnostics
    fake_suite._normalize_run_provenance = normalize_run_provenance
    fake_suite.emit_consumer_contract_bundle = emit_consumer_contract_bundle
    monkeypatch.setitem(sys.modules, "run_full_experiment_suite", fake_suite)


def _install_runner_stubs(monkeypatch, counters):
    def fake_relativity(self, corpus_dir, articles, config, corpus_name, normalize_run_provenance):
        counters["relativity"] += 1
        corpus_dir = Path(corpus_dir)
        rel_dir = corpus_dir / "relativity_cache"
        rel_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.tensor([[1.0]])}, corpus_dir / "observer_global.pt")
        for idx, _article in enumerate(articles):
            torch.save({"observer_id": idx}, rel_dir / f"observer_{idx}.pt")
        return {
            "status": "success",
            "observer_payloads": len(articles),
            "global_payload": str(corpus_dir / "observer_global.pt"),
        }

    def fake_compute(self, observer_payloads):
        counters["diagnostics"] += 1
        return {
            "stage_3_nmi": 0.82,
            "n_loaded": len(observer_payloads),
        }

    monkeypatch.setattr(AblationRunner, "_materialize_relativity_payloads", fake_relativity)
    monkeypatch.setattr(AblationRunner, "_compute_lab_diagnostics", fake_compute)


def _build_config(tmp_path: Path) -> AblationConfig:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path)
    return AblationConfig(
        run_name="dag_probe",
        output_dir=str(tmp_path / "outputs"),
        corpus_path=str(corpus_path),
        control_paths={},
        corpora=["real"],
        observer_seeds=[11, 22],
        max_articles=2,
        device="cpu",
    )


def test_node_cache_key_is_stable_and_tracks_corpus_fingerprint(tmp_path):
    config = _build_config(tmp_path)
    runner = AblationRunner(config)
    corpus = Path(config.corpus_path)

    key_a = runner._node_cache_key(
        config=config,
        corpus_name="real",
        corpus_fingerprint=fingerprint_path(corpus),
        node_name="experiment",
    )
    key_b = runner._node_cache_key(
        config=config,
        corpus_name="real",
        corpus_fingerprint=fingerprint_path(corpus),
        node_name="experiment",
    )
    assert key_a == key_b

    corpus.write_text(
        corpus.read_text(encoding="utf-8") + "\n" + json.dumps({"title": "Extra", "text": "Changed"}),
        encoding="utf-8",
    )
    key_c = runner._node_cache_key(
        config=config,
        corpus_name="real",
        corpus_fingerprint=fingerprint_path(corpus),
        node_name="experiment",
    )
    assert key_c != key_a


def test_run_single_writes_dag_manifests_and_reuses_completed_nodes(monkeypatch, tmp_path):
    counters = {"experiment": 0, "relativity": 0, "diagnostics": 0, "bundle": 0}
    _install_fake_modules(monkeypatch, counters)
    _install_runner_stubs(monkeypatch, counters)
    config = _build_config(tmp_path)
    runner = AblationRunner(config)

    first_manifest = runner.run_single(config)
    second_manifest = runner.run_single(config)

    assert counters == {"experiment": 1, "relativity": 1, "diagnostics": 1, "bundle": 1}
    assert first_manifest["run_status"] == "completed"
    assert second_manifest["run_status"] == "completed"
    assert second_manifest["results"]["real"]["dag"]["experiment"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["input_resolution"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["track15_extract"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["track5_assembly"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["diagnostics"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["airflow_csv_ablation"]["status"] == "cached"

    run_dir = Path(config.output_dir) / config.run_name
    assert (run_dir / "real" / "airflow_ablation" / "manifold_ablation_summary.json").exists()
    airflow_manifest_path = run_dir / "real" / "airflow_ablation" / "airflow_ablation_manifest.json"
    assert airflow_manifest_path.exists()
    airflow_manifest = json.loads(airflow_manifest_path.read_text(encoding="utf-8"))
    assert airflow_manifest["base_csv_fingerprint"]["path"].endswith("MONOLITH_DATA.csv")
    assert airflow_manifest["sidecar"] == "analysis.airflow_ablation_orchestrator.run_ablation_matrix"
    status_blob = json.loads((run_dir / "_ablation_dag" / "status.json").read_text(encoding="utf-8"))
    assert status_blob["status"] == "completed"

    experiment_manifest = json.loads(
        (run_dir / "_ablation_dag" / "nodes" / "real_experiment.json").read_text(encoding="utf-8")
    )
    assert experiment_manifest["status"] == "completed"
    assert experiment_manifest["cache_key"]
    assert experiment_manifest["outputs_present"] is True
    input_manifest = json.loads(
        (run_dir / "_ablation_dag" / "nodes" / "real_input_resolution.json").read_text(encoding="utf-8")
    )
    assert input_manifest["status"] == "completed"
    assert input_manifest["outputs_present"] is True
    track15_manifest = json.loads(
        (run_dir / "_ablation_dag" / "nodes" / "real_track15_extract.json").read_text(encoding="utf-8")
    )
    assert track15_manifest["status"] == "completed"
    assert track15_manifest["outputs_present"] is True


def test_run_single_records_missing_requested_corpus_without_plain_completed_manifest(tmp_path):
    config = _build_config(tmp_path)
    missing_path = tmp_path / "missing.jsonl"
    config.corpora = ["missing_control"]
    config.control_paths = {"missing_control": str(missing_path)}
    runner = AblationRunner(config)

    manifest = runner.run_single(config)

    assert manifest["run_status"] == "completed_with_missing_corpora"
    assert manifest["corpus_resolution"]["requested"] == ["missing_control"]
    assert manifest["corpus_resolution"]["resolved"] == {}
    assert manifest["corpus_resolution"]["missing"] == [
        {
            "corpus": "missing_control",
            "requested_path": str(missing_path),
            "reason": "path_not_found",
        }
    ]
    assert manifest["results"]["missing_control"]["status"] == "missing_corpus"
    persisted_manifest = json.loads(
        (Path(config.output_dir) / config.run_name / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["run_status"] == "completed_with_missing_corpora"
    assert persisted_manifest["results"] == manifest["results"]
    assert all(result["status"] == "missing_corpus" for result in persisted_manifest["results"].values())


def test_run_single_mixed_success_and_missing_corpus_is_not_plain_completed(monkeypatch, tmp_path):
    counters = {"experiment": 0, "relativity": 0, "diagnostics": 0, "bundle": 0}
    _install_fake_modules(monkeypatch, counters)
    _install_runner_stubs(monkeypatch, counters)
    config = _build_config(tmp_path)
    missing_path = tmp_path / "missing.jsonl"
    config.corpora = ["real", "missing_control"]
    config.control_paths = {"missing_control": str(missing_path)}
    runner = AblationRunner(config)

    manifest = runner.run_single(config)

    assert counters == {"experiment": 1, "relativity": 1, "diagnostics": 1, "bundle": 1}
    assert manifest["run_status"] == "completed_with_missing_corpora"
    assert "real" in manifest["results"]
    assert manifest["results"]["missing_control"]["status"] == "missing_corpus"
    assert manifest["corpus_resolution"]["resolved"] == {"real": config.corpus_path}
    assert manifest["corpus_resolution"]["missing"][0]["corpus"] == "missing_control"


def test_run_single_resumes_incomplete_node_without_rerunning_completed_upstream(monkeypatch, tmp_path):
    counters = {"experiment": 0, "relativity": 0, "diagnostics": 0, "bundle": 0}
    _install_fake_modules(monkeypatch, counters)
    _install_runner_stubs(monkeypatch, counters)
    config = _build_config(tmp_path)
    runner = AblationRunner(config)

    runner.run_single(config)

    run_dir = Path(config.output_dir) / config.run_name
    diagnostics_manifest_path = run_dir / "_ablation_dag" / "nodes" / "real_diagnostics.json"
    diagnostics_manifest = json.loads(diagnostics_manifest_path.read_text(encoding="utf-8"))
    diagnostics_manifest["status"] = "running"
    diagnostics_manifest.pop("finished_at", None)
    diagnostics_manifest_path.write_text(json.dumps(diagnostics_manifest, indent=2), encoding="utf-8")

    for artifact in (
        run_dir / "real" / "lab_diagnostics.json",
        run_dir / "real" / "ablation_summary.json",
        run_dir / "real" / "ablation_results.json",
    ):
        artifact.unlink()

    resumed_manifest = runner.run_single(config)

    assert counters["experiment"] == 1
    assert counters["relativity"] == 1
    assert counters["diagnostics"] == 2
    assert counters["bundle"] == 1
    assert resumed_manifest["results"]["real"]["dag"]["input_resolution"]["status"] == "cached"
    assert resumed_manifest["results"]["real"]["dag"]["experiment"]["status"] == "cached"
    assert resumed_manifest["results"]["real"]["dag"]["track2_projection"]["status"] == "cached"
    assert resumed_manifest["results"]["real"]["dag"]["diagnostics"]["status"] == "completed"

    diagnostics_manifest = json.loads(diagnostics_manifest_path.read_text(encoding="utf-8"))
    assert diagnostics_manifest["status"] == "completed"
    assert diagnostics_manifest["resumed_from_status"] == "running"


def test_airflow_csv_ablation_cache_key_tracks_monolith_csv_fingerprint(monkeypatch, tmp_path):
    counters = {"experiment": 0, "relativity": 0, "diagnostics": 0, "bundle": 0, "airflow": 0}
    _install_fake_modules(monkeypatch, counters)
    _install_runner_stubs(monkeypatch, counters)

    import analysis.airflow_ablation_orchestrator as airflow_sidecar

    def fake_run_ablation_matrix(base_csv: Path, output_dir: Path, scalar_bins: int = 8) -> Path:
        counters["airflow"] += 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "manifold_ablation_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "OK",
                    "scalar_bins": scalar_bins,
                    "base_csv_size": Path(base_csv).stat().st_size,
                    "cells": [],
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "manifold_ablation_summary.csv").write_text(
            "stress_mode,zone_rule\nstub,stub\n",
            encoding="utf-8",
        )
        return summary_path

    monkeypatch.setattr(airflow_sidecar, "run_ablation_matrix", fake_run_ablation_matrix)

    config = _build_config(tmp_path)
    runner = AblationRunner(config)
    runner.run_single(config)

    run_dir = Path(config.output_dir) / config.run_name
    monolith_csv = run_dir / "real" / "MONOLITH_DATA.csv"
    monolith_csv.write_text(
        monolith_csv.read_text(encoding="utf-8") + "u1,0.7,0.2\n",
        encoding="utf-8",
    )

    second_manifest = runner.run_single(config)

    assert counters["experiment"] == 1
    assert counters["relativity"] == 1
    assert counters["diagnostics"] == 1
    assert counters["bundle"] == 1
    assert counters["airflow"] == 2
    assert second_manifest["results"]["real"]["dag"]["track5_assembly"]["status"] == "cached"
    assert second_manifest["results"]["real"]["dag"]["airflow_csv_ablation"]["status"] == "completed"


def test_ablation_dag_blocks_cached_child_when_dependency_manifest_failed(tmp_path):
    run_dir = tmp_path / "run"
    dag = AblationDag(run_dir=run_dir, run_id="probe", config_hash="cfg")
    upstream = tmp_path / "upstream.txt"
    child = tmp_path / "child.txt"

    dag.execute(
        NodeSpec(node_id="upstream", cache_key="u1", outputs=[upstream]),
        lambda: upstream.write_text("ok", encoding="utf-8") or {"status": "ok"},
    )
    dag.execute(
        NodeSpec(node_id="child", cache_key="c1", outputs=[child], dependencies=["upstream"]),
        lambda: child.write_text("ok", encoding="utf-8") or {"status": "ok"},
    )

    upstream_manifest_path = dag.node_manifest_path("upstream")
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    upstream_manifest["status"] = "failed"
    upstream_manifest_path.write_text(json.dumps(upstream_manifest, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="blocked dependencies"):
        dag.execute(
            NodeSpec(node_id="child", cache_key="c1", outputs=[child], dependencies=["upstream"]),
            lambda: pytest.fail("child action must not execute when dependency is failed"),
        )

    child_manifest = json.loads(dag.node_manifest_path("child").read_text(encoding="utf-8"))
    assert child_manifest["status"] == "failed"
    assert child_manifest["blocked_dependencies"] == {"upstream": "failed"}


def test_orchestration_contract_emits_topological_order_and_dedupes_dependencies():
    contract = build_orchestration_contract(
        dag_id="probe",
        nodes=["resolve", "track2", "track5", "evidence"],
        dependencies={
            "track2": ["resolve"],
            "track5": ["track2", "track2"],
            "evidence": ["resolve", "track5"],
        },
    )

    assert contract["dependencies"]["track5"] == ["track2"]
    assert contract["topological_order"].index("resolve") < contract["topological_order"].index("track2")
    assert contract["topological_order"].index("track2") < contract["topological_order"].index("track5")
    assert contract["topological_order"].index("track5") < contract["topological_order"].index("evidence")


def test_orchestration_contract_rejects_unknown_dependency_target():
    with pytest.raises(ValueError, match="unknown node_id"):
        build_orchestration_contract(
            dag_id="probe",
            nodes=["resolve"],
            dependencies={"missing": ["resolve"]},
        )


def test_orchestration_contract_rejects_cycles():
    with pytest.raises(ValueError, match="cyclic dependencies"):
        build_orchestration_contract(
            dag_id="probe",
            nodes=["a", "b", "c"],
            dependencies={"a": ["c"], "b": ["a"], "c": ["b"]},
        )
