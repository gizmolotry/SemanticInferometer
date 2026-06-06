from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stable_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(dict(payload)), sort_keys=True, separators=(",", ":"), default=str)


def build_cache_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_payload(payload).encode("utf-8")).hexdigest()[:16]


def build_orchestration_contract(
    *,
    dag_id: str,
    nodes: Sequence[Mapping[str, Any] | str],
    dependencies: Optional[Mapping[str, Sequence[str]]] = None,
    executor: str = "manifest",
    airflow_compatible: bool = True,
    manifest_paths: Optional[Sequence[Path | str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a JSON-safe DAG contract without changing execution behavior.

    This is intentionally metadata-only. It lets script-style orchestrators
    advertise the same node/dependency shape as the lower-level AblationDag
    without taking a runtime dependency on Airflow or replacing subprocess
    execution.
    """

    normalized_nodes: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_node in nodes:
        if isinstance(raw_node, str):
            node = {"node_id": raw_node}
        else:
            node = dict(raw_node)
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("orchestration contract nodes require non-empty node_id")
        if node_id in seen:
            raise ValueError(f"duplicate orchestration node_id: {node_id}")
        seen.add(node_id)
        node.setdefault("status", "pending")
        normalized_nodes.append(_jsonable(node))

    normalized_dependencies: Dict[str, list[str]] = {}
    for node in normalized_nodes:
        node_id = str(node["node_id"])
        normalized_dependencies[node_id] = []
    for node_id, deps in (dependencies or {}).items():
        node_id = str(node_id)
        if node_id not in seen:
            raise ValueError(f"orchestration contract has dependency entry for unknown node_id: {node_id}")
        normalized_dependencies.setdefault(node_id, [])
        normalized_dependencies[node_id] = list(dict.fromkeys(str(dep) for dep in deps))

    unknown_deps = sorted(
        {
            dep
            for deps in normalized_dependencies.values()
            for dep in deps
            if dep not in seen
        }
    )
    if unknown_deps:
        raise ValueError(f"orchestration contract has unknown dependencies: {unknown_deps}")
    topological_order = _topological_order(normalized_dependencies)

    return {
        "schema_version": "1.0",
        "dag_id": str(dag_id),
        "executor": str(executor),
        "airflow_compatible": bool(airflow_compatible),
        "node_count": len(normalized_nodes),
        "nodes": normalized_nodes,
        "dependencies": normalized_dependencies,
        "topological_order": topological_order,
        "manifest_paths": [str(Path(path)) for path in (manifest_paths or [])],
        "metadata": _jsonable(dict(metadata or {})),
    }


def _topological_order(dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    """Return a deterministic dependency-first order or fail on cycles."""

    permanent: set[str] = set()
    temporary: set[str] = set()
    order: list[str] = []

    def visit(node_id: str, stack: list[str]) -> None:
        if node_id in permanent:
            return
        if node_id in temporary:
            cycle_start = stack.index(node_id) if node_id in stack else 0
            cycle = [*stack[cycle_start:], node_id]
            raise ValueError(f"orchestration contract has cyclic dependencies: {' -> '.join(cycle)}")
        temporary.add(node_id)
        for dep in dependencies.get(node_id, []):
            visit(str(dep), [*stack, node_id])
        temporary.remove(node_id)
        permanent.add(node_id)
        order.append(node_id)

    for node_id in sorted(dependencies):
        visit(node_id, [])
    return order


def fingerprint_path(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    cache_key: str
    outputs: Sequence[Path] = field(default_factory=tuple)
    dependencies: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def artifact_fingerprint(paths: Iterable[Path]) -> Dict[str, Any]:
    materialized = [Path(p) for p in paths if Path(p).exists()]
    return {
        "artifact_count": len(materialized),
        "artifacts": [fingerprint_path(path) for path in sorted(materialized)],
    }


@dataclass
class AblationTaskResult:
    name: str
    status: str
    cache_key: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    artifact_fingerprint: Dict[str, Any] = field(default_factory=dict)
    manifest_path: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def output_fingerprint(self) -> str:
        payload = {
            "cache_key": self.cache_key,
            "artifacts": self.artifact_fingerprint,
            "outputs": _jsonable(self.outputs),
        }
        return _hash_payload(payload)


class AblationTaskGraph:
    def __init__(
        self,
        graph_dir: Path,
        *,
        config_hash: str,
        graph_name: str,
        branch_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.graph_name = graph_name
        self.branch_metadata = branch_metadata or {}
        self.results: Dict[str, AblationTaskResult] = {}

    def run_node(
        self,
        name: str,
        *,
        action: Callable[[], Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Iterable[str]] = None,
        expected_artifacts: Optional[Callable[[Dict[str, Any]], Iterable[Path]]] = None,
    ) -> AblationTaskResult:
        params = params or {}
        dependency_names = list(dependencies or [])
        dep_results = {dep: self.results[dep] for dep in dependency_names}

        cache_key = _hash_payload(
            {
                "graph": self.graph_name,
                "node": name,
                "config_hash": self.config_hash,
                "branch_metadata": self.branch_metadata,
                "params": params,
                "dependencies": {dep: dep_results[dep].output_fingerprint for dep in dependency_names},
            }
        )
        manifest_path = self.graph_dir / f"{name}.json"

        cached = self._load_cached_result(
            name=name,
            cache_key=cache_key,
            manifest_path=manifest_path,
            expected_artifacts=expected_artifacts,
        )
        if cached is not None:
            self.results[name] = cached
            return cached

        started_at = _utc_now()
        self._write_manifest(
            manifest_path,
            {
                "name": name,
                "status": "running",
                "cache_key": cache_key,
                "config_hash": self.config_hash,
                "graph_name": self.graph_name,
                "branch_metadata": _jsonable(self.branch_metadata),
                "params": _jsonable(params),
                "dependencies": {dep: dep_results[dep].output_fingerprint for dep in dependency_names},
                "started_at": started_at,
            },
        )

        try:
            raw_result = action() or {}
            outputs = dict(raw_result.get("outputs", {}))
            artifact_paths = [str(Path(p)) for p in raw_result.get("artifacts", [])]
            if not artifact_paths and expected_artifacts is not None:
                artifact_paths = [str(Path(p)) for p in expected_artifacts(outputs)]
            artifact_meta = artifact_fingerprint(Path(p) for p in artifact_paths)
            result = AblationTaskResult(
                name=name,
                status="completed",
                cache_key=cache_key,
                outputs=outputs,
                artifacts=artifact_paths,
                dependencies={dep: dep_results[dep].output_fingerprint for dep in dependency_names},
                artifact_fingerprint=artifact_meta,
                manifest_path=str(manifest_path),
                started_at=started_at,
                finished_at=_utc_now(),
            )
            self._write_manifest(
                manifest_path,
                {
                    "name": name,
                    "status": result.status,
                    "cache_key": result.cache_key,
                    "config_hash": self.config_hash,
                    "graph_name": self.graph_name,
                    "branch_metadata": _jsonable(self.branch_metadata),
                    "params": _jsonable(params),
                    "dependencies": result.dependencies,
                    "artifacts": result.artifacts,
                    "artifact_fingerprint": result.artifact_fingerprint,
                    "outputs": _jsonable(result.outputs),
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "output_fingerprint": result.output_fingerprint,
                },
            )
            self.results[name] = result
            return result
        except Exception as exc:
            self._write_manifest(
                manifest_path,
                {
                    "name": name,
                    "status": "failed",
                    "cache_key": cache_key,
                    "config_hash": self.config_hash,
                    "graph_name": self.graph_name,
                    "branch_metadata": _jsonable(self.branch_metadata),
                    "params": _jsonable(params),
                    "dependencies": {dep: dep_results[dep].output_fingerprint for dep in dependency_names},
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "error": repr(exc),
                },
            )
            raise

    def _load_cached_result(
        self,
        *,
        name: str,
        cache_key: str,
        manifest_path: Path,
        expected_artifacts: Optional[Callable[[Dict[str, Any]], Iterable[Path]]],
    ) -> Optional[AblationTaskResult]:
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if payload.get("status") != "completed":
            return None
        if payload.get("cache_key") != cache_key:
            return None
        outputs = dict(payload.get("outputs", {}))
        artifact_paths = [Path(p) for p in payload.get("artifacts", [])]
        if not artifact_paths and expected_artifacts is not None:
            artifact_paths = [Path(p) for p in expected_artifacts(outputs)]
        if any(not path.exists() for path in artifact_paths):
            return None
        return AblationTaskResult(
            name=name,
            status="cached",
            cache_key=cache_key,
            outputs=outputs,
            artifacts=[str(path) for path in artifact_paths],
            dependencies=dict(payload.get("dependencies", {})),
            artifact_fingerprint=artifact_fingerprint(artifact_paths),
            manifest_path=str(manifest_path),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
        )

    def _write_manifest(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


class AblationDag:
    """Compatibility wrapper for node-manifest based execution in master_ablation."""

    def __init__(self, run_dir: Path, run_id: str, config_hash: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.config_hash = config_hash
        self.dag_dir = self.run_dir / "_ablation_dag"
        self.nodes_dir = self.dag_dir / "nodes"
        self.status_path = self.dag_dir / "status.json"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self._run_status = "running"
        self._error: Optional[str] = None
        self._write_status()

    def execute(self, spec: NodeSpec, action: Callable[[], Any]) -> Dict[str, Any]:
        manifest_path = self.node_manifest_path(spec.node_id)
        existing = self._read_json(manifest_path)
        outputs = [str(Path(path)) for path in spec.outputs]
        outputs_present = self._outputs_exist(spec.outputs)
        attempt = int(existing.get("attempts", 0)) if isinstance(existing, dict) else 0
        blocked_dependencies = self._blocked_dependencies(spec.dependencies)
        if blocked_dependencies:
            error = (
                f"Cannot execute {spec.node_id}; blocked dependencies: "
                f"{blocked_dependencies}"
            )
            self._write_json(
                manifest_path,
                {
                    "node_id": spec.node_id,
                    "cache_key": spec.cache_key,
                    "status": "failed",
                    "run_id": self.run_id,
                    "config_hash": self.config_hash,
                    "dependencies": list(spec.dependencies),
                    "blocked_dependencies": blocked_dependencies,
                    "metadata": dict(spec.metadata),
                    "outputs": outputs,
                    "outputs_present": outputs_present,
                    "attempts": attempt + 1,
                    "resumed_from_status": existing.get("status") if isinstance(existing, dict) else None,
                    "error": error,
                    "started_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                },
            )
            self._write_status()
            raise RuntimeError(error)

        if (
            isinstance(existing, dict)
            and existing.get("status") == "completed"
            and existing.get("cache_key") == spec.cache_key
            and outputs_present
        ):
            self._write_status()
            return {
                "status": "cached",
                "cache_key": spec.cache_key,
                "manifest_path": str(manifest_path),
                "result": existing.get("result"),
                "outputs": outputs,
            }

        running_manifest = {
            "node_id": spec.node_id,
            "cache_key": spec.cache_key,
            "status": "running",
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "dependencies": list(spec.dependencies),
            "metadata": dict(spec.metadata),
            "outputs": outputs,
            "attempts": attempt + 1,
            "resumed_from_status": existing.get("status") if isinstance(existing, dict) else None,
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_json(manifest_path, running_manifest)
        self._write_status()

        try:
            result = action()
        except Exception as exc:
            failed_manifest = dict(running_manifest)
            failed_manifest.update(
                {
                    "status": "failed",
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "outputs_present": self._outputs_exist(spec.outputs),
                }
            )
            self._write_json(manifest_path, failed_manifest)
            self._write_status()
            raise

        completed_manifest = dict(running_manifest)
        completed_manifest.update(
            {
                "status": "completed",
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
                "outputs_present": self._outputs_exist(spec.outputs),
                "result": result,
            }
        )
        self._write_json(manifest_path, completed_manifest)
        self._write_status()
        return {
            "status": "completed",
            "cache_key": spec.cache_key,
            "manifest_path": str(manifest_path),
            "result": result,
            "outputs": outputs,
        }

    def mark_failed(self, error: str) -> None:
        self._run_status = "failed"
        self._error = error
        self._write_status()

    def mark_completed(self) -> None:
        self._run_status = "completed"
        self._error = None
        self._write_status()

    def node_manifest_path(self, node_id: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in node_id)
        return self.nodes_dir / f"{safe_name}.json"

    def _blocked_dependencies(self, dependencies: Iterable[str]) -> Dict[str, str]:
        blocked: Dict[str, str] = {}
        for dependency in dependencies:
            dependency_manifest = self._read_json(self.node_manifest_path(dependency))
            status = str(dependency_manifest.get("status") or "missing")
            if status != "completed":
                blocked[str(dependency)] = status
        return blocked

    def snapshot(self) -> Dict[str, Any]:
        nodes: Dict[str, Any] = {}
        for path in sorted(self.nodes_dir.glob("*.json")):
            payload = self._read_json(path)
            if not payload:
                continue
            node_id = str(payload.get("node_id") or path.stem)
            nodes[node_id] = {
                "status": payload.get("status"),
                "cache_key": payload.get("cache_key"),
                "manifest_path": str(path),
                "outputs_present": payload.get("outputs_present"),
                "resumed_from_status": payload.get("resumed_from_status"),
            }

        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "status": self._run_status,
            "error": self._error,
            "status_path": str(self.status_path),
            "nodes": nodes,
        }

    def _write_status(self) -> None:
        self._write_json(
            self.status_path,
            {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "status": self._run_status,
                "error": self._error,
                "updated_at": _utc_now(),
                "nodes": self.snapshot().get("nodes", {}),
            },
        )

    @staticmethod
    def _outputs_exist(outputs: Iterable[Path]) -> bool:
        normalized = [Path(path) for path in outputs]
        return bool(normalized) and all(path.exists() for path in normalized)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, default=str), encoding="utf-8")
