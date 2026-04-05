# core/provenance_tracker.py

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# --------------------------------------------------------------------------------------
# Row-level provenance (legacy)
# --------------------------------------------------------------------------------------

@dataclass
class ProvenanceEntry:
    """Row-level provenance for a single extracted item (legacy).

    This matches earlier tooling that tracked per-row provenance for embeddings/UMAP points.
    """
    global_idx: int                 # row index for embeddings & UMAP
    article_id: str
    source: str
    url: str
    timestamp: Optional[str]
    sentence_idx: int
    observer_seed: int
    attention_head: Optional[int]
    provenance_tokens: List[int]    # whatever metadata encoding you used


# --------------------------------------------------------------------------------------
# Pipeline-level provenance (current architecture)
# --------------------------------------------------------------------------------------

@dataclass
class PipelineProvenanceEntry:
    """Pipeline-run provenance for a month/batch.

    This is what complete_pipeline.py emits: a thesis-friendly record of what was run,
    how long it took, and key variance/diagnostic summaries.
    """
    month: str
    seed: int
    mode: str
    embedding_type: str
    n_articles: int
    dim: int
    pipeline_steps: List[str]
    timings: Dict[str, float]
    variance: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceTracker:
    """Collects provenance for both legacy row-level entries and pipeline-run entries."""

    def __init__(self):
        # Legacy list used by earlier visualizers
        self.entries: List[ProvenanceEntry] = []
        # New pipeline-level list
        self.pipeline_entries: List[PipelineProvenanceEntry] = []

    # ---- Legacy API ----
    def add(self, **kwargs):
        """Add a legacy row-level entry (backwards compatible)."""
        self.entries.append(ProvenanceEntry(**kwargs))

    def to_list(self) -> List[Dict[str, Any]]:
        """Legacy export: row-level entries only."""
        return [asdict(e) for e in self.entries]

    # ---- New API ----
    def add_entry(self, entry: Union[ProvenanceEntry, PipelineProvenanceEntry]):
        """Add either a legacy or pipeline-level entry."""
        if isinstance(entry, PipelineProvenanceEntry):
            self.pipeline_entries.append(entry)
        elif isinstance(entry, ProvenanceEntry):
            self.entries.append(entry)
        else:
            raise TypeError(f"Unsupported provenance entry type: {type(entry)}")

    def to_dict(self) -> Dict[str, Any]:
        """Full export: both legacy + pipeline entries."""
        return {
            "entries": [asdict(e) for e in self.entries],
            "pipeline_entries": [e.to_dict() for e in self.pipeline_entries],
        }

    def save(self, provenance_dir: Union[str, Path], filename: Optional[str] = None) -> Path:
        """
        Persist current provenance payload to a JSON file.

        Args:
            provenance_dir: Directory where the provenance file should be written.
            filename: Optional filename. Defaults to "provenance.json".

        Returns:
            Path to the saved JSON file.
        """
        out_dir = Path(provenance_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (filename or "provenance.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path

    def __len__(self):
        return len(self.entries) + len(self.pipeline_entries)
