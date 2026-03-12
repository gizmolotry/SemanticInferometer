from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.complete_pipeline import _build_article_metadata_row, parse_timestamp_to_utc
from core.metric_fusion import calculate_unified_metric


@pytest.fixture
def tmp_path(request):
    root = Path.cwd() / ".pytest_local_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / request.node.name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def test_build_article_metadata_row_preserves_rich_article_fields():
    raw_ts = "2023-03-04"
    _, epoch_s, iso_utc = parse_timestamp_to_utc(raw_ts)
    article = {
        "title": "Analysis: Border Escalation and Its Implications",
        "publication": "Wafa News Agency",
        "timestamp": raw_ts,
        "author": "Desk",
        "perspective_tag": "Secular Palestinian Nationalist",
        "perspective_type": "nationalist",
        "content": "The border escalation is another violation of the 1967 Borders.",
        "event_id": "evt-1",
    }

    row = _build_article_metadata_row(
        article=article,
        index=0,
        bt_uid="uid-0",
        timestamp_field="timestamp",
        timestamp_raw=raw_ts,
        timestamp_epoch_s=epoch_s,
        timestamp_iso_utc=iso_utc,
    )

    assert row["bt_uid"] == "uid-0"
    assert row["source"] == "Wafa News Agency"
    assert row["publication"] == "Wafa News Agency"
    assert row["published_at"] == raw_ts
    assert row["author"] == "Desk"
    assert row["perspective_tag"] == "Secular Palestinian Nationalist"
    assert row["perspective_type"] == "nationalist"
    assert row["timestamp_field"] == "timestamp"
    assert row["timestamp_raw"] == raw_ts
    assert row["timestamp_epoch_s"] == epoch_s
    assert row["timestamp_iso_utc"] == iso_utc
    assert row["snippet"].startswith("The border escalation")
    assert row["content_preview"].startswith("The border escalation")


def test_metric_fusion_preserves_rich_metadata_columns(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], dtype=float)
    gradients = np.array([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]], dtype=float)
    np.save(exp_dir / "features.npy", embeddings)
    np.save(exp_dir / "spectral_u_axis.npy", gradients)
    np.save(exp_dir / "walker_work_integrals.npy", np.array([0.4, 0.8, 0.6], dtype=float))

    with (exp_dir / "walker_states.json").open("w", encoding="utf-8") as f:
        json.dump(["honest", "phantom", "tautology"], f)
    with (exp_dir / "phantom_verdicts.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {"verdict": "HONEST"},
                {"verdict": "PHANTOM"},
                {"verdict": "TAUTOLOGY"},
            ],
            f,
        )
    with (exp_dir / "validation.json").open("w", encoding="utf-8") as f:
        json.dump({"nmi": 0.5, "track_nmi": {"T1.5": 0.4}}, f)

    metadata_path = exp_dir / "article_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "bt_uid",
                "title",
                "source",
                "publication",
                "author",
                "published_at",
                "timestamp_iso_utc",
                "snippet",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "index": 0,
                "bt_uid": "a0",
                "title": "t0",
                "source": "S0",
                "publication": "P0",
                "author": "A0",
                "published_at": "2023-03-04",
                "timestamp_iso_utc": "2023-03-04T00:00:00Z",
                "snippet": "n0",
            }
        )
        writer.writerow(
            {
                "index": 1,
                "bt_uid": "a1",
                "title": "t1",
                "source": "S1",
                "publication": "P1",
                "author": "A1",
                "published_at": "2023-03-05",
                "timestamp_iso_utc": "2023-03-05T00:00:00Z",
                "snippet": "n1",
            }
        )
        writer.writerow(
            {
                "index": 2,
                "bt_uid": "a2",
                "title": "t2",
                "source": "S2",
                "publication": "P2",
                "author": "A2",
                "published_at": "2023-03-06",
                "timestamp_iso_utc": "2023-03-06T00:00:00Z",
                "snippet": "n2",
            }
        )

    output_path = exp_dir / "MONOLITH_DATA.csv"
    df = calculate_unified_metric(
        embeddings_path=exp_dir / "features.npy",
        gradients_path=exp_dir / "spectral_u_axis.npy",
        metadata_path=metadata_path,
        output_path=output_path,
        knn_k=1,
    )

    assert output_path.exists()
    assert {"publication", "author", "published_at", "timestamp_iso_utc", "snippet"}.issubset(df.columns)

    written = pd.read_csv(output_path)
    assert written.loc[0, "publication"] == "P0"
    assert written.loc[1, "author"] == "A1"
    assert written.loc[2, "published_at"] == "2023-03-06"
    assert written.loc[0, "timestamp_iso_utc"] == "2023-03-04T00:00:00Z"
    assert written.loc[1, "snippet"] == "n1"
    assert {"density", "stress", "z_height", "zone", "color_code", "verdict"}.issubset(written.columns)
