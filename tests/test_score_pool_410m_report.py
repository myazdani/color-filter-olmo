from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_pool_410m_report import (  # noqa: E402
    DEFAULT_C4_PROXY_NOTE,
    HARD_SOURCE_SEED_T_CRITICAL_95,
    aggregate_hard_source_seed_eval,
    c4_proxy_note,
)


def test_c4_proxy_note_falls_back_for_legacy_manifest() -> None:
    manifest = {
        "c4_val_proxy": {
            "label": "c4_val_proxy",
            "source": "allenai/c4 en validation streaming split",
            "path": "/tmp/c4.npy",
        }
    }

    assert c4_proxy_note(manifest) == DEFAULT_C4_PROXY_NOTE


def test_c4_proxy_note_preserves_manifest_note() -> None:
    manifest = {"c4_val_proxy": {"note": "Exact source note."}}

    assert c4_proxy_note(manifest) == "Exact source note."


def test_hard_source_seed_eval_includes_base_seed_in_student_t_interval() -> None:
    base_run_id = "hard_pair_mid2_only_100k"
    rows = []
    for run_id, value in [
        (base_run_id, 1.0),
        ("hard_pair_mid2_only_seed18_100k", 2.0),
        ("hard_pair_mid2_only_seed19_100k", 3.0),
    ]:
        rows.append(
            {
                "run_id": run_id,
                "label": "books_val",
                "metric": "CrossEntropyLoss",
                "step": 780,
                "value": value,
            }
        )

    summary = aggregate_hard_source_seed_eval(
        pd.DataFrame(rows),
        [row["run_id"] for row in rows],
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    expected_half_width = HARD_SOURCE_SEED_T_CRITICAL_95 / (3**0.5)
    assert row["base_run_id"] == base_run_id
    assert row["seed_count"] == 3
    assert row["mean"] == pytest.approx(2.0)
    assert row["sample_std"] == pytest.approx(1.0)
    assert row["ci95_half_width"] == pytest.approx(expected_half_width)
    assert row["ci95_lower"] == pytest.approx(2.0 - expected_half_width)
    assert row["ci95_upper"] == pytest.approx(2.0 + expected_half_width)
