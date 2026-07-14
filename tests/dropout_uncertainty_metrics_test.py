import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "21_dropout_uncertainty_metrics.py"
SPEC = importlib.util.spec_from_file_location("dropout_uncertainty_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def test_optional_tables_follow_score_index(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    full_scores_path = tmp_path / "full_scores.csv"
    pd.DataFrame(
        {
            "seq_idx": [100, 101, 102],
            "pool_name": ["zero", "one", "two"],
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame({"full_color_score": [0.1, 0.2, 0.3]}).to_csv(full_scores_path, index=False)
    summary = pd.DataFrame({"seq_idx": [0, 1, 2], "mean_color": [3.0, 1.0, 2.0]})

    aligned = METRICS.attach_optional_tables(
        summary,
        metadata_path,
        full_scores_path,
        np.asarray([2, 0, 1], dtype=np.int64),
    )

    assert aligned["seq_idx"].tolist() == [102, 100, 101]
    assert aligned["pool_name"].tolist() == ["two", "zero", "one"]
    assert aligned["full_color_score"].tolist() == pytest.approx([0.3, 0.1, 0.2])


def test_score_index_is_the_fallback_selection_id():
    summary = pd.DataFrame({"seq_idx": [0, 1], "mean_color": [1.0, 2.0]})

    aligned = METRICS.attach_optional_tables(
        summary,
        metadata_path=None,
        full_scores_path=None,
        score_index=np.asarray([7, 3], dtype=np.int64),
    )

    assert aligned["seq_idx"].tolist() == [7, 3]


def test_duplicate_score_index_is_rejected(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"seq_idx": [10, 11]}).to_csv(metadata_path, index=False)
    summary = pd.DataFrame({"seq_idx": [0, 1]})

    with pytest.raises(ValueError, match="duplicate rows"):
        METRICS.attach_optional_tables(
            summary,
            metadata_path=metadata_path,
            full_scores_path=None,
            score_index=np.asarray([0, 0], dtype=np.int64),
        )
