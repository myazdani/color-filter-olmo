from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_score_pool_extra_training_sets import resolve_summary_dropout_rate  # noqa: E402


def test_resolves_target_specific_rate_when_generic_rate_is_null() -> None:
    summary = pd.DataFrame(
        {
            "dropout_rate": [np.nan, np.nan],
            "embedding_dropout": [0.005, 0.005],
        }
    )
    assert resolve_summary_dropout_rate(summary, target="embedding") == pytest.approx(0.005)


def test_rejects_disagreement_between_generic_and_target_specific_rates() -> None:
    summary = pd.DataFrame(
        {
            "dropout_rate": [0.01, 0.01],
            "embedding_dropout": [0.005, 0.005],
        }
    )
    with pytest.raises(ValueError, match="disagrees"):
        resolve_summary_dropout_rate(summary, target="embedding")
