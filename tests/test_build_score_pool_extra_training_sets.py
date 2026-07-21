from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_score_pool_extra_training_sets import (  # noqa: E402
    resolve_summary_dropout_rate,
    validate_pair_score_frame,
)


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


def pair_mid4_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(
        {
            "seq_idx": [0, 1],
            "pool_name": ["hard_positive", "hard_negative"],
        }
    )
    scores = metadata.assign(
        ablated_color_score=[-0.2, 0.1],
        variant_id="pair_mid4",
        variant_family="paired",
        cond_removed_layers="[4, 5, 6, 7]",
        marg_removed_layers="[4, 5, 6, 7]",
        cond_kept_layers=8,
        marg_kept_layers=8,
    )
    return scores, metadata


def test_validates_pair_mid4_provenance_for_both_scoring_models() -> None:
    scores, metadata = pair_mid4_scores()

    validated = validate_pair_score_frame(
        scores,
        run_id="hard_pair_mid4_only_100k",
        expected_variant_id="pair_mid4",
        expected_removed_layers=(4, 5, 6, 7),
        metadata=metadata,
    )

    assert validated is scores


def test_rejects_pair_mid4_scores_with_wrong_marginal_layers() -> None:
    scores, metadata = pair_mid4_scores()
    scores["marg_removed_layers"] = "[5, 6]"

    with pytest.raises(ValueError, match="marg_removed_layers"):
        validate_pair_score_frame(
            scores,
            run_id="hard_pair_mid4_only_100k",
            expected_variant_id="pair_mid4",
            expected_removed_layers=(4, 5, 6, 7),
            metadata=metadata,
        )
