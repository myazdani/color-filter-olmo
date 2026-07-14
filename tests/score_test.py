from types import SimpleNamespace
from unittest.mock import patch

from olmo.score import Scorer


def scorer_for_seed(*, data_start_step: int, global_step: int) -> Scorer:
    scorer = object.__new__(Scorer)
    scorer.cfg = SimpleNamespace(
        seed=1,
        data_start_step=data_start_step,
        uncertainty_scoring=SimpleNamespace(enabled=True, num_samples=8, coupled_masks=True),
    )
    scorer.global_step = global_step
    return scorer


def test_stochastic_seed_schedule_continues_across_row_shards():
    sharded = scorer_for_seed(data_start_step=781, global_step=1)
    unsharded = scorer_for_seed(data_start_step=0, global_step=782)

    with patch("olmo.score.torch.manual_seed") as sharded_seed:
        sharded.set_stochastic_sample_seed(sample_idx=3)
    with patch("olmo.score.torch.manual_seed") as unsharded_seed:
        unsharded.set_stochastic_sample_seed(sample_idx=3)

    assert sharded_seed.call_args == unsharded_seed.call_args
    sharded_seed.assert_called_once_with(1 + 782 * 8 + 3)
