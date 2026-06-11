import pytest
import torch

from olmo.color_uncertainty import summarize_color_samples


def test_shape_mismatch_raises():
    prior = torch.ones(2, 3)
    conditional = torch.ones(2, 2)

    with pytest.raises(ValueError, match="same shape"):
        summarize_color_samples(prior, conditional)


def test_non_matrix_input_raises():
    prior = torch.ones(3)
    conditional = torch.ones(3)

    with pytest.raises(ValueError, match="shape"):
        summarize_color_samples(prior, conditional)


def test_one_sample_returns_zero_std():
    prior = torch.tensor([[3.0, 5.0]])
    conditional = torch.tensor([[1.0, 8.0]])

    summary = summarize_color_samples(prior, conditional)

    torch.testing.assert_close(summary["mean"], torch.tensor([2.0, -3.0]))
    torch.testing.assert_close(summary["std"], torch.zeros(2))


def test_mean_prob_positive_lcb_and_g_snr():
    prior = torch.tensor(
        [
            [5.0, 3.0, 1.0],
            [7.0, 1.0, 1.0],
            [9.0, 5.0, 1.0],
        ]
    )
    conditional = torch.tensor(
        [
            [1.0, 4.0, 1.0],
            [4.0, 4.0, 1.0],
            [9.0, 4.0, 1.0],
        ]
    )

    summary = summarize_color_samples(prior, conditional, alpha=0.5)
    samples = prior - conditional
    expected_mean = samples.mean(dim=0)
    expected_std = samples.std(dim=0, unbiased=True)

    torch.testing.assert_close(summary["mean"], expected_mean)
    torch.testing.assert_close(summary["std"], expected_std)
    torch.testing.assert_close(summary["prob_positive"], torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0]))
    torch.testing.assert_close(summary["lcb"], expected_mean - 0.5 * expected_std)
    torch.testing.assert_close(summary["g_snr"], expected_mean / (expected_std + 1e-8))
    assert torch.isfinite(summary["g_snr"]).all()
