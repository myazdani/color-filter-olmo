from __future__ import annotations

from typing import Dict

import torch


COLOR_UNCERTAINTY_METRICS = ("mean", "std", "lcb", "prob_positive", "g_snr")


def summarize_color_samples(
    prior_losses: torch.Tensor,
    conditional_losses: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Summarize stochastic CoLoR samples.

    Args:
        prior_losses: Tensor of shape [num_samples, num_examples].
        conditional_losses: Tensor of shape [num_samples, num_examples].
        alpha: Lower-confidence-bound multiplier.
        eps: Small constant for numerical stability.

    Returns:
        Dictionary with tensors of shape [num_examples].
    """
    if prior_losses.shape != conditional_losses.shape:
        raise ValueError("prior_losses and conditional_losses must have the same shape")
    if prior_losses.ndim != 2:
        raise ValueError("loss tensors must have shape [num_samples, num_examples]")
    if prior_losses.shape[0] < 1:
        raise ValueError("loss tensors must include at least one stochastic sample")

    samples = prior_losses - conditional_losses
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=True) if samples.shape[0] > 1 else torch.zeros_like(mean)

    return {
        "mean": mean,
        "std": std,
        "lcb": mean - alpha * std,
        "prob_positive": (samples > 0).float().mean(dim=0),
        "g_snr": mean / (std + eps),
    }
