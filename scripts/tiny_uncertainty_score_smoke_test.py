from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from olmo.color_uncertainty import COLOR_UNCERTAINTY_METRICS, summarize_color_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny uncertainty-aware CoLoR scoring smoke test.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--select-fraction", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny_uncertainty_score_smoke"))
    return parser.parse_args()


def select_top_indices(scores: torch.Tensor, indices: torch.Tensor, select_fraction: float) -> torch.Tensor:
    k = max(1, int(round(select_fraction * scores.numel())))
    top = torch.argsort(scores)[-k:]
    return indices[top]


def make_fake_losses(num_samples: int, num_examples: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.arange(num_examples, device=device)

    base_score = torch.linspace(-0.4, 1.1, num_examples, device=device)
    uncertainty = torch.linspace(0.05, 0.75, num_examples, device=device)
    score_noise = torch.randn(num_samples, num_examples, generator=generator, device=device) * uncertainty
    score_samples = base_score.unsqueeze(0) + score_noise

    conditional_center = 2.0 + 0.1 * torch.sin(indices.float() / 3.0)
    conditional_noise = 0.02 * torch.randn(num_samples, num_examples, generator=generator, device=device)
    conditional_losses = conditional_center.unsqueeze(0) + conditional_noise
    prior_losses = conditional_losses + score_samples
    return indices, prior_losses, conditional_losses


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if not (0.0 < args.select_fraction <= 1.0):
        raise ValueError("--select-fraction must be in (0, 1]")

    device = torch.device(args.device)
    indices, prior_losses, conditional_losses = make_fake_losses(
        args.num_samples,
        args.num_examples,
        device,
        args.seed,
    )
    summary = summarize_color_samples(prior_losses, conditional_losses, alpha=args.alpha)

    selections = {
        metric: select_top_indices(summary[metric], indices, args.select_fraction).cpu().numpy()
        for metric in COLOR_UNCERTAINTY_METRICS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "uncertainty_scores.npz",
        index=indices.cpu().numpy().astype(np.uint32),
        prior_losses=prior_losses.cpu().numpy().astype(np.float32),
        conditional_losses=conditional_losses.cpu().numpy().astype(np.float32),
        **{metric: values.cpu().numpy().astype(np.float32) for metric, values in summary.items()},
    )
    for metric, selected in selections.items():
        np.save(args.output_dir / f"selected_indices_{metric}.npy", selected.astype(np.uint32))

    mean_selection = set(selections["mean"].tolist())
    report_lines = [
        "metric\tselected\tmean_score\tmean_std\toverlap_with_mean",
    ]
    for metric, selected in selections.items():
        selected_tensor = torch.as_tensor(selected, device=device).long()
        overlap = len(mean_selection.intersection(selected.tolist()))
        report_lines.append(
            "\t".join(
                [
                    metric,
                    ",".join(str(int(i)) for i in selected),
                    f"{summary['mean'][selected_tensor].mean().item():.4f}",
                    f"{summary['std'][selected_tensor].mean().item():.4f}",
                    str(overlap),
                ]
            )
        )

    report = "\n".join(report_lines) + "\n"
    (args.output_dir / "selection_report.tsv").write_text(report)
    print(report, end="")
    print(f"Wrote smoke outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
