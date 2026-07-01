from __future__ import annotations

import argparse
import csv
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


PAIRWISE_TASKS = (
    ("hp_vs_hn", "hard_positive", "hard_negative"),
    ("hp_vs_rn", "hard_positive", "random_negative"),
    ("hp_vs_tn", "hard_positive", "tail_negative"),
    ("rp_vs_hn", "random_positive", "hard_negative"),
    ("rp_vs_rn", "random_positive", "random_negative"),
    ("rp_vs_tn", "random_positive", "tail_negative"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate offline CoLoR selection strategies from saved dropout MC samples."
    )
    parser.add_argument("--mc-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--rates", default="0.015625,0.03125,0.0625,0.125")
    parser.add_argument("--betas", default="0.5,1.0,2.0")
    parser.add_argument("--quantiles", default="0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99")
    parser.add_argument("--tau64-cutoff", type=float, default=None)
    parser.add_argument("--max-strategies-for-overlap", type=int, default=80)
    parser.add_argument("--save-selected-indices", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"Expected at least one float in {text!r}")
    return values


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def roc_auc(labels: np.ndarray, decision_scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(decision_scores)
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, decision_scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-decision_scores, kind="mergesort")
    ranked = labels[order]
    tp = np.cumsum(ranked)
    precision = tp / (np.arange(len(ranked)) + 1)
    return float((precision * ranked).sum() / n_pos)


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    tp = int((labels & predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_lowest(scores: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive")
    k = min(k, len(scores))
    selected = np.argpartition(scores, k - 1)[:k]
    return selected[np.argsort(scores[selected], kind="mergesort")]


def selected_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    mask[select_lowest(scores, k)] = True
    return mask


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a = set(int(x) for x in a)
    set_b = set(int(x) for x in b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_samples(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if "color_samples" not in arrays:
        if "conditional_losses" not in arrays or "prior_losses" not in arrays:
            raise ValueError("MC sample npz must contain color_samples or prior/conditional losses")
        arrays["color_samples"] = arrays["conditional_losses"] - arrays["prior_losses"]
    return arrays


def load_summary(path: Path | None):
    if path is None:
        return None
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def get_pool_names(samples: dict[str, np.ndarray], summary) -> np.ndarray:
    if summary is not None and "pool_name" in summary.columns:
        return summary["pool_name"].astype(str).str.lower().to_numpy()
    if "pool_name" in samples:
        return np.char.lower(samples["pool_name"].astype(str))
    return np.asarray([""] * len(samples["color_samples"]), dtype="U1")


def get_seq_idx(samples: dict[str, np.ndarray], summary) -> np.ndarray:
    if summary is not None and "seq_idx" in summary.columns:
        return summary["seq_idx"].to_numpy(dtype=np.int64)
    if "seq_idx" in samples:
        return samples["seq_idx"].astype(np.int64)
    return np.arange(len(samples["color_samples"]), dtype=np.int64)


def get_full_color(samples: dict[str, np.ndarray], summary) -> np.ndarray | None:
    if summary is not None and "full_color_score" in summary.columns:
        return summary["full_color_score"].to_numpy(dtype=np.float64)
    if "full_color_score" in samples:
        return samples["full_color_score"].astype(np.float64)
    return None


def build_strategies(color_samples: np.ndarray, betas: list[float], quantiles: list[float]) -> dict[str, np.ndarray]:
    mean = color_samples.mean(axis=1)
    std = color_samples.std(axis=1, ddof=1) if color_samples.shape[1] > 1 else np.zeros(len(mean))
    strategies: dict[str, np.ndarray] = {"mean": mean.astype(np.float64)}
    for beta in betas:
        beta_label = f"{beta:g}"
        strategies[f"conservative_mean_plus_{beta_label}std"] = (mean + beta * std).astype(np.float64)
        strategies[f"optimistic_mean_minus_{beta_label}std"] = (mean - beta * std).astype(np.float64)
    for q in quantiles:
        strategies[f"quantile_q{int(round(q * 100)):02d}"] = np.quantile(color_samples, q, axis=1).astype(np.float64)
    return strategies


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_metrics(
    pool_name: np.ndarray,
    strategies: dict[str, np.ndarray],
    tau64_cutoff: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id, positive_pool, negative_pool in PAIRWISE_TASKS:
        task_mask = (pool_name == positive_pool) | (pool_name == negative_pool)
        if not task_mask.any():
            continue
        labels = pool_name[task_mask] == positive_pool
        n_pos = int(labels.sum())
        n_neg = int((~labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        for strategy_name, scores in strategies.items():
            task_scores = scores[task_mask]
            decision_scores = -task_scores
            balanced_pred = selected_mask(task_scores, n_pos)
            balanced = binary_metrics(labels, balanced_pred)
            row: dict[str, object] = {
                "metric_scope": "pairwise",
                "task_id": task_id,
                "strategy": strategy_name,
                "n_examples": int(len(labels)),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "roc_auc": roc_auc(labels, decision_scores),
                "average_precision": average_precision(labels, decision_scores),
                "balanced_precision": balanced["precision"],
                "balanced_recall": balanced["recall"],
                "balanced_f1": balanced["f1"],
            }
            if tau64_cutoff is not None:
                cutoff = binary_metrics(labels, task_scores <= tau64_cutoff)
                row.update(
                    {
                        "tau64_precision": cutoff["precision"],
                        "tau64_recall": cutoff["recall"],
                        "tau64_f1": cutoff["f1"],
                    }
                )
            else:
                row.update({"tau64_precision": float("nan"), "tau64_recall": float("nan"), "tau64_f1": float("nan")})
            rows.append(row)
    return rows


def full_pool_metrics(
    full_color: np.ndarray | None,
    strategies: dict[str, np.ndarray],
    rates: list[float],
) -> list[dict[str, object]]:
    if full_color is None:
        return []
    rows: list[dict[str, object]] = []
    finite = np.isfinite(full_color)
    if not finite.all():
        raise ValueError("full_color_score contains non-finite values")
    for rate in rates:
        k = max(1, int(round(rate * len(full_color))))
        reference = set(select_lowest(full_color, k).tolist())
        for strategy_name, scores in strategies.items():
            selected = set(select_lowest(scores, k).tolist())
            intersection = len(reference & selected)
            rows.append(
                {
                    "metric_scope": "full_pool",
                    "task_id": f"rate_{rate:g}",
                    "strategy": strategy_name,
                    "n_examples": int(len(full_color)),
                    "n_selected": k,
                    "selection_rate": rate,
                    "recall_vs_full": intersection / len(reference),
                    "precision_vs_full": intersection / len(selected) if selected else 0.0,
                    "jaccard_vs_full": jaccard(reference, selected),
                }
            )
    return rows


def selection_overlap(
    strategies: dict[str, np.ndarray],
    rates: list[float],
    seq_idx: np.ndarray,
    max_strategies: int,
) -> list[dict[str, object]]:
    strategy_items = list(strategies.items())[:max_strategies]
    rows: list[dict[str, object]] = []
    for rate in rates:
        k = max(1, int(round(rate * len(seq_idx))))
        selections = {
            name: set(seq_idx[select_lowest(scores, k)].astype(np.int64).tolist()) for name, scores in strategy_items
        }
        for left, right in combinations(selections, 2):
            rows.append(
                {
                    "selection_rate": rate,
                    "k": k,
                    "left_strategy": left,
                    "right_strategy": right,
                    "jaccard": jaccard(selections[left], selections[right]),
                    "intersection": len(selections[left] & selections[right]),
                }
            )
    return rows


def save_selected_indices(
    output_dir: Path,
    strategies: dict[str, np.ndarray],
    rates: list[float],
    seq_idx: np.ndarray,
) -> None:
    selected_dir = output_dir / "strategy_selected_indices"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        k = max(1, int(round(rate * len(seq_idx))))
        rate_label = f"{rate:g}".replace(".", "p")
        for strategy_name, scores in strategies.items():
            selected = seq_idx[select_lowest(scores, k)].astype(np.int64)
            np.save(selected_dir / f"{sanitize_name(strategy_name)}__rate_{rate_label}.npy", selected)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rates = parse_float_list(args.rates)
    betas = parse_float_list(args.betas)
    quantiles = parse_float_list(args.quantiles)
    for rate in rates:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"Selection rate must be in (0, 1]: {rate}")
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1]: {quantile}")

    samples = load_samples(args.mc_samples)
    summary = load_summary(args.summary)
    color_samples = samples["color_samples"].astype(np.float64)
    if color_samples.ndim != 2:
        raise ValueError(f"color_samples must be rank-2, got shape {color_samples.shape}")
    if not np.isfinite(color_samples).all():
        raise ValueError("color_samples contains non-finite values")

    seq_idx = get_seq_idx(samples, summary)
    pool_name = get_pool_names(samples, summary)
    full_color = get_full_color(samples, summary)
    if len(seq_idx) != len(color_samples) or len(pool_name) != len(color_samples):
        raise ValueError("Metadata length does not match color sample rows")

    strategies = build_strategies(color_samples, betas=betas, quantiles=quantiles)
    pair_rows = pairwise_metrics(pool_name, strategies, tau64_cutoff=args.tau64_cutoff)
    full_rows = full_pool_metrics(full_color, strategies, rates=rates)
    overlap_rows = selection_overlap(strategies, rates, seq_idx, max_strategies=args.max_strategies_for_overlap)

    metrics_path = args.output_dir / "strategy_sweep_metrics.csv"
    overlap_path = args.output_dir / "strategy_selection_overlap.csv"
    write_csv(metrics_path, pair_rows + full_rows)
    write_csv(overlap_path, overlap_rows)
    if args.save_selected_indices:
        save_selected_indices(args.output_dir, strategies, rates, seq_idx)

    print(f"loaded rows: {len(color_samples):,}")
    print(f"num samples: {color_samples.shape[1]}")
    print(f"strategies: {len(strategies)}")
    print(f"pairwise metric rows: {len(pair_rows)}")
    print(f"full-pool metric rows: {len(full_rows)}")
    print("wrote metrics:", metrics_path)
    print("wrote overlap:", overlap_path)
    if args.save_selected_indices:
        print("wrote selections:", args.output_dir / "strategy_selected_indices")


if __name__ == "__main__":
    main()
