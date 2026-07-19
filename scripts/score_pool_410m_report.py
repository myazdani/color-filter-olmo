#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RUN_IDS = (
    "random_positive_oracle_100k",
    "random_pair_cascade_100k",
    "random_union_control_100k",
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
    "hard_union_control_100k",
    "hard_pair_mid2_only_100k",
    "hard_dropout_embed_p000001_conservative_100k",
    "hard_dropout_embed_p0005_conservative_100k",
    "hard_dropout_embed_p001_conservative_100k",
)

RUN_LABELS = {
    "random_positive_oracle_100k": "Random positive-only baseline",
    "random_pair_cascade_100k": "Random PN cascade",
    "random_union_control_100k": "Random PN mixture control",
    "hard_positive_oracle_100k": "Hard positive-only baseline",
    "hard_pair_cascade_100k": "Hard PN cascade",
    "hard_union_control_100k": "Hard PN mixture control",
    "hard_pair_mid2_only_100k": "Hard PN pair-mid2-only",
    "hard_dropout_embed_p000001_conservative_100k": "Hard PN embedding-dropout LCB p=1e-5",
    "hard_dropout_embed_p0005_conservative_100k": "Hard PN embedding-dropout LCB p=0.005",
    "hard_dropout_embed_p001_conservative_100k": "Hard PN embedding-dropout LCB p=0.01",
}

RUN_GROUPS = {
    "random_source": {
        "title": "Random-source comparison",
        "run_ids": (
            "random_positive_oracle_100k",
            "random_pair_cascade_100k",
            "random_union_control_100k",
        ),
    },
    "hard_source": {
        "title": "Hard-source comparison",
        "run_ids": (
            "hard_positive_oracle_100k",
            "hard_pair_cascade_100k",
            "hard_union_control_100k",
            "hard_pair_mid2_only_100k",
            "hard_dropout_embed_p000001_conservative_100k",
            "hard_dropout_embed_p0005_conservative_100k",
            "hard_dropout_embed_p001_conservative_100k",
        ),
    },
}

HARD_SEED_PAIR_BASE_RUN_IDS = (
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
)

FIGURES = (
    "train_loss_by_run.png",
    "train_loss_random_source.png",
    "train_loss_hard_source.png",
    "eval_loss_books_by_run.png",
    "eval_loss_books_random_source.png",
    "eval_loss_books_hard_source.png",
    "eval_loss_books_hard_oracle_vs_cascade_seeds.png",
    "eval_loss_c4_by_run.png",
    "eval_loss_c4_random_source.png",
    "eval_loss_c4_hard_source.png",
    "tokens_per_second_by_run.png",
    "tokens_per_second_random_source.png",
    "tokens_per_second_hard_source.png",
    "selection_full_score_distributions.png",
    "selection_pair_mid2_score_distributions.png",
    "selected_set_overlap_heatmap.png",
    "selected_set_overlap_random_source.png",
    "selected_set_overlap_hard_source.png",
)


def run_seed(run_id: str) -> int | None:
    match = re.match(r"^(.+)_seed(\d+)_100k$", run_id)
    return int(match.group(2)) if match else None


def base_run_id(run_id: str) -> str:
    match = re.match(r"^(.+)_seed\d+_100k$", run_id)
    return f"{match.group(1)}_100k" if match else run_id


def run_label(run_id: str) -> str:
    seed = run_seed(run_id)
    label = RUN_LABELS.get(base_run_id(run_id), run_id)
    return f"{label} seed {seed}" if seed is not None else label


def dynamic_run_groups(run_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    run_ids = list(run_ids)
    groups: dict[str, dict[str, object]] = {}
    for group_name, group in RUN_GROUPS.items():
        base_ids = set(group["run_ids"])
        groups[group_name] = {
            "title": group["title"],
            "run_ids": tuple(run_id for run_id in run_ids if base_run_id(run_id) in base_ids),
        }
    return groups


def hard_seed_pair_run_ids(run_ids: Iterable[str]) -> tuple[str, ...]:
    base_ids = set(HARD_SEED_PAIR_BASE_RUN_IDS)
    return tuple(run_id for run_id in run_ids if base_run_id(run_id) in base_ids)


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def parse_timestamp(line: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")


def parse_logs(
    results_dir: Path,
    run_ids: Iterable[str],
    eval_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step_re = re.compile(r"\[step=(\d+)/(\d+)\]")
    metric_re = re.compile(r"^\s+([^=]+)=([0-9.,eE+-]+)\s*$")

    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []

    for run_id in run_ids:
        log_path = results_dir / f"{run_id}.log"
        if not log_path.exists():
            print("missing log:", log_path)
            continue

        current_train: dict[str, object] | None = None
        current_eval_label: str | None = None
        current_eval_step: int | None = None
        eval_round = 0
        last_step: int | None = None
        checkpoint_start: datetime | None = None

        for line in log_path.read_text(errors="ignore").splitlines():
            step_match = step_re.search(line)
            if step_match:
                if current_train and "train_cross_entropy" in current_train:
                    train_rows.append(current_train)
                last_step = int(step_match.group(1))
                current_train = {
                    "run_id": run_id,
                    "step": last_step,
                    "max_step": int(step_match.group(2)),
                }
                current_eval_label = None
                continue

            if "Saving checkpoint..." in line:
                checkpoint_start = parse_timestamp(line)
            if "Checkpoint saved to" in line:
                end = parse_timestamp(line)
                duration = (
                    (end - checkpoint_start).total_seconds()
                    if end is not None and checkpoint_start is not None
                    else math.nan
                )
                checkpoint_rows.append(
                    {
                        "run_id": run_id,
                        "step": last_step,
                        "checkpoint_path": line.split("Checkpoint saved to", 1)[-1].strip(),
                        "save_seconds": duration,
                    }
                )
                checkpoint_start = None

            if "INFO\tbooks_val" in line:
                eval_round += 1
                current_eval_step = eval_round * eval_interval
                current_eval_label = "books_val"
                continue
            if "INFO\tc4_val_proxy" in line:
                if current_eval_step is None:
                    current_eval_step = max(eval_round, 1) * eval_interval
                current_eval_label = "c4_val_proxy"
                continue

            metric_match = metric_re.match(line)
            if not metric_match:
                continue
            name = metric_match.group(1).strip()
            value = parse_number(metric_match.group(2))

            if current_eval_label and name.startswith(f"eval/{current_eval_label}/"):
                eval_rows.append(
                    {
                        "run_id": run_id,
                        "step": current_eval_step or last_step,
                        "label": current_eval_label,
                        "metric": name.split("/")[-1],
                        "value": value,
                    }
                )
                continue

            if current_train is None:
                continue
            if name == "train/CrossEntropyLoss":
                current_train["train_cross_entropy"] = value
            elif name == "train/Perplexity":
                current_train["train_perplexity"] = value
            elif name == "throughput/device/tokens_per_second":
                current_train["tokens_per_second"] = value
            elif name == "throughput/device/batches_per_second":
                current_train["batches_per_second"] = value
            elif name == "throughput/total_tokens":
                current_train["total_tokens"] = value
            elif name == "System/Peak GPU Memory (MB)":
                current_train["peak_gpu_memory_mb"] = value

        if current_train and "train_cross_entropy" in current_train:
            train_rows.append(current_train)

    return (
        pd.DataFrame(train_rows).sort_values(["run_id", "step"]) if train_rows else pd.DataFrame(),
        pd.DataFrame(eval_rows).sort_values(["run_id", "label", "step", "metric"]) if eval_rows else pd.DataFrame(),
        pd.DataFrame(checkpoint_rows) if checkpoint_rows else pd.DataFrame(),
    )


def load_meta_frames(
    train_data_dir: Path,
    run_ids: Iterable[str],
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        data_run_id = run_id
        parquet_path = train_data_dir / data_run_id / "train_meta.parquet"
        csv_path = train_data_dir / data_run_id / "train_meta.csv"
        if not parquet_path.exists() and not csv_path.exists():
            candidate = base_run_id(run_id)
            if candidate != run_id:
                candidate_parquet = train_data_dir / candidate / "train_meta.parquet"
                candidate_csv = train_data_dir / candidate / "train_meta.csv"
                if candidate_parquet.exists() or candidate_csv.exists():
                    data_run_id = candidate
                    parquet_path = candidate_parquet
                    csv_path = candidate_csv
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            frame = pd.read_csv(csv_path)
        else:
            if allow_missing:
                print("missing train metadata:", parquet_path)
                continue
            raise FileNotFoundError(parquet_path)
        frame["run_id"] = run_id
        frame["selection_source_run_id"] = data_run_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def expand_selection_for_run_ids(selection: pd.DataFrame, run_ids: Iterable[str]) -> pd.DataFrame:
    if selection.empty or "run_id" not in selection.columns:
        return selection
    rows: list[pd.Series] = []
    selection_by_run = {str(row["run_id"]): row for _, row in selection.iterrows()}
    for run_id in run_ids:
        source_run_id = run_id if run_id in selection_by_run else base_run_id(run_id)
        if source_run_id not in selection_by_run:
            continue
        row = selection_by_run[source_run_id].copy()
        row["run_id"] = run_id
        row["selection_source_run_id"] = source_run_id
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else selection.iloc[0:0].copy()


def overlap_from_meta(meta_all: pd.DataFrame) -> pd.DataFrame:
    if meta_all.empty or "seq_idx" not in meta_all.columns or "run_id" not in meta_all.columns:
        return pd.DataFrame()
    run_order = list(dict.fromkeys(meta_all["run_id"].astype(str)))
    selected = {
        run_id: set(frame["seq_idx"].dropna().astype(int).tolist())
        for run_id, frame in meta_all.groupby("run_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for left_index, left_run_id in enumerate(run_order):
        for right_run_id in run_order[left_index + 1 :]:
            left = selected[left_run_id]
            right = selected[right_run_id]
            union_count = len(left | right)
            intersection_count = len(left & right)
            rows.append(
                {
                    "left_run_id": left_run_id,
                    "right_run_id": right_run_id,
                    "seq_idx_intersection": intersection_count,
                    "seq_idx_union": union_count,
                    "seq_idx_jaccard": intersection_count / union_count if union_count else math.nan,
                }
            )
    return pd.DataFrame(rows)


def savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print("wrote:", path)


def add_variant_label(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "run_id" not in frame.columns:
        return frame
    labelled = frame.copy()
    labelled.insert(1, "variant", labelled["run_id"].map(run_label))
    return labelled


def sort_by_run_order(frame: pd.DataFrame, run_ids: Iterable[str], column: str = "run_id") -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    order = {run_id: index for index, run_id in enumerate(run_ids)}
    ordered = frame.copy()
    ordered["_run_order"] = ordered[column].map(order).fillna(len(order))
    return ordered.sort_values("_run_order").drop(columns="_run_order").reset_index(drop=True)


def label_overlap_frame(frame: pd.DataFrame, run_ids: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    labelled = frame.copy()
    order = {run_id: index for index, run_id in enumerate(run_ids)}
    if "left_run_id" in labelled.columns:
        labelled.insert(
            labelled.columns.get_loc("left_run_id") + 1,
            "left_variant",
            labelled["left_run_id"].map(run_label),
        )
        labelled["_left_order"] = labelled["left_run_id"].map(order).fillna(len(order))
    if "right_run_id" in labelled.columns:
        labelled.insert(
            labelled.columns.get_loc("right_run_id") + 1,
            "right_variant",
            labelled["right_run_id"].map(run_label),
        )
        labelled["_right_order"] = labelled["right_run_id"].map(order).fillna(len(order))
    sort_columns = [col for col in ["_left_order", "_right_order"] if col in labelled.columns]
    if sort_columns:
        labelled = labelled.sort_values(sort_columns).drop(columns=sort_columns).reset_index(drop=True)
    return labelled


def generate_figures(
    *,
    train_data_dir: Path,
    reports_dir: Path,
    run_ids: list[str],
    train_metrics: pd.DataFrame,
    eval_metrics: pd.DataFrame,
    overlap: pd.DataFrame,
    allow_log_only_runs: bool,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir = reports_dir / "figures"
    plt.style.use("default")
    run_groups = dynamic_run_groups(run_ids)

    def plot_metric_lines(
        frame: pd.DataFrame,
        *,
        run_subset: Iterable[str],
        y_column: str,
        filename: str,
        title: str,
        ylabel: str,
    ) -> None:
        plt.figure(figsize=(8, 5))
        plotted = False
        for run_id in run_subset:
            group = frame[frame["run_id"] == run_id]
            if group.empty or y_column not in group.columns:
                continue
            plt.plot(
                group["step"],
                group[y_column],
                marker="o",
                linewidth=1.5,
                label=run_label(run_id),
            )
            plotted = True
        plt.xlabel("Optimizer step")
        plt.ylabel(ylabel)
        plt.title(title)
        if plotted:
            plt.legend(fontsize=8)
        else:
            plt.text(0.5, 0.5, "No metrics parsed", ha="center", va="center")
        plt.grid(alpha=0.3)
        savefig(figures_dir / filename)

    def plot_learning_family(
        frame: pd.DataFrame,
        *,
        y_column: str,
        filename_prefix: str,
        title_metric: str,
        ylabel: str,
    ) -> None:
        plot_metric_lines(
            frame,
            run_subset=run_ids,
            y_column=y_column,
            filename=f"{filename_prefix}_by_run.png",
            title=f"{title_metric} By Run",
            ylabel=ylabel,
        )
        for suffix, group in run_groups.items():
            plot_metric_lines(
                frame,
                run_subset=group["run_ids"],
                y_column=y_column,
                filename=f"{filename_prefix}_{suffix}.png",
                title=f"{group['title']}: {title_metric}",
                ylabel=ylabel,
            )

    if not train_metrics.empty:
        plot_learning_family(
            train_metrics,
            y_column="train_cross_entropy",
            filename_prefix="train_loss",
            title_metric="Training Loss",
            ylabel="Train cross entropy",
        )
        plot_learning_family(
            train_metrics,
            y_column="tokens_per_second",
            filename_prefix="tokens_per_second",
            title_metric="Throughput",
            ylabel="Device tokens/sec",
        )

    eval_ce = (
        eval_metrics[eval_metrics["metric"] == "CrossEntropyLoss"].copy()
        if not eval_metrics.empty
        else pd.DataFrame()
    )
    for label, filename, title in [
        ("books_val", "eval_loss_books_by_run.png", "Books Validation Loss By Run"),
        ("c4_val_proxy", "eval_loss_c4_by_run.png", "C4 Validation Proxy Loss By Run"),
    ]:
        frame = eval_ce[eval_ce["label"] == label] if not eval_ce.empty else pd.DataFrame()
        if frame.empty:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, f"No {label} eval metrics parsed", ha="center", va="center")
            plt.xlabel("Optimizer step")
            plt.ylabel("Eval cross entropy")
            plt.title(title)
            plt.grid(alpha=0.3)
            savefig(figures_dir / filename)
            continue
        title_metric = "Books Validation Loss" if label == "books_val" else "C4 Validation Proxy Loss"
        plot_learning_family(
            frame,
            y_column="value",
            filename_prefix="eval_loss_books" if label == "books_val" else "eval_loss_c4",
            title_metric=title_metric,
            ylabel="Eval cross entropy",
        )
        if label == "books_val":
            plot_metric_lines(
                frame,
                run_subset=hard_seed_pair_run_ids(run_ids),
                y_column="value",
                filename="eval_loss_books_hard_oracle_vs_cascade_seeds.png",
                title="Hard Source Books Validation Loss: Oracle vs Cascade Seeds",
                ylabel="Eval cross entropy",
            )

    meta_all = load_meta_frames(train_data_dir, run_ids, allow_missing=allow_log_only_runs)
    for column, filename, title in [
        ("local_full_color_score", "selection_full_score_distributions.png", "Selected Full CoLoR Score Distributions"),
        ("pair_mid2_color_score", "selection_pair_mid2_score_distributions.png", "Selected Pair-Mid2 Score Distributions"),
    ]:
        if column not in meta_all.columns:
            fallback = "full_color_score" if column == "local_full_color_score" else column
            column_to_plot = fallback if fallback in meta_all.columns else None
        else:
            column_to_plot = column
        plt.figure(figsize=(8, 5))
        if column_to_plot and not meta_all.empty:
            for run_id, frame in meta_all.groupby("run_id"):
                values = frame[column_to_plot].dropna().to_numpy()
                plt.hist(values, bins=60, alpha=0.35, density=True, label=run_label(run_id))
            plt.xlabel(column_to_plot)
            plt.ylabel("Density")
            plt.legend(fontsize=8)
        elif meta_all.empty:
            plt.text(0.5, 0.5, "No local selection metadata found", ha="center", va="center")
        else:
            plt.text(0.5, 0.5, f"Missing column {column}", ha="center", va="center")
        plt.title(title)
        plt.grid(alpha=0.2)
        savefig(figures_dir / filename)

    def available_overlap_run_ids(candidate_run_ids: Iterable[str]) -> list[str]:
        selected_run_ids = list(candidate_run_ids)
        if allow_log_only_runs and not overlap.empty:
            overlap_ids = set(overlap["left_run_id"]).union(set(overlap["right_run_id"]))
            selected_run_ids = [run_id for run_id in selected_run_ids if run_id in overlap_ids]
        return selected_run_ids

    def plot_overlap_heatmap(candidate_run_ids: Iterable[str], filename: str, title: str) -> None:
        selected_run_ids = available_overlap_run_ids(candidate_run_ids)
        plt.figure(figsize=(7, 6))
        if not selected_run_ids:
            plt.text(0.5, 0.5, "No local selection-overlap metadata found", ha="center", va="center")
            plt.title(title)
            savefig(figures_dir / filename)
            return
        matrix = pd.DataFrame(np.eye(len(selected_run_ids)), index=selected_run_ids, columns=selected_run_ids)
        for _, row in overlap.iterrows():
            left = row["left_run_id"]
            right = row["right_run_id"]
            if left not in matrix.index or right not in matrix.columns:
                continue
            value = row.get("seq_idx_jaccard", np.nan)
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
        image = plt.imshow(matrix.loc[selected_run_ids, selected_run_ids], vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(image, label="Jaccard overlap")
        plt.xticks(
            range(len(selected_run_ids)),
            [run_label(r) for r in selected_run_ids],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        plt.yticks(range(len(selected_run_ids)), [run_label(r) for r in selected_run_ids], fontsize=8)
        for i in range(len(selected_run_ids)):
            for j in range(len(selected_run_ids)):
                value = matrix.iloc[i, j]
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.5 else "black",
                    fontsize=8,
                )
        plt.title(title)
        savefig(figures_dir / filename)

    plot_overlap_heatmap(run_ids, "selected_set_overlap_heatmap.png", "Selected Set Overlap Matrix")
    plot_overlap_heatmap(
        run_groups["random_source"]["run_ids"],
        "selected_set_overlap_random_source.png",
        "Random-source Selected Set Overlap",
    )
    plot_overlap_heatmap(
        run_groups["hard_source"]["run_ids"],
        "selected_set_overlap_hard_source.png",
        "Hard-source Selected Set Overlap",
    )


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, floatfmt: str = ".4f") -> str:
    if columns is not None:
        frame = frame[columns].copy()
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    *,
    run_ids: list[str],
    train_data_dir: Path,
    results_dir: Path,
    reports_dir: Path,
    checkpoints_dir: Path,
    eval_manifest_path: Path,
    runtime_config_dir: str,
    ablation_sha: str,
    olmo_sha: str,
    sequence_length: int,
    eval_subset_num_batches: int,
    device_eval_batch_size: int,
    selection: pd.DataFrame,
    sensitivity: pd.DataFrame,
    overlap: pd.DataFrame,
    train_metrics: pd.DataFrame,
    eval_metrics: pd.DataFrame,
    allow_log_only_runs: bool,
) -> None:
    final_train = (
        train_metrics.sort_values(["run_id", "step"]).groupby("run_id").tail(1).reset_index(drop=True)
        if not train_metrics.empty
        else pd.DataFrame()
    )
    final_train = add_variant_label(sort_by_run_order(final_train, run_ids))

    eval_summary = pd.DataFrame()
    if not eval_metrics.empty:
        eval_ce = eval_metrics[eval_metrics["metric"] == "CrossEntropyLoss"]
        eval_summary = (
            eval_ce.sort_values(["run_id", "label", "step"])
            .groupby(["run_id", "label"])
            .tail(1)
            .pivot(index="run_id", columns="label", values="value")
            .reset_index()
        )
        eval_summary = add_variant_label(sort_by_run_order(eval_summary, run_ids))

    manifest = json.loads(eval_manifest_path.read_text())
    run_count_word = {4: "four", 5: "five", 6: "six"}.get(len(run_ids), str(len(run_ids)))
    run_groups = dynamic_run_groups(run_ids)
    seed_repeat_values = sorted({seed for seed in (run_seed(run_id) for run_id in run_ids) if seed is not None})
    has_seed_repeats = bool(seed_repeat_values)
    seed_line = (
        "- Training seeds: base runs use `17`; hard-source repeat runs use "
        f"`{', '.join(str(seed) for seed in seed_repeat_values)}`."
        if has_seed_repeats
        else "- Training seed: `17`."
    )
    setup_summary = (
        "same architecture, optimizer, scheduler, tokenizer, batch size, sequence length, "
        "and eval schedule. Seed-repeat runs intentionally vary only the training seed while "
        "reusing the matching base selected training data."
        if has_seed_repeats
        else "same architecture, seed, optimizer, scheduler, tokenizer, batch size, sequence length, "
        "and eval schedule; they differ only in the selected training data."
    )
    selection = add_variant_label(sort_by_run_order(selection, run_ids))
    sensitivity = add_variant_label(sort_by_run_order(sensitivity, run_ids))
    overlap = label_overlap_frame(overlap, run_ids)
    random_overlap = overlap[
        overlap["left_run_id"].isin(run_groups["random_source"]["run_ids"])
        & overlap["right_run_id"].isin(run_groups["random_source"]["run_ids"])
    ].reset_index(drop=True)
    hard_overlap = overlap[
        overlap["left_run_id"].isin(run_groups["hard_source"]["run_ids"])
        & overlap["right_run_id"].isin(run_groups["hard_source"]["run_ids"])
    ].reset_index(drop=True)
    selection_run_ids = set(selection["run_id"]) if "run_id" in selection.columns else set()
    missing_selection_run_ids = [run_id for run_id in run_ids if run_id not in selection_run_ids]

    report: list[str] = [
        "# 410M Score-Pool Mini-Universe Training Report",
        "",
        "## Executive Summary",
        "",
        (
            f"This report compares {run_count_word} 410M-class OLMo-style models trained for two passes over "
            "matched 100K-row score-pool mini-universe selections. All production configs use the "
            f"{setup_summary}"
        ),
        "",
        "## Data Provenance",
        "",
        f"- Training data: `{train_data_dir}`",
        f"- Eval data: `{eval_manifest_path.parent}`",
        f"- Books eval source: `{manifest['books_val']['source']}`",
        f"- C4 eval source: `{manifest['c4_val_proxy']['source']}`",
        f"- C4 caveat: {manifest['c4_val_proxy']['note']}",
        "",
        "## Mini-Universe Definitions",
        "",
        "- `random_positive_oracle_100k` (Random positive-only baseline): all 100K rows from `random_positive`.",
        "- `hard_positive_oracle_100k` (Hard positive-only baseline): all 100K rows from `hard_positive`.",
        (
            "- `random_pair_cascade_100k` (Random PN cascade): rank `random_positive union random_negative` "
            "with `pair_mid2`, keep `m * 100K` candidates, then rerank by full CoLoR."
        ),
        (
            "- `random_union_control_100k` (Random PN mixture control): random 100K rows "
            "from `random_positive union random_negative`."
        ),
        (
            "- `hard_pair_cascade_100k` (Hard PN cascade): rank `hard_positive union hard_negative` "
            "with `pair_mid2`, keep `m * 100K` candidates, then rerank by full CoLoR."
        ),
        (
            "- `hard_union_control_100k` (Hard PN mixture control): random 100K rows "
            "from `hard_positive union hard_negative`."
        ),
        (
            "- `hard_pair_mid2_only_100k` (Hard PN pair-mid2-only): choose the lowest 100K "
            "`pair_mid2` scores directly from `hard_positive union hard_negative`, with no full rerank."
        ),
        (
            "- `hard_dropout_embed_p000001_conservative_100k`, "
            "`hard_dropout_embed_p0005_conservative_100k`, and "
            "`hard_dropout_embed_p001_conservative_100k` (Hard PN embedding-dropout LCB): "
            "choose the lowest 100K `mean_color + std_color` scores from the hard union at "
            "embedding dropout rates `1e-5`, `0.005`, and `0.01`, respectively."
        ),
        *(
            [
                (
                    "- Seed-repeat run IDs such as `hard_pair_cascade_seed18_100k` reuse the matching "
                    "base 100K training selection and vary only the training seed/log/checkpoint identity."
                )
            ]
            if has_seed_repeats
            else []
        ),
        "- PN means the positive/negative source-pool union for that source family.",
        "- Cascade multiplier for trained cascade runs: `m = 1.5`.",
        "- Score convention: `conditional_books_loss - prior_loss`; lower is better.",
        "",
        "## Matched Training Setup",
        "",
        "- Total parameters: `522,097,920`.",
        "- Non-embedding parameters: `393,319,680`.",
        "- Architecture: `d_model=1280`, `n_layers=20`, `n_heads=20`, sequence length `512`.",
        seed_line,
        "- Global train batch size: `256` sequences.",
        "- Training duration: `2ep`, which is `780` optimizer steps for each 100K-row memmap.",
        "- Eval interval: every `78` optimizer steps, including final step `780`.",
        "",
        "## Selection Diagnostics",
        "",
        *(
            [
                (
                    "_Local selection metadata is not mirrored for "
                    f"`{', '.join(missing_selection_run_ids)}`; learning curves and final metrics "
                    "include those run logs, while selection diagnostics and overlap use the "
                    "available local training-set metadata._"
                ),
                "",
            ]
            if allow_log_only_runs and missing_selection_run_ids
            else []
        ),
        md_table(
            selection,
            [
                "run_id",
                "variant",
                "selected_rows",
                "true_positive_count",
                "true_positive_rate",
                "oracle_positive_recall",
            ],
        ),
        "",
        "## Cascade Multiplier Sensitivity",
        "",
        (
            md_table(
                sensitivity,
                [
                    col
                    for col in [
                        "run_id",
                        "variant",
                        "cascade_multiplier",
                        "candidate_count",
                        "true_positive_count",
                        "true_positive_rate",
                        "oracle_positive_recall",
                        "trained_p0",
                    ]
                    if col in sensitivity.columns
                ],
            )
            if not sensitivity.empty
            else "_No sensitivity diagnostics found._"
        ),
        "",
        "## Selected Set Overlap Matrices",
        "",
        (
            "Entries are Jaccard overlap between selected training-row sets over `seq_idx`, "
            "split into random-source and hard-source comparisons."
        ),
        *(
            [
                (
                    "_Union controls without local selected-row metadata are omitted from their "
                    "source-family overlap matrices until those metadata files are mirrored._"
                ),
                "",
            ]
            if any(run_id.endswith("union_control_100k") for run_id in missing_selection_run_ids)
            else []
        ),
        "",
        "### Random Source",
        "",
        md_table(random_overlap) if not random_overlap.empty else "_No random-source overlap diagnostics parsed._",
        "",
        "### Hard Source",
        "",
        md_table(hard_overlap) if not hard_overlap.empty else "_No hard-source overlap diagnostics parsed._",
        "",
        "## Final Training Metrics",
        "",
    ]
    if not final_train.empty:
        report.append(
            md_table(
                final_train,
                [
                    "run_id",
                    "variant",
                    "step",
                    "train_cross_entropy",
                    "train_perplexity",
                    "tokens_per_second",
                    "peak_gpu_memory_mb",
                ],
            )
        )
    else:
        report.append("_No training metrics parsed._")
    report.extend(["", "## Final Eval Metrics", ""])
    report.append(
        md_table(eval_summary, ["run_id", "variant", "books_val", "c4_val_proxy"])
        if not eval_summary.empty
        else "_No eval metrics parsed._"
    )
    report.extend(
        [
            "",
            "## Figures",
            "",
            (
                "Learning curves are split by source family so each panel compares selection rules "
                "within the same positive/negative universe."
            ),
            "",
        ]
    )
    for filename, caption in [
        ("train_loss_random_source.png", "Random-source training cross entropy over optimizer steps."),
        ("train_loss_hard_source.png", "Hard-source training cross entropy over optimizer steps."),
        ("eval_loss_books_random_source.png", "Random-source Books validation cross entropy over optimizer steps."),
        ("eval_loss_books_hard_source.png", "Hard-source Books validation cross entropy over optimizer steps."),
        (
            "eval_loss_books_hard_oracle_vs_cascade_seeds.png",
            "Hard-source Books validation cross entropy for oracle and cascade seed repeats.",
        ),
        ("eval_loss_c4_random_source.png", "Random-source C4 validation proxy cross entropy over optimizer steps."),
        ("eval_loss_c4_hard_source.png", "Hard-source C4 validation proxy cross entropy over optimizer steps."),
        ("tokens_per_second_random_source.png", "Random-source device tokens per second over optimizer steps."),
        ("tokens_per_second_hard_source.png", "Hard-source device tokens per second over optimizer steps."),
        ("selected_set_overlap_random_source.png", "Random-source selected-set overlap matrix, reported as Jaccard overlap over `seq_idx`."),
        ("selected_set_overlap_hard_source.png", "Hard-source selected-set overlap matrix, reported as Jaccard overlap over `seq_idx`."),
        ("selection_full_score_distributions.png", "Distribution of selected full CoLoR scores."),
        ("selection_pair_mid2_score_distributions.png", "Distribution of selected pair-mid2 CoLoR scores."),
    ]:
        report.extend([f"![{caption}](figures/{filename})", ""])
    report.extend(
        [
            "## Reproducibility Appendix",
            "",
            f"- CoLoR-ablation SHA: `{ablation_sha}`",
            f"- color-filter-olmo SHA: `{olmo_sha}`",
            f"- Runtime config dir: `{runtime_config_dir}`",
            f"- Checkpoints: `{checkpoints_dir}`",
            f"- Results: `{results_dir}`",
            f"- Reports: `{reports_dir}`",
            f"- Sequence length: `{sequence_length}`",
            f"- Eval subset batches: `{eval_subset_num_batches}`",
            f"- Device eval batch size: `{device_eval_batch_size}`",
            "",
            "## Limitations",
            "",
            (
                "- Seed-repeat coverage is limited to the hard positive-only baseline and hard PN cascade."
                if has_seed_repeats
                else "- This is a single-seed pilot."
            ),
            (
                "- The C4 metric is a fixed public validation proxy because the original CoLoR-Filter "
                "downstream data exposes Books validation but not a C4 validation memmap."
            ),
            "- The model is 410M-class by non-embedding parameters and is not a 1.2B reproduction.",
        ]
    )

    report_md = reports_dir / "report.md"
    report_md.write_text("\n".join(report), encoding="utf-8")
    print("wrote:", report_md, report_md.stat().st_size)

    try:
        import markdown

        body = markdown.markdown(report_md.read_text(encoding="utf-8"), extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(report_md.read_text(encoding="utf-8")) + "</pre>"

    report_html = reports_dir / "report.html"
    report_html.write_text("<html><body>" + body + "</body></html>", encoding="utf-8")
    print("wrote:", report_html, report_html.stat().st_size)


def write_metrics_and_report(args: argparse.Namespace) -> None:
    run_ids = list(args.run_id or DEFAULT_RUN_IDS)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    train_metrics, eval_metrics, checkpoint_saves = parse_logs(args.results_dir, run_ids, args.eval_interval)
    train_metrics.to_csv(args.results_dir / "train_metrics_from_logs.csv", index=False)
    eval_metrics.to_csv(args.results_dir / "eval_metrics_from_logs.csv", index=False)
    checkpoint_saves.to_csv(args.results_dir / "checkpoint_save_times.csv", index=False)

    for run_id, frame in train_metrics.groupby("run_id") if not train_metrics.empty else []:
        frame.to_json(args.results_dir / f"{run_id}_train_metrics.jsonl", orient="records", lines=True)

    throughput = pd.DataFrame()
    if not train_metrics.empty:
        throughput = (
            train_metrics.groupby("run_id")
            .agg(
                final_step=("step", "max"),
                final_train_cross_entropy=("train_cross_entropy", "last"),
                final_train_perplexity=("train_perplexity", "last"),
                median_tokens_per_second=("tokens_per_second", "median"),
                max_peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
            )
            .reset_index()
        )
    throughput.to_csv(args.results_dir / "throughput_comparison.csv", index=False)

    selection = expand_selection_for_run_ids(
        pd.read_csv(args.train_data_dir / "selection_diagnostics.csv"),
        run_ids,
    )
    meta_for_overlap = load_meta_frames(
        args.train_data_dir,
        run_ids,
        allow_missing=args.allow_log_only_runs,
    )
    overlap = overlap_from_meta(meta_for_overlap)
    if overlap.empty:
        overlap = pd.read_csv(args.train_data_dir / "overlap_jaccard.csv")
    sensitivity_path = args.train_data_dir / "selection_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    selection.to_csv(args.results_dir / "selection_diagnostics.csv", index=False)
    overlap.to_csv(args.results_dir / "overlap_jaccard.csv", index=False)
    if not sensitivity.empty:
        sensitivity.to_csv(args.results_dir / "selection_sensitivity.csv", index=False)

    checkpoint_manifest = {
        "experiment": args.experiment,
        "checkpoints_root": str(args.checkpoints_dir),
        "runs": {},
    }
    for run_id in run_ids:
        run_dir = args.checkpoints_dir / run_id
        steps = sorted(path for path in run_dir.glob("step*") if path.is_dir()) if run_dir.exists() else []
        checkpoint_manifest["runs"][run_id] = {
            "checkpoint_dir": str(run_dir),
            "step_dirs": [path.name for path in steps],
            "has_final_step780": any(path.name == "step780" for path in steps),
        }
    (args.results_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2))

    generate_figures(
        train_data_dir=args.train_data_dir,
        reports_dir=args.reports_dir,
        run_ids=run_ids,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        overlap=overlap,
        allow_log_only_runs=args.allow_log_only_runs,
    )
    write_report(
        run_ids=run_ids,
        train_data_dir=args.train_data_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        checkpoints_dir=args.checkpoints_dir,
        eval_manifest_path=args.eval_manifest,
        runtime_config_dir=args.runtime_config_dir,
        ablation_sha=args.ablation_sha,
        olmo_sha=args.olmo_sha,
        sequence_length=args.sequence_length,
        eval_subset_num_batches=args.eval_subset_num_batches,
        device_eval_batch_size=args.device_eval_batch_size,
        selection=selection,
        sensitivity=sensitivity,
        overlap=overlap,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        allow_log_only_runs=args.allow_log_only_runs,
    )
    acceptance_check(args)


def acceptance_check(args: argparse.Namespace) -> None:
    args.run_id = list(args.run_id or DEFAULT_RUN_IDS)
    required_artifacts = [
        args.results_dir / "train_metrics_from_logs.csv",
        args.results_dir / "eval_metrics_from_logs.csv",
        args.results_dir / "checkpoint_save_times.csv",
        args.results_dir / "throughput_comparison.csv",
        args.results_dir / "selection_diagnostics.csv",
        args.results_dir / "overlap_jaccard.csv",
        args.results_dir / "checkpoint_manifest.json",
        args.reports_dir / "report.md",
        args.reports_dir / "report.html",
    ]
    if (args.train_data_dir / "selection_sensitivity.csv").exists():
        required_artifacts.append(args.results_dir / "selection_sensitivity.csv")
    required_figures = [args.reports_dir / "figures" / name for name in FIGURES]
    for path in required_artifacts + required_figures:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty artifact: {path}")
        print("artifact ok:", path, path.stat().st_size)

    run_ids = set(args.run_id)
    selection = pd.read_csv(args.results_dir / "selection_diagnostics.csv")
    selection_run_ids = set(selection["run_id"])
    if args.allow_log_only_runs:
        if not selection_run_ids.issubset(run_ids):
            raise AssertionError(selection[["run_id"]])
    elif len(selection) != len(run_ids) or selection_run_ids != run_ids:
        raise AssertionError(selection[["run_id"]])
    if not (selection["selected_rows"] == 100_000).all():
        raise AssertionError(selection[["run_id", "selected_rows"]])
    print(selection[["run_id", "true_positive_count", "true_positive_rate"]])

    train_metrics = pd.read_csv(args.results_dir / "train_metrics_from_logs.csv")
    if set(train_metrics["run_id"]) != run_ids:
        raise AssertionError(train_metrics["run_id"].unique())
    final_steps = train_metrics.groupby("run_id")["step"].max()
    if not (final_steps == 780).all():
        raise AssertionError(final_steps)
    print(train_metrics.sort_values(["run_id", "step"]).groupby("run_id").tail(1))

    eval_metrics = pd.read_csv(args.results_dir / "eval_metrics_from_logs.csv")
    if not {"books_val", "c4_val_proxy"}.issubset(set(eval_metrics["label"])):
        raise AssertionError(eval_metrics["label"].unique())
    if not {"CrossEntropyLoss", "Perplexity"}.issubset(set(eval_metrics["metric"])):
        raise AssertionError(eval_metrics["metric"].unique())
    final_eval_steps = (
        eval_metrics[eval_metrics["metric"] == "CrossEntropyLoss"].groupby(["run_id", "label"])["step"].max()
    )
    if not (final_eval_steps == 780).all():
        raise AssertionError(final_eval_steps)
    expected_eval_points = 780 // args.eval_interval
    expected_eval_index = pd.MultiIndex.from_product(
        [list(args.run_id), ["books_val", "c4_val_proxy"]],
        names=["run_id", "label"],
    )
    eval_counts = (
        eval_metrics[eval_metrics["metric"] == "CrossEntropyLoss"]
        .groupby(["run_id", "label"])["step"]
        .nunique()
        .reindex(expected_eval_index, fill_value=0)
    )
    if not (eval_counts >= expected_eval_points).all():
        raise AssertionError(eval_counts)
    print(
        eval_metrics[eval_metrics["metric"] == "CrossEntropyLoss"]
        .sort_values(["run_id", "label", "step"])
        .groupby(["run_id", "label"])
        .tail(1)
    )

    manifest = json.loads((args.results_dir / "checkpoint_manifest.json").read_text())
    log_checkpoint_steps = pd.read_csv(args.results_dir / "checkpoint_save_times.csv")
    for run_id in args.run_id:
        if not manifest["runs"][run_id]["has_final_step780"]:
            has_log_checkpoint = (
                args.allow_log_only_runs
                and not log_checkpoint_steps.empty
                and (
                    (log_checkpoint_steps["run_id"] == run_id)
                    & (log_checkpoint_steps["step"] == 780)
                ).any()
            )
            if not has_log_checkpoint:
                raise AssertionError(manifest["runs"][run_id])
    print("acceptance checks passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--experiment", default="train-410m-score-pool-mini-universes")
    parser.add_argument("--runtime-config-dir", default="")
    parser.add_argument("--ablation-sha", default="")
    parser.add_argument("--olmo-sha", default="")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--eval-subset-num-batches", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=78)
    parser.add_argument("--device-eval-batch-size", type=int, default=16)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--allow-log-only-runs", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_only:
        acceptance_check(args)
    else:
        write_metrics_and_report(args)


if __name__ == "__main__":
    main()
