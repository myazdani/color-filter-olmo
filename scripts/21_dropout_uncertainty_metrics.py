from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FILE_SEQS = 1_048_576
SCORE_DTYPE = np.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge prior and conditional stochastic score memmaps, persist raw MC samples, "
            "and write CoLoR distribution summaries."
        )
    )
    parser.add_argument("--prior-score-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--conditional-score-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--full-scores", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=float("nan"))
    parser.add_argument("--dropout-target", default="attention+residual+embedding")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--coupled-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lcb-alpha", type=float, default=1.0)
    parser.add_argument("--file-seqs", type=int, default=FILE_SEQS)
    parser.add_argument("--dtype", default="float32", choices=("float32",))
    parser.add_argument("--summary-format", default="parquet", choices=("parquet", "csv", "both"))
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument("--compress-npz", action="store_true")
    return parser.parse_args()


def normalize_score_dir(path: Path) -> Path:
    if (path / "files.txt").exists():
        return path
    nested = path / "score"
    if (nested / "files.txt").exists():
        return nested
    raise FileNotFoundError(f"Could not find files.txt in {path} or {nested}")


def score_files(score_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for line in (score_dir / "files.txt").read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.exists():
            fallback = score_dir / path.name
            if fallback.exists():
                path = fallback
        if not path.exists():
            raise FileNotFoundError(f"Score shard listed in {score_dir / 'files.txt'} does not exist: {line}")
        paths.append(path)
    if not paths:
        raise ValueError(f"No score shards listed in {score_dir / 'files.txt'}")
    return paths


def infer_score_width(path: Path, file_seqs: int) -> int:
    n_values = path.stat().st_size // np.dtype(SCORE_DTYPE).itemsize
    if n_values % file_seqs != 0:
        raise ValueError(f"Could not infer score width for {path}; values={n_values}, file_seqs={file_seqs}")
    width = n_values // file_seqs
    if width < 1:
        raise ValueError(f"Inferred invalid score width {width} for {path}")
    return width


def infer_index_len(index_path: Path, max_rows: int | None = None, chunk_size: int = 1_000_000) -> int:
    index = np.memmap(index_path, dtype=np.int64, mode="r")
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("--max-rows must be positive")
        return min(max_rows, len(index))

    end = len(index)
    while end > 0:
        start = max(0, end - chunk_size)
        chunk = np.asarray(index[start:end])
        nonzero = np.flatnonzero(chunk)
        if len(nonzero):
            return start + int(nonzero[-1]) + 1
        end = start
    raise ValueError(
        f"Could not infer score row count from {index_path}; pass --max-rows for all-zero/tiny smoke indexes"
    )


def load_index(score_dir: Path, max_rows: int | None) -> np.ndarray:
    index_path = score_dir / "mmap_index.npy"
    if not index_path.exists():
        legacy = score_dir / "index.npy"
        if legacy.exists():
            index = np.load(legacy)
            return np.asarray(index[:max_rows] if max_rows is not None else index, dtype=np.int64)
        raise FileNotFoundError(index_path)
    rows = infer_index_len(index_path, max_rows=max_rows)
    index = np.memmap(index_path, dtype=np.int64, mode="r")
    return np.asarray(index[:rows], dtype=np.int64)


def load_scores(score_dir: Path, rows: int, num_samples: int | None, file_seqs: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    remaining = rows
    expected_width: int | None = None
    for path in score_files(score_dir):
        if remaining <= 0:
            break
        width = infer_score_width(path, file_seqs=file_seqs)
        if expected_width is None:
            expected_width = width
        elif width != expected_width:
            raise ValueError(f"Score width mismatch: expected {expected_width}, found {width} in {path}")
        read_rows = min(file_seqs, remaining)
        memmap = np.memmap(path, dtype=SCORE_DTYPE, mode="r", shape=(file_seqs, width))
        chunks.append(np.asarray(memmap[:read_rows], dtype=np.float32))
        remaining -= read_rows
    if remaining > 0:
        raise ValueError(f"Score directory {score_dir} has too few rows; missing {remaining}")
    scores = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
    if num_samples is not None:
        if num_samples < 1:
            raise ValueError("--num-samples must be positive")
        if scores.shape[1] < num_samples:
            raise ValueError(f"Requested {num_samples} samples, but score width is {scores.shape[1]}")
        scores = scores[:, :num_samples]
    return scores.astype(np.float32, copy=False)


def load_score_collection(
    score_dirs: list[Path],
    *,
    max_rows: int | None,
    num_samples: int | None,
    file_seqs: int,
) -> tuple[np.ndarray, np.ndarray]:
    indexes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    remaining = max_rows
    for score_dir in score_dirs:
        if remaining is not None and remaining <= 0:
            break
        index = load_index(score_dir, max_rows=remaining)
        shard_scores = load_scores(score_dir, len(index), num_samples, file_seqs=file_seqs)
        indexes.append(index)
        scores.append(shard_scores)
        if remaining is not None:
            remaining -= len(index)
    if not indexes:
        raise ValueError("No score rows loaded")
    return np.concatenate(indexes), np.concatenate(scores, axis=0)


def load_optional_frame(path: Path | None):
    if path is None:
        return None
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def first_existing_column(frame: Any, columns: Iterable[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def finite_std(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] <= 1:
        return np.zeros(samples.shape[0], dtype=np.float32)
    return samples.std(axis=1, ddof=1).astype(np.float32)


def quantile(samples: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(samples, q, axis=1).astype(np.float32)


def as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def build_summary(
    *,
    seq_idx: np.ndarray,
    score_index: np.ndarray,
    color_samples: np.ndarray,
    utility_samples: np.ndarray,
    args: argparse.Namespace,
):
    import pandas as pd

    mean_color = color_samples.mean(axis=1).astype(np.float32)
    std_color = finite_std(color_samples)
    mean_utility = utility_samples.mean(axis=1).astype(np.float32)
    std_utility = finite_std(utility_samples)

    frame = pd.DataFrame(
        {
            "seq_idx": seq_idx.astype(np.int64),
            "score_index": score_index.astype(np.int64),
            "config_id": args.config_id,
            "num_samples": color_samples.shape[1],
            "dropout_rate": args.dropout_rate,
            "dropout_target": args.dropout_target,
            "seed": -1 if args.seed is None else args.seed,
            "coupled_masks": args.coupled_masks,
            "sign_convention": "color=conditional-prior; lower_is_better",
            "mean_color": mean_color,
            "std_color": std_color,
            "q01_color": quantile(color_samples, 0.01),
            "q05_color": quantile(color_samples, 0.05),
            "q10_color": quantile(color_samples, 0.10),
            "q50_color": quantile(color_samples, 0.50),
            "q90_color": quantile(color_samples, 0.90),
            "q95_color": quantile(color_samples, 0.95),
            "q99_color": quantile(color_samples, 0.99),
            "mean_utility": mean_utility,
            "std_utility": std_utility,
            "prob_utility_positive": (utility_samples > 0).mean(axis=1).astype(np.float32),
            "score_mean": mean_color,
            "score_conservative": (mean_color + args.lcb_alpha * std_color).astype(np.float32),
            "score_optimistic": (mean_color - args.lcb_alpha * std_color).astype(np.float32),
        }
    )
    return frame


def attach_optional_tables(summary, metadata_path: Path | None, full_scores_path: Path | None):
    metadata = load_optional_frame(metadata_path)
    if metadata is not None:
        if len(metadata) < len(summary):
            raise ValueError(f"Metadata has {len(metadata)} rows, but summary has {len(summary)}")
        metadata = metadata.iloc[: len(summary)].reset_index(drop=True)
        for column in ("pool_name", "c4_index", "row_position"):
            if column in metadata.columns and column not in summary.columns:
                summary[column] = metadata[column].to_numpy()
        if "seq_idx" in metadata.columns:
            summary["meta_seq_idx"] = metadata["seq_idx"].to_numpy()

    full_scores = load_optional_frame(full_scores_path)
    if full_scores is not None:
        if len(full_scores) < len(summary):
            raise ValueError(f"Full-score table has {len(full_scores)} rows, but summary has {len(summary)}")
        full_scores = full_scores.iloc[: len(summary)].reset_index(drop=True)
        color_col = first_existing_column(
            full_scores,
            (
                "full_color_score",
                "ablated_color_score",
                "local_full_color_score",
                "color",
                "color_score",
            ),
        )
        if color_col is None:
            raise ValueError(f"Could not find a full-color score column in {full_scores_path}")
        summary["full_color_score"] = full_scores[color_col].to_numpy(dtype=np.float32)
    return summary


def write_npz(
    path: Path,
    *,
    seq_idx: np.ndarray,
    score_index: np.ndarray,
    prior_samples: np.ndarray,
    conditional_samples: np.ndarray,
    color_samples: np.ndarray,
    utility_samples: np.ndarray,
    summary,
    args: argparse.Namespace,
) -> None:
    metadata_json = json.dumps(
        {
            "config_id": args.config_id,
            "num_samples": int(color_samples.shape[1]),
            "dropout_rate": None if math.isnan(args.dropout_rate) else args.dropout_rate,
            "dropout_target": args.dropout_target,
            "seed": args.seed,
            "coupled_masks": args.coupled_masks,
            "sample_axis_description": "axis 1 is stochastic sample index k",
            "sign_convention": "color=conditional-prior; utility=prior-conditional",
        },
        sort_keys=True,
    )
    if "pool_name" in summary.columns:
        pool_values = summary["pool_name"].astype(str).tolist()
        max_pool_len = max(1, max(len(value) for value in pool_values))
        pool_name = np.asarray(pool_values, dtype=f"U{max_pool_len}")
    else:
        pool_name = np.asarray([""] * len(seq_idx), dtype="U1")
    save = np.savez_compressed if args.compress_npz else np.savez
    kwargs = {
        "seq_idx": seq_idx.astype(np.int64),
        "score_index": score_index.astype(np.int64),
        "pool_name": pool_name,
        "prior_losses": prior_samples.astype(np.float32),
        "conditional_losses": conditional_samples.astype(np.float32),
        "color_samples": color_samples.astype(np.float32),
        "utility_samples": utility_samples.astype(np.float32),
        "metadata_json": np.asarray(metadata_json, dtype=f"U{len(metadata_json)}"),
    }
    if "full_color_score" in summary.columns:
        kwargs["full_color_score"] = summary["full_color_score"].to_numpy(dtype=np.float32)
    save(path, **kwargs)


def write_raw_parquet(path: Path, summary, prior_samples: np.ndarray, conditional_samples: np.ndarray) -> None:
    import pandas as pd

    sample_data: dict[str, np.ndarray] = {}
    for sample_idx in range(prior_samples.shape[1]):
        sample_data[f"prior_loss_sample_{sample_idx:03d}"] = prior_samples[:, sample_idx]
        sample_data[f"conditional_loss_sample_{sample_idx:03d}"] = conditional_samples[:, sample_idx]
        sample_data[f"color_sample_{sample_idx:03d}"] = conditional_samples[:, sample_idx] - prior_samples[:, sample_idx]
    raw = pd.concat([summary.reset_index(drop=True), pd.DataFrame(sample_data)], axis=1)
    raw.to_parquet(path, index=False)


def write_summary_tables(output_dir: Path, summary, summary_format: str) -> dict[str, str | None]:
    paths: dict[str, str | None] = {"summary_parquet_path": None, "summary_csv_path": None}
    if summary_format in ("parquet", "both"):
        path = output_dir / "color_distribution_summary.parquet"
        summary.to_parquet(path, index=False)
        paths["summary_parquet_path"] = str(path)
    if summary_format in ("csv", "both"):
        path = output_dir / "color_distribution_summary.csv"
        summary.to_csv(path, index=False)
        paths["summary_csv_path"] = str(path)
    return paths


def main() -> None:
    args = parse_args()
    if len(args.prior_score_dir) != len(args.conditional_score_dir):
        raise ValueError(
            "--prior-score-dir and --conditional-score-dir must list the same number of shard directories"
        )
    prior_dirs = [normalize_score_dir(path) for path in args.prior_score_dir]
    conditional_dirs = [normalize_score_dir(path) for path in args.conditional_score_dir]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior_index, prior_samples = load_score_collection(
        prior_dirs,
        max_rows=args.max_rows,
        num_samples=args.num_samples,
        file_seqs=args.file_seqs,
    )
    conditional_index, conditional_samples = load_score_collection(
        conditional_dirs,
        max_rows=args.max_rows,
        num_samples=args.num_samples,
        file_seqs=args.file_seqs,
    )
    if len(prior_index) != len(conditional_index):
        raise ValueError(f"Index length mismatch: prior={len(prior_index)}, conditional={len(conditional_index)}")
    if not np.array_equal(prior_index, conditional_index):
        raise ValueError("Prior and conditional score indexes differ; sample alignment is unsafe")
    if prior_samples.shape != conditional_samples.shape:
        raise ValueError(f"Score shape mismatch: prior={prior_samples.shape}, conditional={conditional_samples.shape}")
    if not np.isfinite(prior_samples).all() or not np.isfinite(conditional_samples).all():
        raise ValueError("Found non-finite raw score samples")

    color_samples = (conditional_samples - prior_samples).astype(np.float32)
    utility_samples = -color_samples
    seq_idx = np.arange(len(prior_index), dtype=np.int64)
    summary = build_summary(
        seq_idx=seq_idx,
        score_index=prior_index,
        color_samples=color_samples,
        utility_samples=utility_samples,
        args=args,
    )
    summary = attach_optional_tables(summary, args.metadata, args.full_scores)

    npz_path = args.output_dir / f"mc_samples_{args.config_id}.npz"
    parquet_path = args.output_dir / f"mc_samples_{args.config_id}.parquet"
    manifest_path = args.output_dir / f"mc_samples_{args.config_id}_manifest.json"

    summary_paths = write_summary_tables(args.output_dir, summary, args.summary_format)
    write_npz(
        npz_path,
        seq_idx=seq_idx,
        score_index=prior_index,
        prior_samples=prior_samples,
        conditional_samples=conditional_samples,
        color_samples=color_samples,
        utility_samples=utility_samples,
        summary=summary,
        args=args,
    )
    if not args.skip_parquet:
        write_raw_parquet(parquet_path, summary, prior_samples, conditional_samples)

    manifest = {
        "config_id": args.config_id,
        "rows": int(len(summary)),
        "num_samples": int(prior_samples.shape[1]),
        "prior_score_dirs": [str(path) for path in prior_dirs],
        "conditional_score_dirs": [str(path) for path in conditional_dirs],
        **summary_paths,
        "npz_path": str(npz_path),
        "parquet_path": None if args.skip_parquet else str(parquet_path),
        "metadata_path": None if args.metadata is None else str(args.metadata),
        "full_scores_path": None if args.full_scores is None else str(args.full_scores),
        "dropout_rate": None if math.isnan(args.dropout_rate) else args.dropout_rate,
        "dropout_target": args.dropout_target,
        "seed": args.seed,
        "coupled_masks": args.coupled_masks,
        "sign_convention": "color=conditional-prior; lower_is_better",
        "sample_axis_description": "axis 1 is stochastic sample index k",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if summary_paths["summary_parquet_path"] is not None:
        print("wrote summary parquet:", summary_paths["summary_parquet_path"])
    if summary_paths["summary_csv_path"] is not None:
        print("wrote summary csv:", summary_paths["summary_csv_path"])
    print("wrote raw npz:", npz_path)
    if not args.skip_parquet:
        print("wrote raw parquet:", parquet_path)


if __name__ == "__main__":
    main()
