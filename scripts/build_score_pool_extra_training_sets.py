#!/usr/bin/env python
"""Build additional 100K score-pool training sets from deterministic and MC scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HARD_POOLS = ("hard_positive", "hard_negative")
PAIR_SCORE_SPECS = {
    "hard_pair_mid2_only_100k": {
        "variant_id": "pair_mid2",
        "removed_layers": (5, 6),
        "score_column": "pair_mid2_color_score",
    },
    "hard_pair_mid4_only_100k": {
        "variant_id": "pair_mid4",
        "removed_layers": (4, 5, 6, 7),
        "score_column": "pair_mid4_color_score",
    },
}
LCB_SPECS = {
    "hard_dropout_embed_p000001_conservative_100k": 1e-5,
    "hard_dropout_embed_p0005_conservative_100k": 0.005,
    "hard_dropout_embed_p001_conservative_100k": 0.01,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dropout_selection(value: str) -> tuple[str, Path]:
    run_id, separator, raw_path = value.partition("=")
    if not separator or not run_id or not raw_path:
        raise argparse.ArgumentTypeError("Expected RUN_ID=/path/to/color_distribution_summary.parquet")
    if run_id not in LCB_SPECS:
        raise argparse.ArgumentTypeError(f"Unexpected dropout run ID: {run_id}")
    return run_id, Path(raw_path)


def select_lowest(frame: pd.DataFrame, column: str, rows: int) -> pd.DataFrame:
    if len(frame) < rows:
        raise ValueError(f"Need {rows:,} rows, found {len(frame):,}")
    return frame.sort_values([column, "seq_idx"], kind="mergesort").head(rows).copy()


def _parse_removed_layers(value: object) -> tuple[int, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected a layer-index sequence, found {value!r}")
    return tuple(int(layer) for layer in parsed)


def validate_pair_score_frame(
    scores: pd.DataFrame,
    *,
    run_id: str,
    expected_variant_id: str,
    expected_removed_layers: tuple[int, ...],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "seq_idx",
        "pool_name",
        "ablated_color_score",
        "variant_id",
        "variant_family",
        "cond_removed_layers",
        "marg_removed_layers",
        "cond_kept_layers",
        "marg_kept_layers",
    }
    if missing := sorted(required - set(scores.columns)):
        raise ValueError(f"{run_id}: pair scores missing required columns: {missing}")
    if len(scores) != len(metadata) or scores["seq_idx"].nunique() != len(scores):
        raise ValueError(f"{run_id}: expected one unique pair score for every metadata row")
    if set(scores["seq_idx"].astype(np.int64)) != set(metadata["seq_idx"].astype(np.int64)):
        raise ValueError(f"{run_id}: pair-score seq_idx values do not match the official score pool")
    if set(scores["variant_id"].astype(str)) != {expected_variant_id}:
        raise ValueError(f"{run_id}: expected variant_id={expected_variant_id}")
    if set(scores["variant_family"].astype(str)) != {"paired"}:
        raise ValueError(f"{run_id}: expected paired conditional/marginal ablation")

    expected_removed = tuple(expected_removed_layers)
    for column in ("cond_removed_layers", "marg_removed_layers"):
        actual = {_parse_removed_layers(value) for value in scores[column].dropna().unique()}
        if actual != {expected_removed}:
            raise ValueError(f"{run_id}: expected {column}={expected_removed}, found {sorted(actual)}")
    expected_kept = 12 - len(expected_removed)
    for column in ("cond_kept_layers", "marg_kept_layers"):
        actual = set(scores[column].dropna().astype(int))
        if actual != {expected_kept}:
            raise ValueError(f"{run_id}: expected {column}={expected_kept}, found {sorted(actual)}")
    return scores


def validate_pair_scores(
    path: Path,
    *,
    run_id: str,
    expected_variant_id: str,
    expected_removed_layers: tuple[int, ...],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_pair_score_frame(
        pd.read_parquet(path),
        run_id=run_id,
        expected_variant_id=expected_variant_id,
        expected_removed_layers=expected_removed_layers,
        metadata=metadata,
    )


def write_tokens(tokens: np.ndarray, seq_idx: np.ndarray, path: Path, sequence_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(path, mode="w+", dtype=np.uint16, shape=(len(seq_idx) * sequence_length,))
    for start in range(0, len(seq_idx), 10_000):
        end = min(start + 10_000, len(seq_idx))
        output[start * sequence_length : end * sequence_length] = tokens[seq_idx[start:end]].astype(
            np.uint16, copy=False
        ).reshape(-1)
    output.flush()


def selection_diagnostic(selected: pd.DataFrame, run_id: str, policy: str, score_column: str) -> dict[str, object]:
    positive = selected["pool_name"] == "hard_positive"
    return {
        "run_id": run_id,
        "selection_policy": policy,
        "source_pools": ",".join(HARD_POOLS),
        "target_rows": len(selected),
        "selected_rows": len(selected),
        "unique_seq_idx": int(selected["seq_idx"].nunique()),
        "unique_c4_index": int(selected["c4_index"].nunique()),
        "universe_rows": 200_000,
        "positive_pool": "hard_positive",
        "true_positive_count": int(positive.sum()),
        "true_positive_rate": float(positive.mean()),
        "false_positive_count": int((~positive).sum()),
        "oracle_positive_count": 100_000,
        "oracle_positive_recall": float(positive.sum() / 100_000),
        "selection_score_column": score_column,
        "sequence_length": 512,
    }


def resolve_summary_dropout_rate(summary: pd.DataFrame, *, target: str) -> float:
    target_column = f"{target}_dropout"
    if target_column not in summary.columns:
        raise ValueError(f"Summary is missing target-specific rate column: {target_column}")

    generic_rates = summary["dropout_rate"].dropna().astype(float).unique()
    target_rates = summary[target_column].dropna().astype(float).unique()
    if len(target_rates) != 1 or not math.isfinite(float(target_rates[0])):
        raise ValueError(f"Expected one finite {target_column} value, found {target_rates}")
    resolved = float(target_rates[0])
    if len(generic_rates) > 1:
        raise ValueError(f"Expected at most one generic dropout_rate value, found {generic_rates}")
    if len(generic_rates) == 1 and not math.isclose(
        float(generic_rates[0]), resolved, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Generic dropout_rate={generic_rates[0]} disagrees with {target_column}={resolved}"
        )
    return resolved


def validate_dropout_summary(path: Path, *, run_id: str, expected_rate: float, metadata: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = pd.read_parquet(path)
    required = {
        "seq_idx",
        "pool_name",
        "score_mean",
        "std_color",
        "score_conservative",
        "dropout_rate",
        "dropout_target",
        "embedding_dropout",
        "num_samples",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if len(summary) != len(metadata) or summary["seq_idx"].nunique() != len(summary):
        raise ValueError(f"{path} does not contain one unique score for every metadata row")
    if set(summary["seq_idx"].astype(np.int64)) != set(metadata["seq_idx"].astype(np.int64)):
        raise ValueError(f"{path} seq_idx values do not match the official score pool")
    if set(summary["dropout_target"].astype(str)) != {"embedding"}:
        raise ValueError(f"{run_id}: expected embedding-only dropout")
    if set(summary["num_samples"].astype(int)) != {8}:
        raise ValueError(f"{run_id}: expected K=8")
    resolved_rate = resolve_summary_dropout_rate(summary, target="embedding")
    if not math.isclose(resolved_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{run_id}: expected dropout rate {expected_rate}, found {resolved_rate}")
    conservative = summary["score_mean"].astype(float) + summary["std_color"].astype(float)
    if not np.allclose(conservative, summary["score_conservative"].astype(float), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{run_id}: score_conservative is not mean_color + std_color")
    return summary


def write_dataset(
    *,
    output_dir: Path,
    run_id: str,
    selected: pd.DataFrame,
    tokens: np.ndarray,
    selection_policy: str,
    score_column: str,
    source_artifact: Path,
    source_artifact_sha256: str,
) -> dict[str, object]:
    run_dir = output_dir / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("run_id") != run_id
            or manifest.get("actual_unique_rows") != len(selected)
            or manifest.get("source_artifact") != str(source_artifact)
            or manifest.get("source_artifact_sha256") != source_artifact_sha256
        ):
            raise RuntimeError(f"Existing dataset is incompatible: {run_dir}")
        print("existing validated dataset:", run_dir)
        return selection_diagnostic(selected, run_id, selection_policy, score_column)

    selected = selected.copy().sort_values("seq_idx", kind="mergesort").reset_index(drop=True)
    selected.insert(0, "run_id", run_id)
    selected["selection_policy"] = selection_policy
    selected["selection_source_pools"] = ",".join(HARD_POOLS)
    write_tokens(tokens, selected["seq_idx"].to_numpy(dtype=np.int64), run_dir / "train_tokens.npy", 512)
    selected.to_parquet(run_dir / "train_meta.parquet", index=False)
    diagnostic = selection_diagnostic(selected, run_id, selection_policy, score_column)
    manifest = {
        **diagnostic,
        "actual_rows": len(selected),
        "actual_unique_rows": int(selected["seq_idx"].nunique()),
        "source_pool_names": list(HARD_POOLS),
        "source_universe": ",".join(HARD_POOLS),
        "score_sign_convention": "color=conditional_books_loss_minus_prior_loss_lower_is_better",
        "lcb_equivalent": "select lowest mean_color + 1.0 * std_color" if "conservative" in run_id else None,
        "source_artifact": str(source_artifact),
        "source_artifact_sha256": source_artifact_sha256,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "token_dtype": "uint16",
        "raw_memmap_note": "train_tokens.npy is an OLMo-compatible raw memmap despite the .npy suffix.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote:", run_dir)
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-mid2-scores", type=Path, required=True)
    parser.add_argument("--pair-mid4-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dropout-selection", action="append", type=parse_dropout_selection, default=[])
    parser.add_argument("--target-rows", type=int, default=100_000)
    args = parser.parse_args()

    provided_dropout = dict(args.dropout_selection)
    if set(provided_dropout) != set(LCB_SPECS):
        missing = sorted(set(LCB_SPECS) - set(provided_dropout))
        extra = sorted(set(provided_dropout) - set(LCB_SPECS))
        raise ValueError(f"Provide exactly the three dropout selections; missing={missing}, extra={extra}")

    tokens = np.load(args.tokens, mmap_mode="r")
    metadata = pd.read_parquet(args.metadata)
    if tokens.shape != (len(metadata), 512):
        raise ValueError(f"Token shape {tokens.shape} does not match metadata rows {len(metadata):,}")
    required_meta = {"seq_idx", "pool_name", "c4_index"}
    if missing := sorted(required_meta - set(metadata.columns)):
        raise ValueError(f"Metadata missing columns: {missing}")
    hard_metadata = metadata[metadata["pool_name"].isin(HARD_POOLS)].copy()
    if len(hard_metadata) != 200_000:
        raise ValueError(f"Expected 200K hard-union rows, found {len(hard_metadata):,}")

    diagnostics = []
    pair_paths = {
        "hard_pair_mid2_only_100k": args.pair_mid2_scores,
        "hard_pair_mid4_only_100k": args.pair_mid4_scores,
    }
    for run_id, spec in PAIR_SCORE_SPECS.items():
        pair_path = pair_paths[run_id]
        pair_scores = validate_pair_scores(
            pair_path,
            run_id=run_id,
            expected_variant_id=str(spec["variant_id"]),
            expected_removed_layers=tuple(spec["removed_layers"]),
            metadata=metadata,
        )
        score_column = str(spec["score_column"])
        pair_scores = pair_scores.rename(columns={"ablated_color_score": score_column})
        hard_pair = hard_metadata.merge(
            pair_scores[
                [
                    "seq_idx",
                    "pool_name",
                    score_column,
                    "variant_id",
                    "variant_family",
                    "cond_removed_layers",
                    "marg_removed_layers",
                    "cond_kept_layers",
                    "marg_kept_layers",
                ]
            ],
            on=["seq_idx", "pool_name"],
            how="inner",
            validate="one_to_one",
        )
        if len(hard_pair) != len(hard_metadata):
            raise ValueError(f"{spec['variant_id']} scores do not cover the hard-union pool")
        pair_selected = select_lowest(hard_pair, score_column, args.target_rows)
        diagnostics.append(
            write_dataset(
                output_dir=args.output_dir,
                run_id=run_id,
                selected=pair_selected,
                tokens=tokens,
                selection_policy=f"{spec['variant_id']}_direct_topk",
                score_column=score_column,
                source_artifact=pair_path,
                source_artifact_sha256=sha256_file(pair_path),
            )
        )

    metadata_by_seq = metadata[["seq_idx", "pool_name", "c4_index"]]
    for run_id, expected_rate in LCB_SPECS.items():
        summary = validate_dropout_summary(
            provided_dropout[run_id], run_id=run_id, expected_rate=expected_rate, metadata=metadata
        )
        summary_sha256 = sha256_file(provided_dropout[run_id])
        joined = summary[["seq_idx", "pool_name", "score_conservative"]].merge(
            metadata_by_seq, on=["seq_idx", "pool_name"], how="inner", validate="one_to_one"
        )
        hard_summary = joined[joined["pool_name"].isin(HARD_POOLS)].copy()
        if len(hard_summary) != 200_000:
            raise ValueError(f"{run_id}: expected 200K hard-union scores, found {len(hard_summary):,}")
        selected = select_lowest(hard_summary, "score_conservative", args.target_rows)
        diagnostics.append(
            write_dataset(
                output_dir=args.output_dir,
                run_id=run_id,
                selected=selected,
                tokens=tokens,
                selection_policy="embedding_dropout_conservative_lcb",
                score_column="score_conservative",
                source_artifact=provided_dropout[run_id],
                source_artifact_sha256=summary_sha256,
            )
        )

    diagnostics_path = args.output_dir / "selection_diagnostics.csv"
    existing = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    extra = pd.DataFrame(diagnostics)
    if not existing.empty:
        existing = existing[~existing["run_id"].isin(extra["run_id"])]
    pd.concat([existing, extra], ignore_index=True).to_csv(diagnostics_path, index=False)
    print(pd.DataFrame(diagnostics)[["run_id", "true_positive_count", "true_positive_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
