"""Analysis-only recovery workflow for the legacy broad MC-dropout run.

The original compact artifact attached source-ordered metadata and deterministic
scores to shuffled scorer rows. This module repairs only those reference arrays,
reruns offline strategies, writes a corrected report, and builds a verified
handoff bundle. It never mutates the legacy artifact or reruns model inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


PAIRWISE_TASKS = {
    "hp_vs_hn": ("hard_positive", "hard_negative"),
    "hp_vs_rn": ("hard_positive", "random_negative"),
    "hp_vs_tn": ("hard_positive", "tail_negative"),
    "rp_vs_hn": ("random_positive", "hard_negative"),
    "rp_vs_rn": ("random_positive", "random_negative"),
    "rp_vs_tn": ("random_positive", "tail_negative"),
}
EXPECTED_POOLS = set(value for pair in PAIRWISE_TASKS.values() for value in pair)
REQUIRED_ARRAYS = {
    "seq_idx",
    "score_index",
    "pool_name",
    "prior_losses",
    "conditional_losses",
    "color_samples",
    "utility_samples",
    "metadata_json",
    "full_color_score",
}
FIXED_ALIGNMENT_CONTRACTS = {
    "metadata_and_full_scores_indexed_by_score_index",
    "legacy_reference_arrays_indexed_by_score_index",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_permutation(score_index: np.ndarray, rows: int) -> np.ndarray:
    score_index = np.asarray(score_index)
    if score_index.shape != (rows,) or not np.issubdtype(score_index.dtype, np.integer):
        raise ValueError(f"score_index must be an integer vector of shape ({rows},), got {score_index.shape}")
    score_index = score_index.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(score_index), np.arange(rows, dtype=np.int64)):
        raise ValueError(f"score_index is not a complete permutation of 0..{rows - 1}")
    return score_index


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    import pandas as pd

    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def selected_mask(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, int(count)), len(scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argpartition(scores, count - 1)[:count]] = True
    return mask


def uncertainty_error_ratio(
    mean_color: np.ndarray,
    std_color: np.ndarray,
    pool_name: np.ndarray,
) -> Dict[str, float]:
    import pandas as pd

    low_rates = []
    high_rates = []
    for positive_pool, negative_pool in PAIRWISE_TASKS.values():
        task_mask = (pool_name == positive_pool) | (pool_name == negative_pool)
        labels = pool_name[task_mask] == positive_pool
        scores = mean_color[task_mask]
        uncertainty = std_color[task_mask]
        errors = selected_mask(scores, int(labels.sum())) != labels
        deciles = pd.qcut(uncertainty, 10, labels=False, duplicates="drop")
        if pd.isna(deciles).all():
            continue
        deciles = np.asarray(deciles, dtype=np.int64)
        low_rates.append(float(errors[deciles == deciles.min()].mean()))
        high_rates.append(float(errors[deciles == deciles.max()].mean()))
    low = float(np.mean(low_rates)) if low_rates else float("nan")
    high = float(np.mean(high_rates)) if high_rates else float("nan")
    return {
        "low_uncertainty_error_rate": low,
        "high_uncertainty_error_rate": high,
        "error_rate_ratio_high_vs_low": high / low if low > 0 else float("nan"),
    }


def markdown_table(frame, digits: int = 6) -> str:
    display = frame.copy()
    for column in display.columns:
        if np.issubdtype(display[column].dtype, np.floating):
            display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


@dataclass(frozen=True)
class RecoveryContext:
    olmo_dir: Path
    legacy_analysis_dir: Path
    corrected_analysis_dir: Path
    report_dir: Path
    raw_score_root: Path
    metadata_path: Path
    full_scores_path: Path
    producer_sha: str
    analysis_sha: str
    notebook_revision: str
    config_id: str = "dropout_k8_p005"
    num_samples: int = 8
    seed: int = 1
    tau64_cutoff: float = 0.3513622284
    dropout_rate: float = 0.05
    dropout_target: str = "attention+residual+embedding"


class DropoutUncertaintyRecovery:
    EXPECTED_PAIRWISE_ROWS = 96
    EXPECTED_FULL_POOL_ROWS = 64
    EXPECTED_OVERLAP_ROWS = 480
    EXPECTED_SELECTED_FILES = 64

    def __init__(self, context: RecoveryContext):
        self.context = context

    @property
    def legacy_npz(self) -> Path:
        return self.context.legacy_analysis_dir / f"mc_samples_{self.context.config_id}.npz"

    @property
    def legacy_manifest(self) -> Path:
        return self.context.legacy_analysis_dir / f"mc_samples_{self.context.config_id}_manifest.json"

    @property
    def corrected_npz(self) -> Path:
        return self.context.corrected_analysis_dir / f"mc_samples_{self.context.config_id}.npz"

    @property
    def corrected_manifest(self) -> Path:
        return self.context.corrected_analysis_dir / f"mc_samples_{self.context.config_id}_manifest.json"

    @property
    def summary_csv(self) -> Path:
        return self.context.corrected_analysis_dir / "color_distribution_summary.csv"

    @property
    def summary_parquet(self) -> Path:
        return self.context.corrected_analysis_dir / "color_distribution_summary.parquet"

    @property
    def raw_parquet(self) -> Path:
        return self.context.corrected_analysis_dir / f"mc_samples_{self.context.config_id}.parquet"

    @property
    def strategy_dir(self) -> Path:
        return self.context.corrected_analysis_dir / "strategy"

    @property
    def audit_path(self) -> Path:
        return self.context.corrected_analysis_dir / "alignment_audit.json"

    def _validate_scripts(self) -> None:
        metrics_script = self.context.olmo_dir / "scripts/21_dropout_uncertainty_metrics.py"
        strategy_script = self.context.olmo_dir / "scripts/22_dropout_strategy_sweep.py"
        helper_path = self.context.olmo_dir / "scripts/dropout_uncertainty_recovery_colab.py"
        for path in (metrics_script, strategy_script, helper_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        source = metrics_script.read_text(encoding="utf-8")
        if "metadata_and_full_scores_indexed_by_score_index" not in source:
            raise RuntimeError("Pinned analysis revision lacks the score_index alignment fix")

    def validate_legacy_source(self) -> Dict[str, Any]:
        self._validate_scripts()
        if not self.legacy_npz.is_file() or not self.legacy_manifest.is_file():
            raise FileNotFoundError(f"Legacy compact artifacts are incomplete under {self.context.legacy_analysis_dir}")
        arrays = load_npz(self.legacy_npz)
        missing = REQUIRED_ARRAYS.difference(arrays)
        if missing:
            raise ValueError(f"Legacy NPZ is missing arrays: {sorted(missing)}")
        color = arrays["color_samples"]
        rows = len(color)
        if color.shape != (rows, self.context.num_samples):
            raise ValueError(f"Unexpected legacy sample shape: {color.shape}")
        score_index = validate_permutation(arrays["score_index"], rows)
        if np.array_equal(score_index, np.arange(rows, dtype=np.int64)):
            raise ValueError("Legacy score_index is identity; the known shuffled-row defect is not present")
        numeric = (arrays["prior_losses"], arrays["conditional_losses"], color, arrays["full_color_score"])
        if not all(np.isfinite(values).all() for values in numeric):
            raise ValueError("Legacy artifact contains non-finite numeric arrays")
        if not np.allclose(
            color,
            arrays["conditional_losses"] - arrays["prior_losses"],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Legacy color samples do not equal conditional-prior")
        seq_idx = arrays["seq_idx"].astype(np.int64)
        if not np.array_equal(seq_idx, np.arange(rows, dtype=np.int64)):
            raise ValueError("Legacy seq_idx is not source-position ordered; automatic repair would be ambiguous")
        pools, counts = np.unique(arrays["pool_name"].astype(str), return_counts=True)
        if set(pools) != EXPECTED_POOLS or len(set(counts.tolist())) != 1:
            raise ValueError(f"Legacy pool labels do not match the balanced official pool: {dict(zip(pools, counts))}")
        manifest = json.loads(self.legacy_manifest.read_text(encoding="utf-8"))
        if manifest.get("row_alignment") in FIXED_ALIGNMENT_CONTRACTS:
            raise ValueError("Legacy source already declares a fixed alignment contract")
        mean_color = color.mean(axis=1)
        full = arrays["full_color_score"].astype(np.float64)
        raw_spearman = rank_correlation(mean_color, full)
        repaired_spearman = rank_correlation(mean_color, full[score_index])
        if repaired_spearman <= raw_spearman + 0.20:
            raise RuntimeError(
                "Applying score_index does not materially improve alignment; refusing a speculative repair"
            )
        return {
            "rows": rows,
            "num_samples": color.shape[1],
            "score_index_is_permutation": True,
            "score_index_is_identity": False,
            "legacy_npz_sha256": sha256_file(self.legacy_npz),
            "raw_spearman_mean_vs_full": raw_spearman,
            "candidate_repaired_spearman_mean_vs_full": repaired_spearman,
            "pool_counts": dict(zip(pools.tolist(), counts.astype(int).tolist())),
        }

    def preflight(self, mode: str = "repair_compact") -> Dict[str, Any]:
        if mode not in {"repair_compact", "reaggregate_raw"}:
            raise ValueError(f"Unsupported recovery mode: {mode}")
        legacy = self.validate_legacy_source()
        result = {
            "mode": mode,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "legacy": legacy,
        }
        if mode == "reaggregate_raw":
            if not self.context.raw_score_root.is_dir():
                raise FileNotFoundError(self.context.raw_score_root)
            for path in (self.context.metadata_path, self.context.full_scores_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            result["raw_score_dirs"] = {
                role: [str(path) for path in self._discover_score_dirs(role)] for role in ("prior", "books")
            }
        return result

    def _build_summary(self, arrays: Mapping[str, np.ndarray]):
        import pandas as pd

        color = arrays["color_samples"].astype(np.float64)
        mean = color.mean(axis=1)
        std = color.std(axis=1, ddof=1)
        return pd.DataFrame(
            {
                "seq_idx": arrays["seq_idx"].astype(np.int64),
                "score_index": arrays["score_index"].astype(np.int64),
                "pool_name": arrays["pool_name"].astype(str),
                "full_color_score": arrays["full_color_score"].astype(np.float32),
                "mean_color": mean.astype(np.float32),
                "std_color": std.astype(np.float32),
                "q05_color": np.quantile(color, 0.05, axis=1).astype(np.float32),
                "q50_color": np.quantile(color, 0.50, axis=1).astype(np.float32),
                "q95_color": np.quantile(color, 0.95, axis=1).astype(np.float32),
                "score_mean": mean.astype(np.float32),
                "score_conservative": (mean + std).astype(np.float32),
                "score_optimistic": (mean - std).astype(np.float32),
            }
        )

    def _write_parquet_artifacts(self, arrays: Mapping[str, np.ndarray], summary) -> None:
        import pandas as pd

        summary.to_parquet(self.summary_parquet, index=False)
        sample_columns: Dict[str, np.ndarray] = {}
        for prefix, key in (
            ("prior_loss_sample", "prior_losses"),
            ("conditional_loss_sample", "conditional_losses"),
            ("color_sample", "color_samples"),
        ):
            values = arrays[key]
            for sample_index in range(values.shape[1]):
                sample_columns[f"{prefix}_{sample_index:03d}"] = values[:, sample_index]
        raw = pd.concat([summary.reset_index(drop=True), pd.DataFrame(sample_columns)], axis=1)
        raw.to_parquet(self.raw_parquet, index=False)

    def repair_compact(self, write_parquet: bool = True) -> Dict[str, Any]:
        preflight = self.preflight("repair_compact")
        source_sha = preflight["legacy"]["legacy_npz_sha256"]
        if self.audit_path.is_file():
            audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
            if audit.get("legacy_npz_sha256") == source_sha:
                try:
                    self.validate_corrected(require_strategy=False, require_parquet=write_parquet)
                    return {"status": "reused", **audit}
                except (FileNotFoundError, RuntimeError, ValueError):
                    pass

        source = load_npz(self.legacy_npz)
        rows = len(source["color_samples"])
        score_index = validate_permutation(source["score_index"], rows)
        corrected = dict(source)
        corrected["seq_idx"] = source["seq_idx"].astype(np.int64)[score_index]
        corrected["pool_name"] = source["pool_name"].astype(str)[score_index]
        corrected["full_color_score"] = source["full_color_score"].astype(np.float32)[score_index]
        metadata = json.loads(str(source["metadata_json"].item()))
        metadata.update(
            {
                "row_alignment": "legacy_reference_arrays_indexed_by_score_index",
                "selection_id_source": "source_seq_idx_indexed_by_score_index",
                "alignment_repaired": True,
                "analysis_sha": self.context.analysis_sha,
                "notebook_revision": self.context.notebook_revision,
            }
        )
        metadata_text = json.dumps(metadata, sort_keys=True)
        corrected["metadata_json"] = np.asarray(metadata_text, dtype=f"U{len(metadata_text)}")

        self.context.corrected_analysis_dir.mkdir(parents=True, exist_ok=True)
        save_npz_atomic(self.corrected_npz, corrected)
        summary = self._build_summary(corrected)
        summary.to_csv(self.summary_csv, index=False)
        if write_parquet:
            self._write_parquet_artifacts(corrected, summary)

        mean_color = corrected["color_samples"].mean(axis=1)
        raw_full = source["full_color_score"].astype(np.float64)
        fixed_full = corrected["full_color_score"].astype(np.float64)
        selection_count = max(1, round(rows / 64))
        selected = selected_mask(mean_color, selection_count)
        raw_reference = selected_mask(raw_full, selection_count)
        fixed_reference = selected_mask(fixed_full, selection_count)
        audit = {
            "created_utc": utc_now(),
            "config_id": self.context.config_id,
            "rows": rows,
            "num_samples": int(corrected["color_samples"].shape[1]),
            "legacy_npz": str(self.legacy_npz),
            "legacy_npz_sha256": source_sha,
            "alignment_method": "legacy_reference_arrays_indexed_by_score_index",
            "stochastic_sample_arrays_changed": False,
            "raw_pearson_mean_vs_full": float(np.corrcoef(mean_color, raw_full)[0, 1]),
            "raw_spearman_mean_vs_full": rank_correlation(mean_color, raw_full),
            "corrected_pearson_mean_vs_full": float(np.corrcoef(mean_color, fixed_full)[0, 1]),
            "corrected_spearman_mean_vs_full": rank_correlation(mean_color, fixed_full),
            "raw_recall_vs_full_1_64": float((selected & raw_reference).sum() / raw_reference.sum()),
            "corrected_recall_vs_full_1_64": float((selected & fixed_reference).sum() / fixed_reference.sum()),
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
        }
        atomic_write_text(self.audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

        source_manifest = json.loads(self.legacy_manifest.read_text(encoding="utf-8"))
        manifest = {
            **source_manifest,
            "created_utc": utc_now(),
            "npz_path": str(self.corrected_npz),
            "parquet_path": str(self.raw_parquet) if write_parquet else None,
            "summary_csv_path": str(self.summary_csv),
            "summary_parquet_path": str(self.summary_parquet) if write_parquet else None,
            "row_alignment": "legacy_reference_arrays_indexed_by_score_index",
            "selection_id_source": "source_seq_idx_indexed_by_score_index",
            "alignment_repaired": True,
            "legacy_npz": str(self.legacy_npz),
            "legacy_npz_sha256": source_sha,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
        }
        atomic_write_text(self.corrected_manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.validate_corrected(require_strategy=False, require_parquet=write_parquet)
        return {"status": "repaired", **audit}

    def _discover_score_dirs(self, role: str) -> Sequence[Path]:
        role_root = self.context.raw_score_root / role
        if not role_root.is_dir():
            raise FileNotFoundError(role_root)
        directories = sorted({path.parent for path in role_root.rglob("files.txt")})
        if not directories:
            raise FileNotFoundError(f"No score directories found under {role_root}")
        return directories

    def _run_logged(self, command: Sequence[object], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(item) for item in command],
                cwd=str(self.context.olmo_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def reaggregate_raw(self) -> Dict[str, Any]:
        preflight = self.preflight("reaggregate_raw")
        self.context.corrected_analysis_dir.mkdir(parents=True, exist_ok=True)
        prior_dirs = self._discover_score_dirs("prior")
        books_dirs = self._discover_score_dirs("books")
        command: list[object] = [
            sys.executable,
            self.context.olmo_dir / "scripts/21_dropout_uncertainty_metrics.py",
            "--prior-score-dir",
            *prior_dirs,
            "--conditional-score-dir",
            *books_dirs,
            "--output-dir",
            self.context.corrected_analysis_dir,
            "--config-id",
            self.context.config_id,
            "--metadata",
            self.context.metadata_path,
            "--full-scores",
            self.context.full_scores_path,
            "--num-samples",
            self.context.num_samples,
            "--dropout-rate",
            self.context.dropout_rate,
            "--dropout-target",
            self.context.dropout_target,
            "--attention-dropout",
            self.context.dropout_rate,
            "--residual-dropout",
            self.context.dropout_rate,
            "--embedding-dropout",
            self.context.dropout_rate,
            "--seed",
            self.context.seed,
            "--summary-format",
            "both",
            "--compress-npz",
        ]
        self._run_logged(command, self.context.corrected_analysis_dir / "aggregate.log")
        manifest = json.loads(self.corrected_manifest.read_text(encoding="utf-8"))
        manifest.update(
            {
                "producer_sha": self.context.producer_sha,
                "analysis_sha": self.context.analysis_sha,
                "notebook_revision": self.context.notebook_revision,
                "legacy_comparison_npz": str(self.legacy_npz),
                "legacy_comparison_npz_sha256": preflight["legacy"]["legacy_npz_sha256"],
            }
        )
        atomic_write_text(self.corrected_manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.validate_corrected(require_strategy=False, require_parquet=True)
        return {"status": "reaggregated", "prior_dirs": len(prior_dirs), "books_dirs": len(books_dirs)}

    def run_strategy_sweep(self) -> Dict[str, Any]:
        self.validate_corrected(require_strategy=False, require_parquet=False)
        try:
            return {"status": "reused", **self.validate_strategy_outputs()}
        except (FileNotFoundError, RuntimeError, ValueError):
            if self.strategy_dir.exists():
                shutil.rmtree(self.strategy_dir)
        command = [
            sys.executable,
            self.context.olmo_dir / "scripts/22_dropout_strategy_sweep.py",
            "--mc-samples",
            self.corrected_npz,
            "--summary",
            self.summary_csv,
            "--output-dir",
            self.strategy_dir,
            "--tau64-cutoff",
            self.context.tau64_cutoff,
        ]
        self._run_logged(command, self.context.corrected_analysis_dir / "strategy.log")
        return {"status": "generated", **self.validate_strategy_outputs()}

    def validate_corrected(self, require_strategy: bool = True, require_parquet: bool = True) -> Dict[str, Any]:
        import pandas as pd

        required = [self.corrected_npz, self.corrected_manifest, self.summary_csv]
        if require_parquet:
            required.extend([self.summary_parquet, self.raw_parquet])
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise FileNotFoundError("Missing corrected artifacts: " + ", ".join(missing))
        manifest = json.loads(self.corrected_manifest.read_text(encoding="utf-8"))
        if manifest.get("row_alignment") not in FIXED_ALIGNMENT_CONTRACTS:
            raise ValueError(f"Corrected manifest lacks a fixed alignment contract: {manifest.get('row_alignment')}")
        arrays = load_npz(self.corrected_npz)
        missing_arrays = REQUIRED_ARRAYS.difference(arrays)
        if missing_arrays:
            raise ValueError(f"Corrected NPZ is missing arrays: {sorted(missing_arrays)}")
        color = arrays["color_samples"]
        rows = len(color)
        if color.shape != (rows, self.context.num_samples):
            raise ValueError(f"Corrected sample shape is invalid: {color.shape}")
        score_index = validate_permutation(arrays["score_index"], rows)
        seq_idx = arrays["seq_idx"].astype(np.int64)
        if not np.array_equal(seq_idx, score_index):
            raise ValueError("Corrected selected IDs do not follow score_index for the official identity metadata")
        if not np.allclose(
            color,
            arrays["conditional_losses"] - arrays["prior_losses"],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Corrected artifact changed the conditional-prior score contract")
        if not np.allclose(color, -arrays["utility_samples"], rtol=1e-5, atol=1e-6):
            raise ValueError("Corrected utility samples are not negative color samples")
        if not all(
            np.isfinite(values).all()
            for values in (arrays["prior_losses"], arrays["conditional_losses"], color, arrays["full_color_score"])
        ):
            raise ValueError("Corrected artifact contains non-finite arrays")
        pool_name = arrays["pool_name"].astype(str)
        if set(np.unique(pool_name)) != EXPECTED_POOLS:
            raise ValueError("Corrected pool labels do not cover the six pairwise tasks")
        mean = color.mean(axis=1)
        spearman = rank_correlation(mean, arrays["full_color_score"])
        if spearman <= 0.50:
            raise RuntimeError(f"Corrected alignment still fails the Spearman gate: {spearman:.6f}")
        summary = pd.read_csv(self.summary_csv)
        required_columns = {"seq_idx", "score_index", "pool_name", "full_color_score", "mean_color", "std_color"}
        if len(summary) != rows or not required_columns.issubset(summary.columns):
            raise ValueError("Corrected summary has an invalid row count or schema")
        if not np.array_equal(summary["score_index"].to_numpy(dtype=np.int64), score_index):
            raise ValueError("Corrected summary score_index differs from the NPZ")
        result = {"rows": rows, "num_samples": color.shape[1], "spearman_mean_vs_full": spearman}
        if require_strategy:
            result.update(self.validate_strategy_outputs())
        return result

    def validate_strategy_outputs(self) -> Dict[str, Any]:
        import pandas as pd

        metrics_path = self.strategy_dir / "strategy_sweep_metrics.csv"
        overlap_path = self.strategy_dir / "strategy_selection_overlap.csv"
        for path in (metrics_path, overlap_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
        metrics = pd.read_csv(metrics_path)
        pairwise = metrics[metrics["metric_scope"] == "pairwise"]
        full_pool = metrics[metrics["metric_scope"] == "full_pool"]
        if len(pairwise) != self.EXPECTED_PAIRWISE_ROWS or len(full_pool) != self.EXPECTED_FULL_POOL_ROWS:
            raise ValueError(f"Unexpected strategy rows: pairwise={len(pairwise)}, full_pool={len(full_pool)}")
        if metrics["strategy"].nunique() != 16 or set(pairwise["task_id"]) != set(PAIRWISE_TASKS):
            raise ValueError("Strategy output lacks the complete task/strategy grid")
        if not np.isfinite(pairwise["roc_auc"]).all() or not np.isfinite(full_pool["recall_vs_full"]).all():
            raise ValueError("Strategy output contains non-finite decision metrics")
        overlap = pd.read_csv(overlap_path)
        if len(overlap) != self.EXPECTED_OVERLAP_ROWS or not np.isfinite(overlap["jaccard"]).all():
            raise ValueError("Strategy overlap output is incomplete")
        selected = sorted((self.strategy_dir / "strategy_selected_indices").glob("*.npy"))
        if len(selected) != self.EXPECTED_SELECTED_FILES:
            raise ValueError(f"Expected {self.EXPECTED_SELECTED_FILES} selected arrays, found {len(selected)}")
        valid_ids = set(load_npz(self.corrected_npz)["seq_idx"].astype(np.int64).tolist())
        for path in selected:
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or len(values) != len(np.unique(values)):
                raise ValueError(f"Selected IDs are invalid: {path}")
            if not set(values.astype(np.int64).tolist()).issubset(valid_ids):
                raise ValueError(f"Selected IDs escape the corrected source-row set: {path}")
        mean_pair = pairwise[pairwise["strategy"] == "mean"]
        strict_mean = full_pool[
            (full_pool["strategy"] == "mean") & np.isclose(full_pool["selection_rate"], 1 / 64)
        ]
        if len(mean_pair) != 6 or len(strict_mean) != 1:
            raise ValueError("Required mean-strategy rows are missing")
        return {
            "mean_pairwise_auc": float(mean_pair["roc_auc"].mean()),
            "hp_vs_hn_auc": float(mean_pair.loc[mean_pair["task_id"] == "hp_vs_hn", "roc_auc"].iloc[0]),
            "recall_vs_full_1_64": float(strict_mean["recall_vs_full"].iloc[0]),
        }

    def build_report(self) -> Dict[str, Any]:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        validation = self.validate_corrected(require_strategy=True, require_parquet=False)
        arrays = load_npz(self.corrected_npz)
        color = arrays["color_samples"].astype(np.float64)
        mean = color.mean(axis=1)
        std = color.std(axis=1, ddof=1)
        full = arrays["full_color_score"].astype(np.float64)
        pools = arrays["pool_name"].astype(str)
        metrics = pd.read_csv(self.strategy_dir / "strategy_sweep_metrics.csv")
        pair_mean = metrics[(metrics["metric_scope"] == "pairwise") & (metrics["strategy"] == "mean")].copy()
        strict = metrics[(metrics["metric_scope"] == "full_pool") & np.isclose(metrics["selection_rate"], 1 / 64)].copy()
        strict = strict.sort_values("recall_vs_full", ascending=False)

        q05 = np.quantile(color, 0.05, axis=1)
        q95 = np.quantile(color, 0.95, axis=1)
        full_positive = full <= self.context.tau64_cutoff
        labels = np.full(len(mean), "uncertain", dtype=object)
        labels[q95 <= self.context.tau64_cutoff] = "confident_positive"
        labels[q05 > self.context.tau64_cutoff] = "confident_negative"
        triage_rows = []
        for band in ("confident_positive", "uncertain", "confident_negative"):
            mask = labels == band
            triage_rows.append(
                {
                    "band": band,
                    "rows": int(mask.sum()),
                    "fraction": float(mask.mean()),
                    "full_positive_rate": float(full_positive[mask].mean()),
                }
            )
        triage = pd.DataFrame(triage_rows)
        uncertainty = uncertainty_error_ratio(mean, std, pools)
        audit = json.loads(self.audit_path.read_text(encoding="utf-8")) if self.audit_path.exists() else {}
        summary = {
            **validation,
            **uncertainty,
            "pearson_mean_vs_full": float(np.corrcoef(mean, full)[0, 1]),
            "mean_mc_std": float(std.mean()),
            "best_strategy_1_64": str(strict.iloc[0]["strategy"]),
            "best_recall_vs_full_1_64": float(strict.iloc[0]["recall_vs_full"]),
            "triage_uncertain_fraction": float(triage.loc[triage["band"] == "uncertain", "fraction"].iloc[0]),
        }

        self.context.report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(self.context.report_dir / "corrected_summary.csv", index=False)
        pair_mean[["task_id", "roc_auc", "average_precision", "balanced_f1", "tau64_f1"]].to_csv(
            self.context.report_dir / "pairwise_mean.csv", index=False
        )
        triage.to_csv(self.context.report_dir / "triage.csv", index=False)

        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        alignment_labels = ["Pearson", "Spearman", "Recall@1/64"]
        raw_values = [
            audit.get("raw_pearson_mean_vs_full", np.nan),
            audit.get("raw_spearman_mean_vs_full", np.nan),
            audit.get("raw_recall_vs_full_1_64", np.nan),
        ]
        fixed_values = [summary["pearson_mean_vs_full"], summary["spearman_mean_vs_full"], summary["recall_vs_full_1_64"]]
        x = np.arange(len(alignment_labels))
        axes[0].bar(x - 0.18, raw_values, width=0.36, label="misaligned")
        axes[0].bar(x + 0.18, fixed_values, width=0.36, label="corrected")
        axes[0].set_xticks(x, alignment_labels, rotation=20)
        axes[0].set_ylim(min(-0.05, np.nanmin(raw_values) - 0.03), 1.05)
        axes[0].set_title("Alignment audit")
        axes[0].legend()
        axes[1].bar(pair_mean["task_id"], pair_mean["roc_auc"], color="#3b6ea8")
        axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_ylim(0.45, 1.02)
        axes[1].set_title("Corrected mean-strategy AUC")
        axes[2].bar(triage["band"], triage["full_positive_rate"], color="#c05a47")
        axes[2].tick_params(axis="x", rotation=25)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Deterministic-positive rate by triage band")
        figure.tight_layout()
        figure_path = self.context.report_dir / "alignment_corrected_results.png"
        figure.savefig(figure_path, dpi=170)
        plt.close(figure)

        report = f"""# Broad MC-Dropout Alignment-Corrected Report

## Executive Summary

The original near-chance result was caused by positional attachment of source-ordered
metadata and deterministic scores to shuffled scorer rows. Reindexing only those
reference arrays by `score_index` changes Spearman from
{audit.get('raw_spearman_mean_vs_full', float('nan')):.4f} to
{summary['spearman_mean_vs_full']:.4f}, mean pairwise AUC to
{summary['mean_pairwise_auc']:.4f}, and mean-strategy recall@1/64 from
{audit.get('raw_recall_vs_full_1_64', float('nan')):.4f} to
{summary['recall_vs_full_1_64']:.4f}. The stochastic sample tensors were not changed.

Broad attention+residual+embedding dropout at `p=0.05` therefore preserves substantial
global CoLoR signal; it does not collapse to chance. It remains weaker than the later
low-rate targeted configurations, especially on `hp_vs_hn`.

## Corrected Metrics

{markdown_table(pd.DataFrame([summary]))}

## Pairwise Mean Strategy

{markdown_table(pair_mean[['task_id', 'roc_auc', 'average_precision', 'balanced_f1', 'tau64_f1']])}

## Triage

{markdown_table(triage)}

High-uncertainty rows have {summary['error_rate_ratio_high_vs_low']:.3f}x the error rate
of low-uncertainty rows. This supports an offline diagnostic/triage association, not a
calibrated posterior or an end-to-end compute-saving policy.

## Use-Case Conclusions

| Use case | Verdict | Evidence |
| --- | --- | --- |
| standalone selection | signal supported, not preferred | Spearman {summary['spearman_mean_vs_full']:.3f}; recall@1/64 {summary['recall_vs_full_1_64']:.3f}; K=8 requires repeated full-model scoring. |
| cascade candidate generation | not recommended at the strict budget | Same-budget recall@1/64 is {summary['recall_vs_full_1_64']:.3f}, leaving material misses. |
| triage routing | supported offline | The triage bands separate deterministic-positive prevalence, but total rescore cost was not benchmarked. |
| uncertainty diagnostics | modestly supported | High/low error enrichment is {summary['error_rate_ratio_high_vs_low']:.3f}x. |

![Alignment-corrected results](alignment_corrected_results.png)

## Limitations

- One seed and `K=8` were evaluated.
- Dropout scope and rate remain confounded.
- The fixed report repairs legacy reference alignment; it does not rerun model scoring.
- Runtime comparisons and end-to-end routing savings are not re-estimated here.

## Reproducibility

- Producer revision: `{self.context.producer_sha}`
- Analysis revision: `{self.context.analysis_sha}`
- Notebook revision: `{self.context.notebook_revision}`
- Legacy NPZ SHA-256: `{audit.get('legacy_npz_sha256', 'not-recorded')}`
- Alignment contract: `legacy_reference_arrays_indexed_by_score_index`
"""
        report_md = self.context.report_dir / "report.md"
        atomic_write_text(report_md, report)
        report_html = self.context.report_dir / "report.html"
        report_html.write_text(
            '<!doctype html><meta charset="utf-8"><title>Broad MC-dropout corrected report</title>'
            '<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}'
            'table{border-collapse:collapse;width:100%;font-size:12px}'
            'th,td{border:1px solid #ccc;padding:5px}img{max-width:100%;height:auto}</style>'
            '<h1>Broad MC-Dropout Alignment-Corrected Report</h1>'
            f'<p>Spearman: {summary["spearman_mean_vs_full"]:.4f}; mean pairwise AUC: '
            f'{summary["mean_pairwise_auc"]:.4f}; recall@1/64: {summary["recall_vs_full_1_64"]:.4f}.</p>'
            '<h2>Corrected metrics</h2>'
            + pd.DataFrame([summary]).to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + '<h2>Pairwise mean strategy</h2>'
            + pair_mean.to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + '<h2>Triage</h2>'
            + triage.to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + '<img src="alignment_corrected_results.png" alt="Alignment-corrected results">',
            encoding="utf-8",
        )
        report_manifest = {
            "created_utc": utc_now(),
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "alignment_contract": "legacy_reference_arrays_indexed_by_score_index",
            "artifacts": [
                report_md.name,
                report_html.name,
                figure_path.name,
                "corrected_summary.csv",
                "pairwise_mean.csv",
                "triage.csv",
            ],
        }
        atomic_write_text(
            self.context.report_dir / "report_manifest.json",
            json.dumps(report_manifest, indent=2, sort_keys=True) + "\n",
        )
        return {"summary": summary, "triage": triage, "pairwise": pair_mean, "report": report_md}

    def run(self, mode: str = "repair_compact", write_parquet: bool = True) -> Dict[str, Any]:
        recovery = self.repair_compact(write_parquet=write_parquet) if mode == "repair_compact" else self.reaggregate_raw()
        strategy = self.run_strategy_sweep()
        report = self.build_report()
        return {"recovery": recovery, "strategy": strategy, "summary": report["summary"]}

    def status(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "legacy_npz": self.legacy_npz.is_file(),
            "corrected_npz": self.corrected_npz.is_file(),
            "corrected_manifest": self.corrected_manifest.is_file(),
            "strategy_metrics": (self.strategy_dir / "strategy_sweep_metrics.csv").is_file(),
            "report": (self.context.report_dir / "report.md").is_file(),
        }
        try:
            result["validation"] = self.validate_corrected(require_strategy=True, require_parquet=False)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def build_bundle(self, archive_path: Path, drive_archive_path: Path) -> Dict[str, Any]:
        from scripts.targeted_dropout_colab_helpers import create_verified_archive

        validation = self.validate_corrected(require_strategy=True, require_parquet=False)
        report_result = self.build_report()
        files: list[tuple[Path, str]] = []

        def add(path: Path, archive_name: str) -> None:
            files.append((Path(path), archive_name))

        for path in (
            self.corrected_npz,
            self.corrected_manifest,
            self.summary_csv,
            self.audit_path,
            self.context.corrected_analysis_dir / "strategy.log",
        ):
            add(path, f"analysis/{path.name}")
        for optional in (self.summary_parquet, self.raw_parquet, self.context.corrected_analysis_dir / "aggregate.log"):
            if optional.is_file():
                add(optional, f"analysis/{optional.name}")
        for name in ("strategy_sweep_metrics.csv", "strategy_selection_overlap.csv"):
            add(self.strategy_dir / name, f"analysis/strategy/{name}")
        for selected in sorted((self.strategy_dir / "strategy_selected_indices").glob("*.npy")):
            add(selected, f"analysis/strategy/strategy_selected_indices/{selected.name}")
        for name in (
            "report.md",
            "report.html",
            "alignment_corrected_results.png",
            "corrected_summary.csv",
            "pairwise_mean.csv",
            "triage.csv",
            "report_manifest.json",
        ):
            add(self.context.report_dir / name, f"report/{name}")
        add(self.legacy_manifest, "legacy_source/mc_samples_manifest.json")

        manifest = {
            "created_utc": utc_now(),
            "config_id": self.context.config_id,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "alignment_contract": "legacy_reference_arrays_indexed_by_score_index",
            "validation": validation,
            "report": str(report_result["report"]),
            "files": [
                {
                    "source": str(source),
                    "archive": archive_name,
                    "bytes": source.stat().st_size if source.exists() else None,
                }
                for source, archive_name in files
            ],
            "excludes": [
                "model checkpoints",
                "token arrays",
                "raw scorer memmaps",
                "duplicate legacy MC sample NPZ",
            ],
            "drive_fallback": str(drive_archive_path),
        }
        local, drive = create_verified_archive(files, manifest, archive_path, drive_archive_path)
        return {
            "archive_path": local,
            "drive_archive_path": drive,
            "file_count": len(files) + 1,
            "manifest": manifest,
        }


__all__ = [
    "DropoutUncertaintyRecovery",
    "RecoveryContext",
    "FIXED_ALIGNMENT_CONTRACTS",
    "REQUIRED_ARRAYS",
    "rank_correlation",
    "sha256_file",
    "validate_permutation",
]
