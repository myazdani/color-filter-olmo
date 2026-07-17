"""Colab orchestration for the full-pool broad MC-dropout rate sweep."""

from __future__ import annotations

import html
import json
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from scripts.targeted_dropout_colab_helpers import (
    TargetedDropoutRunner,
    build_shard_plan,
    canonical_sha256,
    create_verified_archive,
    select_fastest_benchmark,
    validate_runtime_records,
    verify_bundle_archive,
)


REQUESTED_RATES = (0.0, 0.00001, 0.001, 0.01)


def validate_sweep_configs(configs: Sequence[Mapping[str, Any]]) -> None:
    if len(configs) != len(REQUESTED_RATES):
        raise ValueError(f"Expected {len(REQUESTED_RATES)} broad-dropout configs, found {len(configs)}")
    ids = [str(config["config_id"]) for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Config IDs must be unique: {ids}")
    rates = []
    for config in configs:
        rate = float(config["dropout_rate"])
        rates.append(rate)
        if config.get("dropout_target") != "attention+residual+embedding":
            raise ValueError(f"{config['config_id']}: expected broad dropout target")
        for key in ("attention_dropout", "residual_dropout", "embedding_dropout"):
            if float(config[key]) != rate:
                raise ValueError(f"{config['config_id']}: {key} must equal dropout_rate")
        samples = int(config["num_samples"])
        expected_samples = 1 if rate == 0.0 else 8
        if samples != expected_samples:
            raise ValueError(f"{config['config_id']}: p={rate:g} requires K={expected_samples}, found K={samples}")
    if not np.allclose(sorted(rates), REQUESTED_RATES, rtol=0.0, atol=1e-12):
        raise ValueError(f"Unexpected sweep rates: {sorted(rates)}")


def finite_sample_std(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] == 1:
        return np.zeros(samples.shape[0], dtype=np.float32)
    return samples.std(axis=1, ddof=1).astype(np.float32)


def markdown_table(frame, digits: int = 6) -> str:
    display = frame.copy()
    for column in display.columns:
        if np.issubdtype(display[column].dtype, np.floating):
            display[column] = display[column].map(lambda value: f"{value:.{digits}g}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class BroadSweepContext:
    runner: TargetedDropoutRunner
    analysis_olmo_dir: Path
    configs: Sequence[Mapping[str, Any]]
    shards: Sequence[Mapping[str, int]]
    prior_checkpoint: Path
    books_checkpoint: Path
    metadata_path: Path
    full_scores_path: Path
    stage_root: Path
    report_dir: Path
    config_drive: Path
    run_state_path: Path
    shard_plan_path: Path
    subset_manifest_path: Path
    producer_sha: str
    analysis_sha: str
    notebook_revision: str
    rows: int
    seq_len: int
    tau64_cutoff: float
    global_batch_size: int
    shard_rows: int
    microbatch: int
    reference_p005_analysis: Optional[Path] = None


class BroadDropoutSweep:
    EXPECTED_PAIRWISE_ROWS = 96
    EXPECTED_FULL_POOL_ROWS = 64

    def __init__(self, context: BroadSweepContext):
        self.context = context
        validate_sweep_configs(context.configs)

    @property
    def runner(self) -> TargetedDropoutRunner:
        return self.context.runner

    def config_by_rate(self, rate: float) -> Mapping[str, Any]:
        return next(config for config in self.context.configs if float(config["dropout_rate"]) == rate)

    def analysis_dir(self, config: Mapping[str, Any]) -> Path:
        return self.context.stage_root / str(config["config_id"]) / "analysis"

    def score_dirs(self, config: Mapping[str, Any], model_id: str) -> Sequence[Path]:
        return [
            self.runner.shard_output_dir(str(config["config_id"]), model_id, shard) / "score"
            for shard in self.context.shards
        ]

    @property
    def analysis_script_dir(self) -> Path:
        return self.context.analysis_olmo_dir / "scripts"

    def run_analysis_command(self, args: Sequence[Any], log_path: Path, label: str) -> float:
        return self.runner.run_logged(
            args,
            log_path,
            cwd=self.context.analysis_olmo_dir,
            pythonpath_root=self.context.analysis_olmo_dir,
            label=label,
        )

    def run_zero_dropout_smoke(self, smoke_rows: int) -> Dict[str, float]:
        import pandas as pd

        config = self.config_by_rate(0.0)
        shard = {"start": 0, "end": smoke_rows, "rows": smoke_rows, "data_start_step": 0}
        root = self.context.stage_root / "_smoke_and_benchmark" / "smoke" / str(config["config_id"])
        score_dirs = {}
        for model_id, checkpoint in (
            ("prior", self.context.prior_checkpoint),
            ("books", self.context.books_checkpoint),
        ):
            score_dirs[model_id] = self.runner.run_score_once(
                config,
                model_id,
                checkpoint,
                root / model_id,
                shard,
                self.context.microbatch,
                console_log_interval=1,
            )
        analysis = root / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        self.run_analysis_command(
            [
                sys.executable,
                self.analysis_script_dir / "21_dropout_uncertainty_metrics.py",
                "--prior-score-dir",
                score_dirs["prior"],
                "--conditional-score-dir",
                score_dirs["books"],
                "--output-dir",
                analysis,
                "--config-id",
                config["config_id"],
                "--metadata",
                self.context.metadata_path,
                "--full-scores",
                self.context.full_scores_path,
                "--num-samples",
                config["num_samples"],
                "--max-rows",
                smoke_rows,
                "--dropout-rate",
                config["dropout_rate"],
                "--dropout-target",
                config["dropout_target"],
                "--attention-dropout",
                config["attention_dropout"],
                "--residual-dropout",
                config["residual_dropout"],
                "--embedding-dropout",
                config["embedding_dropout"],
                "--seed",
                self.runner.context.seed,
                "--skip-parquet",
                "--compress-npz",
            ],
            analysis / "aggregate.log",
            "zero-dropout smoke aggregation",
        )
        with np.load(analysis / f"mc_samples_{config['config_id']}.npz", allow_pickle=False) as raw:
            color = raw["color_samples"]
            full = raw["full_color_score"]
        if color.shape != (smoke_rows, 1) or not np.isfinite(color).all():
            raise RuntimeError(f"Zero-dropout smoke output is invalid: {color.shape}")
        spearman = float(pd.Series(color[:, 0]).rank().corr(pd.Series(full).rank()))
        pearson = float(np.corrcoef(color[:, 0], full)[0, 1])
        if spearman < 0.90 or pearson < 0.99:
            raise RuntimeError(f"Zero-dropout smoke gate failed: Spearman={spearman:.6f}, Pearson={pearson:.6f}")
        return {"rows": smoke_rows, "spearman": spearman, "pearson": pearson, "mean_mc_std": 0.0}

    def configure_batch(self, global_batch_size: int, microbatch: int) -> None:
        shard_rows = (self.context.shard_rows // global_batch_size) * global_batch_size
        shards = build_shard_plan(self.context.rows, shard_rows, global_batch_size)
        runner = TargetedDropoutRunner(replace(
            self.runner.context, global_batch_size=global_batch_size, shard_rows=shard_rows
        ))
        self.context = replace(
            self.context, runner=runner, shards=shards, global_batch_size=global_batch_size,
            shard_rows=shard_rows, microbatch=microbatch,
        )

    def benchmark(self, benchmark_rows: int, candidates: Sequence[Sequence[int]]) -> Dict[str, Any]:
        config = self.config_by_rate(max(REQUESTED_RATES))
        root = self.context.stage_root / "_smoke_and_benchmark" / "benchmark" / str(config["config_id"])
        results = []
        for global_batch_size, microbatch in candidates:
            global_batch_size, microbatch = int(global_batch_size), int(microbatch)
            if benchmark_rows % global_batch_size or microbatch > global_batch_size:
                raise ValueError(f"Invalid benchmark tuple: {(global_batch_size, microbatch)}")
            shard = {"start": 0, "end": benchmark_rows, "rows": benchmark_rows,
                     "data_start_step": 0, "batch_size": global_batch_size}
            output = root / f"prior_batch_{global_batch_size}_microbatch_{microbatch}"
            try:
                score = self.runner.run_score_once(config, "prior", self.context.prior_checkpoint, output, shard, microbatch, 1)
                marker = json.loads((output / "completed.json").read_text())
                throughput = float(marker["runtime_metrics"]["tokens_per_second"])
                results.append({"global_batch_size": global_batch_size, "microbatch": microbatch,
                                "tokens_per_second": throughput, "elapsed_seconds": marker["elapsed_seconds"],
                                "rows": benchmark_rows, "tokens": benchmark_rows * self.context.seq_len,
                                "peak_gpu_memory_mb": marker["runtime_metrics"].get("peak_gpu_memory_mb"),
                                "status": "ok", "output": str(score)})
            except Exception as exc:
                results.append({"global_batch_size": global_batch_size, "microbatch": microbatch,
                                "status": f"failed: {exc}"})
                print("stopping after first failed batch tuple", flush=True)
                break
        selected = select_fastest_benchmark(results)
        selected_batch = int(selected["global_batch_size"])
        selected_microbatch = int(selected["microbatch"])
        self.configure_batch(selected_batch, selected_microbatch)
        benchmark_tps = float(selected["tokens_per_second"])
        jobs = len(self.context.configs) * 2 * len(self.context.shards)
        conservative_tokens = len(self.context.configs) * 2 * self.context.rows * self.context.seq_len
        estimated_seconds = conservative_tokens / benchmark_tps
        state = {
            "schema_version": 4,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "subset_fingerprint": self.runner.context.subset_fingerprint,
            "config_fingerprint": canonical_sha256({"configs": [dict(item) for item in self.context.configs]}),
            "microbatch": selected_microbatch,
            "global_batch_size": selected_batch,
            "shard_rows": self.context.shard_rows,
            "benchmark": selected,
            "benchmark_results": results,
            "conservative_eta_seconds": estimated_seconds,
            "production_jobs": jobs,
        }
        self.context.run_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        print(
            f"Conservative production estimate: {estimated_seconds / 3600:.2f} GPU hours across {jobs} jobs "
            f"using (configs * models * rows * seq_len) / benchmark_tokens_per_second; "
            "the p=0 K=1 control should finish faster than this K=8-based estimate.",
            flush=True,
        )
        return state

    def load_microbatch(self) -> int:
        if not self.context.run_state_path.is_file():
            raise FileNotFoundError(self.context.run_state_path)
        state = json.loads(self.context.run_state_path.read_text())
        expected = {
            "schema_version": 4,
            "producer_sha": self.context.producer_sha,
            "subset_fingerprint": self.runner.context.subset_fingerprint,
            "config_fingerprint": canonical_sha256({"configs": [dict(item) for item in self.context.configs]}),
            "shard_rows": self.context.shard_rows,
            "global_batch_size": self.context.global_batch_size,
        }
        mismatches = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Persisted run state does not match the sweep: {mismatches}")
        self.configure_batch(int(state["global_batch_size"]), int(state["microbatch"]))
        return int(state["microbatch"])

    def raw_status(self, microbatch: Optional[int] = None) -> Sequence[Dict[str, Any]]:
        selected_microbatch = int(microbatch or self.load_microbatch())
        rows = []
        for config in self.context.configs:
            for model_id in ("prior", "books"):
                for shard in self.context.shards:
                    output = self.runner.shard_output_dir(config["config_id"], model_id, shard)
                    rows.append(
                        {
                            "config_id": config["config_id"],
                            "dropout_rate": config["dropout_rate"],
                            "num_samples": config["num_samples"],
                            "model_id": model_id,
                            "start": int(shard["start"]),
                            "end": int(shard["end"]),
                            "valid": self.runner.valid_score_output(
                                output, config, model_id, shard, selected_microbatch
                            ),
                            "output": str(output),
                        }
                    )
        return rows

    def run_production(self) -> None:
        microbatch = self.load_microbatch()
        jobs = [
            (config, model_id, checkpoint, shard)
            for config in self.context.configs
            for model_id, checkpoint in (
                ("prior", self.context.prior_checkpoint),
                ("books", self.context.books_checkpoint),
            )
            for shard in self.context.shards
        ]
        initial = self.raw_status(microbatch)
        complete = sum(int(row["valid"]) for row in initial)
        state = json.loads(self.context.run_state_path.read_text())
        eta_seconds = float(state.get("conservative_eta_seconds", 0.0)) * (len(jobs) - complete) / len(jobs)
        print(
            f"Starting/resuming broad sweep: {complete}/{len(jobs)} jobs already valid; "
            f"conservative remaining ETA {eta_seconds / 3600:.2f} hours.",
            flush=True,
        )
        started = time.perf_counter()
        completed_this_call = 0
        pending_at_start = len(jobs) - complete
        for position, (config, model_id, checkpoint, shard) in enumerate(jobs, start=1):
            output = self.runner.shard_output_dir(config["config_id"], model_id, shard)
            valid_before = self.runner.valid_score_output(output, config, model_id, shard, microbatch)
            print(
                f"[{position}/{len(jobs)}] config={config['config_id']} p={config['dropout_rate']:g} "
                f"K={config['num_samples']} model={model_id} rows={shard['start']}:{shard['end']} "
                f"status={'skip' if valid_before else 'run'}",
                flush=True,
            )
            self.runner.run_score_once(config, model_id, checkpoint, output, shard, microbatch)
            if not valid_before:
                completed_this_call += 1
                elapsed = time.perf_counter() - started
                mean_job_seconds = elapsed / completed_this_call
                remaining = pending_at_start - completed_this_call
                print(
                    f"progress: {position}/{len(jobs)} enumerated; {remaining} jobs remain; "
                    f"observed ETA {remaining * mean_job_seconds / 3600:.2f} hours",
                    flush=True,
                )
        missing = [row for row in self.raw_status(microbatch) if not row["valid"]]
        if missing:
            raise RuntimeError(f"Production grid is incomplete; first invalid row: {missing[0]}")
        print(f"Production grid complete: {len(jobs)}/{len(jobs)} valid jobs", flush=True)

    def validate_raw_grid(self, microbatch: Optional[int] = None) -> None:
        selected_microbatch = int(microbatch or self.load_microbatch())
        for config in self.context.configs:
            model_indexes = {}
            for model_id in ("prior", "books"):
                indexes = []
                for shard in self.context.shards:
                    output = self.runner.shard_output_dir(config["config_id"], model_id, shard)
                    if not self.runner.valid_score_output(output, config, model_id, shard, selected_microbatch):
                        raise RuntimeError(f"Invalid raw shard: {output}")
                    index = np.memmap(output / "score" / "mmap_index.npy", dtype=np.int64, mode="r")
                    indexes.append(np.asarray(index, dtype=np.int64))
                combined = np.concatenate(indexes)
                if not np.array_equal(np.sort(combined), np.arange(self.context.rows, dtype=np.int64)):
                    raise RuntimeError(
                        f"{config['config_id']} {model_id}: score indexes do not cover the full pool"
                    )
                model_indexes[model_id] = combined
            if not np.array_equal(model_indexes["prior"], model_indexes["books"]):
                raise RuntimeError(f"{config['config_id']}: prior and Books score-index order differs")

    def _analysis_payload(self, config: Mapping[str, Any], microbatch: int) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "subset_fingerprint": self.runner.context.subset_fingerprint,
            "config": dict(config),
            "microbatch": microbatch,
            "raw_shard_fingerprints": {
                model_id: [
                    self.runner.shard_experiment_fingerprint(config, model_id, shard, microbatch)
                    for shard in self.context.shards
                ]
                for model_id in ("prior", "books")
            },
        }

    def validate_analysis(self, config: Mapping[str, Any], microbatch: Optional[int] = None) -> Dict[str, Any]:
        import pandas as pd

        selected_microbatch = int(microbatch or self.load_microbatch())
        config_id = str(config["config_id"])
        analysis = self.analysis_dir(config)
        strategy = analysis / "strategy"
        paths = {
            "npz": analysis / f"mc_samples_{config_id}.npz",
            "manifest": analysis / f"mc_samples_{config_id}_manifest.json",
            "summary": analysis / "color_distribution_summary.parquet",
            "metrics": strategy / "strategy_sweep_metrics.csv",
            "overlap": strategy / "strategy_selection_overlap.csv",
            "context": analysis / "analysis_context.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise FileNotFoundError("Missing analysis files: " + ", ".join(missing))
        expected_payload = self._analysis_payload(config, selected_microbatch)
        stored = json.loads(paths["context"].read_text())
        if (
            stored.get("fingerprint") != canonical_sha256(expected_payload)
            or stored.get("experiment") != expected_payload
        ):
            raise ValueError(f"{config_id}: stale analysis context")
        manifest = json.loads(paths["manifest"].read_text())
        expected_manifest = {
            "config_id": config_id,
            "rows": self.context.rows,
            "num_samples": int(config["num_samples"]),
            "row_alignment": "metadata_and_full_scores_indexed_by_score_index",
            "dropout_rate": float(config["dropout_rate"]),
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise ValueError(f"{config_id}: manifest {key}={manifest.get(key)!r}, expected {expected!r}")
        with np.load(paths["npz"], allow_pickle=False) as raw:
            color = raw["color_samples"]
            score_index = raw["score_index"].astype(np.int64)
            if color.shape != (self.context.rows, int(config["num_samples"])):
                raise ValueError(f"{config_id}: invalid color sample shape {color.shape}")
            if not np.isfinite(color).all():
                raise ValueError(f"{config_id}: non-finite color samples")
            if not np.array_equal(np.sort(score_index), np.arange(self.context.rows, dtype=np.int64)):
                raise ValueError(f"{config_id}: score_index is not a complete permutation")
        metrics = pd.read_csv(paths["metrics"])
        pairwise = metrics[metrics["metric_scope"] == "pairwise"]
        full_pool = metrics[metrics["metric_scope"] == "full_pool"]
        if len(pairwise) != self.EXPECTED_PAIRWISE_ROWS or len(full_pool) != self.EXPECTED_FULL_POOL_ROWS:
            raise ValueError(f"{config_id}: incomplete strategy metrics")
        return {"config_id": config_id, "rows": self.context.rows, "num_samples": int(config["num_samples"])}

    def analyze(self) -> None:
        microbatch = self.load_microbatch()
        missing_raw = [row for row in self.raw_status(microbatch) if not row["valid"]]
        if missing_raw:
            raise RuntimeError(f"Raw score grid is incomplete; first invalid row: {missing_raw[0]}")
        self.validate_raw_grid(microbatch)
        print("raw score-index grid covers the same complete 500K permutation for both models", flush=True)
        for config in self.context.configs:
            try:
                result = self.validate_analysis(config, microbatch)
                print("skip valid analysis:", result, flush=True)
                continue
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"rebuilding analysis for {config['config_id']}: {exc}", flush=True)
            analysis = self.analysis_dir(config)
            if analysis.exists():
                print("removing isolated stale analysis:", analysis, flush=True)
                shutil.rmtree(analysis)
            analysis.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                self.runner.context.olmo_dir / "scripts/21_dropout_uncertainty_metrics.py",
                "--prior-score-dir",
                *self.score_dirs(config, "prior"),
                "--conditional-score-dir",
                *self.score_dirs(config, "books"),
                "--output-dir",
                analysis,
                "--config-id",
                config["config_id"],
                "--metadata",
                self.context.metadata_path,
                "--full-scores",
                self.context.full_scores_path,
                "--num-samples",
                config["num_samples"],
                "--max-rows",
                self.context.rows,
                "--dropout-rate",
                config["dropout_rate"],
                "--dropout-target",
                config["dropout_target"],
                "--attention-dropout",
                config["attention_dropout"],
                "--residual-dropout",
                config["residual_dropout"],
                "--embedding-dropout",
                config["embedding_dropout"],
                "--seed",
                self.runner.context.seed,
                "--summary-format",
                "both",
                "--compress-npz",
            ]
            print(f"analysis {config['config_id']}: aggregate 500,000 rows", flush=True)
            self.run_analysis_command(command, analysis / "aggregate.log", f"aggregate {config['config_id']}")
            strategy = analysis / "strategy"
            print(f"analysis {config['config_id']}: evaluate strategy sweep", flush=True)
            self.run_analysis_command(
                [
                    sys.executable,
                    self.analysis_script_dir / "22_dropout_strategy_sweep.py",
                    "--mc-samples",
                    analysis / f"mc_samples_{config['config_id']}.npz",
                    "--summary",
                    analysis / "color_distribution_summary.parquet",
                    "--output-dir",
                    strategy,
                    "--tau64-cutoff",
                    self.context.tau64_cutoff,
                ],
                analysis / "strategy.log",
                f"strategy {config['config_id']}",
            )
            payload = self._analysis_payload(config, microbatch)
            (analysis / "analysis_context.json").write_text(
                json.dumps(
                    {"fingerprint": canonical_sha256(payload), "experiment": payload}, indent=2, sort_keys=True
                )
                + "\n"
            )
            print("analysis complete:", self.validate_analysis(config, microbatch), flush=True)

    def _runtime_records(self, config: Mapping[str, Any], microbatch: int) -> Sequence[Dict[str, Any]]:
        records = []
        for model_id in ("prior", "books"):
            for shard in self.context.shards:
                marker = json.loads(
                    (
                        self.runner.shard_output_dir(config["config_id"], model_id, shard) / "completed.json"
                    ).read_text()
                )
                runtime = marker.get("runtime_metrics", {})
                records.append(
                    {
                        "config_id": config["config_id"],
                        "model_id": model_id,
                        "rows": int(shard["rows"]),
                        "elapsed_seconds": marker.get("elapsed_seconds"),
                        "tokens_per_second": runtime.get("tokens_per_second"),
                        "batches_per_second": runtime.get("batches_per_second"),
                        "peak_gpu_memory_mb": runtime.get("peak_gpu_memory_mb"),
                        "microbatch": marker.get("microbatch"),
                    }
                )
        validate_runtime_records(records, self.context.global_batch_size, microbatch)
        return records

    def build_report(self) -> Dict[str, Any]:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        started = time.perf_counter()
        print("report: starting report generation", flush=True)
        microbatch = self.load_microbatch()
        summary_rows = []
        pairwise_rows = []
        runtime_rows = []
        for config in self.context.configs:
            self.validate_analysis(config, microbatch)
            analysis = self.analysis_dir(config)
            with np.load(analysis / f"mc_samples_{config['config_id']}.npz", allow_pickle=False) as raw:
                color = raw["color_samples"]
                full = raw["full_color_score"]
            mean_color = color.mean(axis=1)
            std_color = finite_sample_std(color)
            metrics = pd.read_csv(analysis / "strategy" / "strategy_sweep_metrics.csv")
            pair_mean = metrics[(metrics["metric_scope"] == "pairwise") & (metrics["strategy"] == "mean")]
            full_mean = metrics[(metrics["metric_scope"] == "full_pool") & (metrics["strategy"] == "mean")]
            recall = float(
                full_mean.loc[np.isclose(full_mean["selection_rate"], 1 / 64), "recall_vs_full"].iloc[0]
            )
            runtimes = pd.DataFrame(self._runtime_records(config, microbatch))
            runtime_rows.extend(runtimes.to_dict("records"))
            summary_rows.append(
                {
                    "config_id": config["config_id"],
                    "source": "sweep",
                    "dropout_rate": float(config["dropout_rate"]),
                    "K": int(config["num_samples"]),
                    "rows": self.context.rows,
                    "pearson_mean_vs_full": float(np.corrcoef(mean_color, full)[0, 1]),
                    "spearman_mean_vs_full": float(pd.Series(mean_color).rank().corr(pd.Series(full).rank())),
                    "mean_pairwise_auc": float(pair_mean["roc_auc"].mean()),
                    "hp_vs_hn_auc": float(pair_mean.loc[pair_mean["task_id"] == "hp_vs_hn", "roc_auc"].iloc[0]),
                    "recall_vs_full_1_64": recall,
                    "mean_mc_std": float(std_color.mean()),
                    "runtime_hours": float(runtimes["elapsed_seconds"].sum() / 3600),
                    "tokens_per_second_mean": float(runtimes["tokens_per_second"].dropna().mean()),
                    "peak_gpu_memory_mb": float(runtimes["peak_gpu_memory_mb"].max()),
                }
            )
            for row in pair_mean.itertuples(index=False):
                pairwise_rows.append(
                    {
                        "config_id": config["config_id"],
                        "source": "sweep",
                        "dropout_rate": float(config["dropout_rate"]),
                        "K": int(config["num_samples"]),
                        "task_id": row.task_id,
                        "roc_auc": row.roc_auc,
                        "average_precision": row.average_precision,
                        "balanced_f1": row.balanced_f1,
                        "tau64_f1": row.tau64_f1,
                    }
                )
        reference = self.context.reference_p005_analysis
        if reference is not None and Path(reference).is_dir():
            reference = Path(reference)
            reference_npz = reference / "mc_samples_dropout_k8_p005.npz"
            reference_metrics = reference / "strategy" / "strategy_sweep_metrics.csv"
            reference_manifest = reference / "mc_samples_dropout_k8_p005_manifest.json"
            for path in (reference_npz, reference_metrics, reference_manifest):
                if not path.is_file() or path.stat().st_size <= 0:
                    raise FileNotFoundError(f"Incomplete p=0.05 reference: {path}")
            manifest = json.loads(reference_manifest.read_text())
            if (
                manifest.get("rows") != self.context.rows
                or manifest.get("row_alignment") != "metadata_and_full_scores_indexed_by_score_index"
            ):
                raise ValueError("The p=0.05 reference does not use the corrected full-pool alignment")
            with np.load(reference_npz, allow_pickle=False) as raw:
                reference_color = raw["color_samples"]
                reference_full = raw["full_color_score"]
            reference_mean = reference_color.mean(axis=1)
            reference_std = finite_sample_std(reference_color)
            metrics = pd.read_csv(reference_metrics)
            pair_mean = metrics[(metrics["metric_scope"] == "pairwise") & (metrics["strategy"] == "mean")]
            full_mean = metrics[(metrics["metric_scope"] == "full_pool") & (metrics["strategy"] == "mean")]
            summary_rows.append(
                {
                    "config_id": "dropout_k8_p005",
                    "source": "existing_alignment_fixed_reference",
                    "dropout_rate": 0.05,
                    "K": 8,
                    "rows": self.context.rows,
                    "pearson_mean_vs_full": float(np.corrcoef(reference_mean, reference_full)[0, 1]),
                    "spearman_mean_vs_full": float(
                        pd.Series(reference_mean).rank().corr(pd.Series(reference_full).rank())
                    ),
                    "mean_pairwise_auc": float(pair_mean["roc_auc"].mean()),
                    "hp_vs_hn_auc": float(pair_mean.loc[pair_mean["task_id"] == "hp_vs_hn", "roc_auc"].iloc[0]),
                    "recall_vs_full_1_64": float(
                        full_mean.loc[np.isclose(full_mean["selection_rate"], 1 / 64), "recall_vs_full"].iloc[0]
                    ),
                    "mean_mc_std": float(reference_std.mean()),
                    "runtime_hours": np.nan,
                    "tokens_per_second_mean": np.nan,
                    "peak_gpu_memory_mb": np.nan,
                }
            )
            for row in pair_mean.itertuples(index=False):
                pairwise_rows.append(
                    {
                        "config_id": "dropout_k8_p005",
                        "source": "existing_alignment_fixed_reference",
                        "dropout_rate": 0.05,
                        "K": 8,
                        "task_id": row.task_id,
                        "roc_auc": row.roc_auc,
                        "average_precision": row.average_precision,
                        "balanced_f1": row.balanced_f1,
                        "tau64_f1": row.tau64_f1,
                    }
                )
        summary = pd.DataFrame(summary_rows).sort_values("dropout_rate").reset_index(drop=True)
        pairwise = pd.DataFrame(pairwise_rows).sort_values(["task_id", "dropout_rate"])
        runtime = pd.DataFrame(runtime_rows)
        self.context.report_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.context.report_dir / "sweep_summary.csv", index=False)
        pairwise.to_csv(self.context.report_dir / "pairwise_mean.csv", index=False)
        runtime.to_csv(self.context.report_dir / "runtime_summary.csv", index=False)

        zero = summary.loc[np.isclose(summary["dropout_rate"], 0.0)].iloc[0]
        zero_passed = bool(
            zero["K"] == 1
            and zero["mean_mc_std"] == 0.0
            and zero["pearson_mean_vs_full"] >= 0.99
            and zero["spearman_mean_vs_full"] >= 0.90
        )
        acceptance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "rows": self.context.rows,
            "rates": [float(config["dropout_rate"]) for config in self.context.configs],
            "reference_p005_included": bool((summary["source"] == "existing_alignment_fixed_reference").any()),
            "all_configs_complete": int((summary["source"] == "sweep").sum()) == len(self.context.configs),
            "zero_dropout_control_passed": zero_passed,
            "row_alignment": "metadata_and_full_scores_indexed_by_score_index",
        }
        (self.context.report_dir / "sweep_acceptance.json").write_text(
            json.dumps(acceptance, indent=2, sort_keys=True) + "\n"
        )
        if not zero_passed:
            raise RuntimeError(f"Zero-dropout full-pool control failed: {zero.to_dict()}")

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(
            summary["dropout_rate"],
            summary["spearman_mean_vs_full"],
            marker="o",
            label="Spearman",
        )
        axes[0].plot(
            summary["dropout_rate"],
            summary["recall_vs_full_1_64"],
            marker="s",
            label="Recall@1/64",
        )
        axes[0].set_xscale("symlog", linthresh=1e-5)
        axes[0].set_xlabel("Broad dropout rate p")
        axes[0].set_ylabel("Metric")
        axes[0].set_ylim(0, 1.02)
        axes[0].legend()
        axes[1].plot(
            summary["dropout_rate"],
            summary["mean_pairwise_auc"],
            marker="o",
            label="Mean pairwise AUC",
        )
        axes[1].plot(
            summary["dropout_rate"],
            summary["hp_vs_hn_auc"],
            marker="s",
            label="hp_vs_hn AUC",
        )
        axes[1].set_xscale("symlog", linthresh=1e-5)
        axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Broad dropout rate p")
        axes[1].set_ylabel("AUC")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.context.report_dir / "rate_sweep_metrics.png", dpi=170)
        plt.close(figure)

        report = (
            "# Broad MC-Dropout Rate Sweep\n\n"
            "## Scope\n\n"
            f"- Rows: {self.context.rows:,}\n"
            f"- Rates: {', '.join(f'{rate:g}' for rate in REQUESTED_RATES)}\n"
            "- Dropout target: attention + residual + embedding\n"
            "- Zero-dropout control: K=1; stochastic configurations: K=8\n"
            "- Alignment: metadata and deterministic full scores indexed by `score_index`\n\n"
            "The corrected broad `p=0.05` result is included as a read-only reference when present.\n\n"
            "## Sweep Summary\n\n"
            + markdown_table(summary)
            + "\n\n## Per-Task Mean Strategy\n\n"
            + markdown_table(pairwise)
            + "\n\n## Figure\n\n![Rate sweep metrics](rate_sweep_metrics.png)\n"
        )
        (self.context.report_dir / "report.md").write_text(report, encoding="utf-8")
        (self.context.report_dir / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Broad MC-dropout rate sweep</title>'
            "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}"
            "table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border:1px solid #ccc;padding:5px}img{max-width:100%}</style>"
            "<h1>Broad MC-Dropout Rate Sweep</h1>"
            f"<p>Rows: {self.context.rows:,}; rates: {html.escape(str(REQUESTED_RATES))}</p>"
            + summary.to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + "<h2>Per-task mean strategy</h2>"
            + pairwise.to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + '<h2>Metrics</h2><img src="rate_sweep_metrics.png" alt="Rate sweep metrics">',
            encoding="utf-8",
        )
        print(
            f"report: complete; elapsed_seconds={time.perf_counter() - started:.2f}; "
            f"output={self.context.report_dir}",
            flush=True,
        )
        return {"summary": summary, "pairwise": pairwise, "acceptance": acceptance}

    def build_bundle(self, local_archive: Path, drive_archive: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        print(f"bundle: starting; output={drive_archive}", flush=True)
        report = self.build_report()
        required = [
            self.context.report_dir / "report.md",
            self.context.report_dir / "report.html",
            self.context.report_dir / "rate_sweep_metrics.png",
            self.context.report_dir / "sweep_summary.csv",
            self.context.report_dir / "pairwise_mean.csv",
            self.context.report_dir / "runtime_summary.csv",
            self.context.report_dir / "sweep_acceptance.json",
            self.context.run_state_path,
            self.context.shard_plan_path,
            self.context.subset_manifest_path,
        ]
        files = [(path, f"report/{path.name}") for path in required[:7]]
        files.extend((path, f"manifests/{path.name}") for path in required[7:])
        for config in self.context.configs:
            analysis = self.analysis_dir(config)
            config_id = str(config["config_id"])
            for name in (
                f"mc_samples_{config_id}_manifest.json",
                "analysis_context.json",
                "aggregate.log",
                "strategy.log",
            ):
                files.append((analysis / name, f"analysis/{config_id}/{name}"))
            for name in ("strategy_sweep_metrics.csv", "strategy_selection_overlap.csv"):
                files.append((analysis / "strategy" / name, f"analysis/{config_id}/strategy/{name}"))
        for config_path in sorted(self.context.config_drive.glob("*.yaml")):
            files.append((config_path, f"runtime_configs/{config_path.name}"))
        for config in self.context.configs:
            config_id = str(config["config_id"])
            for model_id in ("prior", "books"):
                for shard in self.context.shards:
                    output = self.runner.shard_output_dir(config_id, model_id, shard)
                    shard_name = f"{int(shard['start']):06d}_{int(shard['end']):06d}"
                    files.append(
                        (
                            output / "completed.json",
                            f"scoring/{config_id}/{model_id}/{shard_name}_completed.json",
                        )
                    )
                    files.append(
                        (
                            output.with_suffix(".log"),
                            f"scoring/{config_id}/{model_id}/{shard_name}.log",
                        )
                    )
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "rows": self.context.rows,
            "configs": [dict(config) for config in self.context.configs],
            "raw_scores_excluded": True,
            "large_mc_sample_tables_excluded": True,
            "zero_dropout_control_passed": report["acceptance"]["zero_dropout_control_passed"],
        }
        local, drive = create_verified_archive(files, manifest, local_archive, drive_archive)
        names = verify_bundle_archive(drive)
        print(
            f"bundle: complete; elapsed_seconds={time.perf_counter() - started:.2f}; "
            f"members={len(names)}; output={drive}",
            flush=True,
        )
        return {"local_archive": local, "drive_archive": drive, "file_count": len(names), "members": names}
