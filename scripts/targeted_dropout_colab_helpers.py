"""Runtime helpers for the targeted-dropout Stage B/C Colab notebook."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def select_fastest_benchmark(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    result_list = [dict(result) for result in results]
    successful = []
    for result in result_list:
        throughput = positive_float(result.get("tokens_per_second"))
        if result.get("status") == "ok" and throughput is not None:
            normalized = dict(result)
            normalized["tokens_per_second"] = throughput
            successful.append(normalized)
    if not successful:
        raise RuntimeError(f"No microbatch candidate recorded valid throughput: {result_list}")
    return max(successful, key=lambda result: result["tokens_per_second"])


@dataclass(frozen=True)
class ScoringContext:
    olmo_dir: Path
    template_config: Path
    runtime_checkpoint_dir: Path
    runtime_config_dir: Path
    config_drive: Path
    raw_score_drive: Path
    stage_root: Path
    subset_raw: Path
    run_state_path: Path
    producer_sha: str
    analysis_sha: str
    notebook_revision: str
    run_stage: str
    subset_id: str
    subset_fingerprint: str
    runtime_identity: Mapping[str, Any]
    checkpoint_identities: Mapping[str, Any]
    seed: int
    num_samples: int
    global_batch_size: int
    stage_rows: int
    shard_rows: int
    file_seqs: int


class TargetedDropoutRunner:
    def __init__(self, context: ScoringContext):
        self.context = context

    @staticmethod
    def _config_types():
        from omegaconf import OmegaConf
        from olmo.config import TrainConfig

        return OmegaConf, TrainConfig

    @staticmethod
    def _symlink_or_refresh(src: Path, dst: Path) -> None:
        src, dst = Path(src), Path(dst)
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return
            except FileNotFoundError:
                pass
            dst.unlink()
        os.symlink(src, dst)

    def prepare_model_only_checkpoint(self, checkpoint_path: Path) -> Path:
        import torch

        source = Path(checkpoint_path)
        runtime = Path(self.context.runtime_checkpoint_dir) / source.name
        runtime.mkdir(parents=True, exist_ok=True)
        self._symlink_or_refresh(source / "model.pt", runtime / "model.pt")
        self._symlink_or_refresh(source / "config.yaml", runtime / "config.yaml")
        for state_name in ("train.pt", "other.pt"):
            state_path = runtime / state_name
            if not state_path.exists():
                torch.save({}, state_path)
        return runtime

    def load_checkpoint_score_config(self, checkpoint_path: Path):
        OmegaConf, _ = self._config_types()
        checkpoint_cfg = OmegaConf.load(Path(checkpoint_path) / "config.yaml")
        cfg = OmegaConf.load(self.context.template_config)
        cfg.model = checkpoint_cfg.model
        if "tokenizer" in checkpoint_cfg:
            cfg.tokenizer = checkpoint_cfg.tokenizer
        if "targeted_ladder" in cfg:
            del cfg["targeted_ladder"]
        return cfg

    def build_score_config(
        self,
        config: Mapping[str, Any],
        model_id: str,
        checkpoint_path: Path,
        output_dir: Path,
        shard: Mapping[str, int],
        microbatch: int,
        console_log_interval: int = 25,
    ):
        rows = int(shard["rows"])
        cfg = self.load_checkpoint_score_config(checkpoint_path)
        cfg.run_name = f"{config['config_id']}_{model_id}_{int(shard['start']):06d}_{int(shard['end']):06d}"
        cfg.save_folder = str(output_dir)
        cfg.load_path = str(self.prepare_model_only_checkpoint(checkpoint_path))
        cfg.load_checkpoint_type = "unsharded"
        cfg.max_duration = rows // self.context.global_batch_size
        cfg.data_start_step = int(shard["data_start_step"])
        cfg.global_train_batch_size = self.context.global_batch_size
        cfg.device_train_batch_size = self.context.global_batch_size
        cfg.device_train_microbatch_size = int(microbatch)
        cfg.data.paths = [str(self.context.subset_raw)]
        cfg.data.memmap_dtype = "uint32"
        cfg.data.num_workers = 0
        cfg.seed = self.context.seed
        cfg.model.attention_dropout = float(config["attention_dropout"])
        cfg.model.residual_dropout = float(config["residual_dropout"])
        cfg.model.embedding_dropout = float(config["embedding_dropout"])
        cfg.uncertainty_scoring.enabled = True
        cfg.uncertainty_scoring.num_samples = self.context.num_samples
        cfg.uncertainty_scoring.perturbation_type = "dropout"
        cfg.uncertainty_scoring.coupled_masks = True
        cfg.restore_dataloader = False
        cfg.reset_optimizer_state = True
        cfg.reset_trainer_state = True
        cfg.console_log_interval = min(max(1, int(console_log_interval)), max(1, int(cfg.max_duration)))
        cfg.gen1_gc_interval = 50
        cfg.save_data_indices = True
        return cfg

    def write_config(self, cfg: Any, name: str) -> Path:
        OmegaConf, TrainConfig = self._config_types()
        local_path = Path(self.context.runtime_config_dir) / name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, local_path)
        TrainConfig.load(str(local_path), validate_paths=False)
        drive_path = Path(self.context.config_drive) / name
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
        return local_path

    def run_logged(
        self,
        args: Sequence[Any],
        log_path: Path,
        cwd: Optional[Path] = None,
    ) -> float:
        env = os.environ.copy()
        olmo_dir = str(self.context.olmo_dir)
        env["PYTHONPATH"] = olmo_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print("running:", " ".join(str(arg) for arg in args))
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(arg) for arg in args],
                cwd=str(cwd or self.context.olmo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            returncode = proc.wait()
        elapsed = time.perf_counter() - start
        if returncode != 0:
            tail = log_path.read_text(errors="ignore")[-4000:]
            raise RuntimeError(f"Command failed with code {returncode}. Log tail:\n{tail}")
        print("elapsed_seconds:", round(elapsed, 2), "log:", log_path)
        return elapsed

    @staticmethod
    def parse_score_log_metrics(log_path: Path) -> Dict[str, Optional[float]]:
        text = Path(log_path).read_text(errors="ignore")

        def last_float(pattern: str) -> Optional[float]:
            matches = re.findall(pattern, text)
            return float(matches[-1].replace(",", "")) if matches else None

        return {
            "tokens_per_second": last_float(r"throughput/device/tokens_per_second=([0-9,.]+)"),
            "batches_per_second": last_float(r"throughput/device/batches_per_second=([0-9,.]+)"),
            "peak_gpu_memory_mb": last_float(r"System/Peak GPU Memory \(MB\)=([0-9,.]+)"),
        }

    def score_width(self, score_dir: Path) -> int:
        score_dir = Path(score_dir)
        files_path = score_dir / "files.txt"
        if not files_path.exists():
            return 0
        files = [line.strip() for line in files_path.read_text().splitlines() if line.strip()]
        if not files:
            return 0
        first = Path(files[0])
        if not first.exists():
            first = score_dir / first.name
        values = first.stat().st_size // np.dtype(np.float32).itemsize
        return values // self.context.file_seqs

    def shard_experiment_payload(
        self,
        config: Mapping[str, Any],
        model_id: str,
        shard: Mapping[str, int],
        microbatch: int,
    ) -> Dict[str, Any]:
        return {
            "schema_version": 4,
            "producer_sha": self.context.producer_sha,
            "stage": self.context.run_stage,
            "subset_id": self.context.subset_id,
            "subset_fingerprint": self.context.subset_fingerprint,
            "config": dict(config),
            "runtime_identity": dict(self.context.runtime_identity),
            "model_id": model_id,
            "checkpoint_identity": self.context.checkpoint_identities[model_id],
            "seed": self.context.seed,
            "num_samples": self.context.num_samples,
            "coupled_masks": True,
            "global_batch_size": self.context.global_batch_size,
            "microbatch": int(microbatch),
            "shard": {key: int(value) for key, value in shard.items()},
        }

    def shard_experiment_fingerprint(
        self,
        config: Mapping[str, Any],
        model_id: str,
        shard: Mapping[str, int],
        microbatch: int,
    ) -> str:
        return canonical_sha256(self.shard_experiment_payload(config, model_id, shard, microbatch))

    def score_prefix_is_finite(self, score_dir: Path, expected_rows: int) -> bool:
        score_dir = Path(score_dir)
        files = [line.strip() for line in (score_dir / "files.txt").read_text().splitlines() if line.strip()]
        if len(files) != 1:
            return False
        score_path = Path(files[0])
        if not score_path.exists():
            score_path = score_dir / score_path.name
        if not score_path.exists() or self.score_width(score_dir) != self.context.num_samples:
            return False
        scores = np.memmap(
            score_path,
            dtype=np.float32,
            mode="r",
            shape=(self.context.file_seqs, self.context.num_samples),
        )
        return bool(np.isfinite(np.asarray(scores[:expected_rows])).all())

    def valid_score_output(
        self,
        output_dir: Path,
        config: Mapping[str, Any],
        model_id: str,
        shard: Mapping[str, int],
        microbatch: int,
    ) -> bool:
        expected_rows = int(shard["rows"])
        output_dir = Path(output_dir)
        score_dir = output_dir / "score"
        marker = output_dir / "completed.json"
        index_path = score_dir / "mmap_index.npy"
        if not marker.exists() or not index_path.exists() or not (score_dir / "files.txt").exists():
            return False
        try:
            record = json.loads(marker.read_text())
            expected_fingerprint = self.shard_experiment_fingerprint(config, model_id, shard, microbatch)
            if record.get("experiment_fingerprint") != expected_fingerprint:
                return False
            if record.get("experiment") != self.shard_experiment_payload(config, model_id, shard, microbatch):
                return False
            index = np.memmap(index_path, dtype=np.int64, mode="r")
            if len(index) != expected_rows:
                return False
            values = np.asarray(index, dtype=np.int64)
            if len(np.unique(values)) != expected_rows:
                return False
            if values.min(initial=0) < 0 or values.max(initial=-1) >= self.context.stage_rows:
                return False
            return self.score_prefix_is_finite(score_dir, expected_rows)
        except Exception:
            return False

    def safely_remove_invalid_shard(self, output_dir: Path) -> None:
        output_dir = Path(output_dir).resolve()
        raw_root = Path(self.context.raw_score_drive).resolve()
        smoke_root = (Path(self.context.stage_root) / "_smoke_and_benchmark").resolve()
        if raw_root not in output_dir.parents and smoke_root not in output_dir.parents:
            raise RuntimeError(f"Refusing to remove output outside isolated roots: {output_dir}")
        shutil.rmtree(output_dir)

    def run_score_once(
        self,
        config: Mapping[str, Any],
        model_id: str,
        checkpoint_path: Path,
        output_dir: Path,
        shard: Mapping[str, int],
        microbatch: int,
        console_log_interval: int = 25,
    ) -> Path:
        expected_rows = int(shard["rows"])
        output_dir = Path(output_dir)
        if self.valid_score_output(output_dir, config, model_id, shard, microbatch):
            print("skip valid shard:", output_dir)
            return output_dir / "score"
        if output_dir.exists():
            print("recomputing invalid isolated shard:", output_dir)
            self.safely_remove_invalid_shard(output_dir)
        cfg = self.build_score_config(
            config,
            model_id,
            checkpoint_path,
            output_dir,
            shard,
            microbatch,
            console_log_interval,
        )
        name = f"{config['config_id']}_{model_id}_{int(shard['start']):06d}_" f"{int(shard['end']):06d}.yaml"
        cfg_path = self.write_config(cfg, name)
        log_path = output_dir.with_suffix(".log")
        elapsed = self.run_logged(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node=1",
                "scripts/train.py",
                cfg_path,
                "--save_overwrite=true",
            ],
            log_path,
        )
        experiment = self.shard_experiment_payload(config, model_id, shard, microbatch)
        marker = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "config_id": config["config_id"],
            "model_id": model_id,
            "start": int(shard["start"]),
            "end": int(shard["end"]),
            "rows": expected_rows,
            "num_samples": self.context.num_samples,
            "microbatch": int(microbatch),
            "elapsed_seconds": elapsed,
            "producer_sha": self.context.producer_sha,
            "runtime_metrics": self.parse_score_log_metrics(log_path),
            "experiment_fingerprint": canonical_sha256(experiment),
            "experiment": experiment,
        }
        (output_dir / "completed.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        if not self.valid_score_output(output_dir, config, model_id, shard, microbatch):
            raise RuntimeError(f"Score shard failed validation: {output_dir}")
        return output_dir / "score"

    def benchmark_microbatches(
        self,
        config: Mapping[str, Any],
        model_id: str,
        checkpoint_path: Path,
        benchmark_root: Path,
        shard: Mapping[str, int],
        candidates: Sequence[int],
        seq_len: int,
    ) -> Tuple[Sequence[Dict[str, Any]], Dict[str, Any]]:
        results = []
        for candidate in candidates:
            output_dir = Path(benchmark_root) / f"{model_id}_microbatch_{candidate}"
            try:
                marker_path = output_dir / "completed.json"
                if marker_path.exists():
                    try:
                        existing_marker = json.loads(marker_path.read_text())
                        existing_tps = positive_float(
                            existing_marker.get("runtime_metrics", {}).get("tokens_per_second")
                        )
                    except (OSError, json.JSONDecodeError):
                        existing_tps = None
                    if existing_tps is None:
                        print("recomputing benchmark missing throughput metrics:", output_dir)
                        self.safely_remove_invalid_shard(output_dir)
                started = time.perf_counter()
                self.run_score_once(
                    config,
                    model_id,
                    checkpoint_path,
                    output_dir,
                    shard,
                    candidate,
                    console_log_interval=1,
                )
                wall = time.perf_counter() - started
                marker = json.loads((output_dir / "completed.json").read_text())
                measured_tps = positive_float(marker.get("runtime_metrics", {}).get("tokens_per_second"))
                if measured_tps is None:
                    raise RuntimeError(
                        "Benchmark did not record valid device throughput: " f"{marker.get('runtime_metrics')}"
                    )
                results.append(
                    {
                        "microbatch": int(candidate),
                        "status": "ok",
                        "cell_wall_seconds": wall,
                        "measured_compute_seconds": float(marker["elapsed_seconds"]),
                        "tokens_per_second": measured_tps,
                        "rows": int(shard["rows"]),
                        "tokens": int(shard["rows"]) * int(seq_len),
                    }
                )
            except Exception as exc:
                results.append({"microbatch": int(candidate), "status": f"failed: {exc}"})
                print("stopping after first failed candidate")
                break
        return results, select_fastest_benchmark(results)

    def load_persisted_microbatch(self) -> int:
        run_state_path = Path(self.context.run_state_path)
        if not run_state_path.exists():
            raise FileNotFoundError(f"Run state is missing; run bounded Section 7 first: {run_state_path}")
        state = json.loads(run_state_path.read_text())
        expected = {
            "schema_version": 2,
            "stage": self.context.run_stage,
            "producer_sha": self.context.producer_sha,
            "subset_fingerprint": self.context.subset_fingerprint,
            "shard_rows": self.context.shard_rows,
            "num_samples": self.context.num_samples,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Persisted run state does not match this experiment: {state}")
        microbatch = int(state["microbatch"])
        if microbatch not in (16, 32):
            raise RuntimeError(f"Unexpected persisted microbatch: {microbatch}")
        return microbatch

    def shard_output_dir(self, config_id: str, model_id: str, shard: Mapping[str, int]) -> Path:
        return (
            Path(self.context.raw_score_drive)
            / config_id
            / model_id
            / f"shard_{int(shard['start']):06d}_{int(shard['end']):06d}"
        )


@dataclass(frozen=True)
class SubsetContext:
    run_stage: str
    subset_id: str
    stage_rows: int
    seq_len: int
    selection_seed: int
    expected_source_rows: int
    rows_per_stage_b_pool: int
    tokens_path: Path
    metadata_path: Path
    full_scores_path: Path
    prior_checkpoint: Path
    books_checkpoint: Path
    local_work: Path
    source_rows_path: Path
    subset_manifest_path: Path
    producer_sha: str
    analysis_sha: str
    notebook_revision: str
    runtime_identity: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedSubset:
    metadata_path: Path
    full_scores_path: Path
    raw_tokens_path: Path
    subset_fingerprint: str
    checkpoint_identities: Mapping[str, Any]
    source_token_identity: Mapping[str, Any]
    pool_counts: Mapping[str, int]


def sampled_file_identity(path: Path, sample_bytes: int = 1_048_576) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {"path": str(path), "bytes": stat.st_size, "sampled_sha256": digest.hexdigest()}


def prepare_fixed_subset(context: SubsetContext) -> PreparedSubset:
    import pandas as pd

    required = [
        context.tokens_path,
        context.metadata_path,
        context.full_scores_path,
        context.prior_checkpoint / "model.pt",
        context.prior_checkpoint / "config.yaml",
        context.books_checkpoint / "model.pt",
        context.books_checkpoint / "config.yaml",
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required Drive inputs:\n" + "\n".join(missing))

    source_token_identity = sampled_file_identity(context.tokens_path)
    checkpoint_identities = {}
    for model_id, checkpoint in (
        ("prior", context.prior_checkpoint),
        ("books", context.books_checkpoint),
    ):
        model_file = checkpoint / "model.pt"
        config_file = checkpoint / "config.yaml"
        if model_file.stat().st_size <= 0 or config_file.stat().st_size <= 0:
            raise ValueError(f"Empty checkpoint artifact under {checkpoint}")
        checkpoint_identities[model_id] = {
            "model": sampled_file_identity(model_file),
            "config": sampled_file_identity(config_file, config_file.stat().st_size),
        }
    if (
        checkpoint_identities["prior"]["model"]["sampled_sha256"]
        == checkpoint_identities["books"]["model"]["sampled_sha256"]
    ):
        raise RuntimeError("Prior and Books checkpoints have the same sampled fingerprint")

    source_tokens = np.load(context.tokens_path, mmap_mode="r")
    source_meta = pd.read_parquet(context.metadata_path)
    source_full = pd.read_parquet(context.full_scores_path)
    if source_tokens.ndim != 2 or source_tokens.shape[1] != context.seq_len:
        raise ValueError(f"Expected token shape (N, {context.seq_len}), found {source_tokens.shape}")
    if source_tokens.dtype not in (np.dtype(np.int32), np.dtype(np.uint32)):
        raise ValueError(f"Expected official tokens to use int32 or uint32, found {source_tokens.dtype}")
    source_rows = source_tokens.shape[0]
    if (
        source_rows != context.expected_source_rows
        or len(source_meta) != source_rows
        or len(source_full) != source_rows
    ):
        raise ValueError(
            "Official source row mismatch: "
            f"tokens={source_rows}, metadata={len(source_meta)}, full={len(source_full)}"
        )
    if "pool_name" not in source_meta.columns:
        raise ValueError("Official metadata is missing pool_name")
    full_color_columns = [
        column
        for column in ("full_color_score", "color", "color_score", "ablated_color_score")
        if column in source_full.columns
    ]
    if not full_color_columns:
        raise ValueError(f"Could not identify deterministic full color column: {source_full.columns.tolist()}")

    expected_pools = (
        "hard_positive",
        "hard_negative",
        "random_positive",
        "random_negative",
        "tail_negative",
    )
    source_pool_counts = source_meta["pool_name"].value_counts().to_dict()
    if any(source_pool_counts.get(pool, 0) < context.rows_per_stage_b_pool for pool in expected_pools):
        raise ValueError(f"Official pool counts cannot support balanced Stage B: {source_pool_counts}")

    if context.run_stage == "stage_b_100k":
        rng = np.random.Generator(np.random.PCG64(context.selection_seed))
        selected_parts = []
        pool_values = source_meta["pool_name"].to_numpy()
        for pool in expected_pools:
            candidates = np.flatnonzero(pool_values == pool)
            selected_parts.append(rng.choice(candidates, size=context.rows_per_stage_b_pool, replace=False))
        selected_rows = np.sort(np.concatenate(selected_parts).astype(np.int64))
    elif context.run_stage == "stage_c_500k":
        selected_rows = np.arange(source_rows, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported stage: {context.run_stage}")
    if len(selected_rows) != context.stage_rows or len(np.unique(selected_rows)) != context.stage_rows:
        raise RuntimeError("Subset source-row selection is not complete and unique")

    subset_meta_frame = source_meta.iloc[selected_rows].copy().reset_index(drop=True)
    subset_full_frame = source_full.iloc[selected_rows].copy().reset_index(drop=True)
    subset_meta_frame["source_row"] = selected_rows
    subset_full_frame["source_row"] = selected_rows
    pool_counts = subset_meta_frame["pool_name"].value_counts().sort_index().to_dict()
    if context.run_stage == "stage_b_100k" and set(pool_counts.values()) != {context.rows_per_stage_b_pool}:
        raise RuntimeError(f"Stage B subset is not balanced: {pool_counts}")

    context.local_work.mkdir(parents=True, exist_ok=True)
    subset_meta = context.local_work / f"{context.subset_id}_meta.parquet"
    subset_full = context.local_work / f"{context.subset_id}_full_scores.parquet"
    subset_raw = context.local_work / f"{context.subset_id}_tokens.uint32.raw"
    subset_raw_state = context.local_work / f"{context.subset_id}_tokens.uint32.manifest.json"
    subset_meta_frame.to_parquet(subset_meta, index=False)
    subset_full_frame.to_parquet(subset_full, index=False)
    context.source_rows_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(context.source_rows_path, selected_rows)

    source_rows_sha256 = hashlib.sha256(selected_rows.tobytes()).hexdigest()
    expected_raw_bytes = context.stage_rows * context.seq_len * np.dtype(np.uint32).itemsize
    cache_payload = {
        "subset_id": context.subset_id,
        "rows": context.stage_rows,
        "seq_len": context.seq_len,
        "dtype": "uint32",
        "source_rows_sha256": source_rows_sha256,
        "source_token_identity": source_token_identity,
    }
    cache_fingerprint = canonical_sha256(cache_payload)
    try:
        cached_state = json.loads(subset_raw_state.read_text()) if subset_raw_state.exists() else {}
    except (OSError, json.JSONDecodeError):
        cached_state = {}
    cache_matches = (
        subset_raw.exists()
        and subset_raw.stat().st_size == expected_raw_bytes
        and cached_state.get("fingerprint") == cache_fingerprint
    )
    if not cache_matches:
        temp_raw = subset_raw.with_suffix(".tmp")
        with temp_raw.open("wb") as handle:
            for start in range(0, context.stage_rows, 4096):
                rows = selected_rows[start : start + 4096]
                np.asarray(source_tokens[rows], dtype=np.uint32).tofile(handle)
        os.replace(temp_raw, subset_raw)
        subset_raw_state.write_text(
            json.dumps({"fingerprint": cache_fingerprint, **cache_payload}, indent=2, sort_keys=True) + "\n"
        )
    if subset_raw.stat().st_size != expected_raw_bytes:
        raise RuntimeError(f"Local raw token copy has wrong size: {subset_raw.stat().st_size}")

    subset_fingerprint = canonical_sha256(
        {
            **cache_payload,
            "pool_counts": pool_counts,
            "full_score_column": full_color_columns[0],
        }
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subset_id": context.subset_id,
        "row_count": context.stage_rows,
        "pool_counts": pool_counts,
        "selection_method": (
            "all_official_rows" if context.run_stage == "stage_c_500k" else "balanced_without_replacement"
        ),
        "selection_seed": (None if context.run_stage == "stage_c_500k" else context.selection_seed),
        "source_rows_path": str(context.source_rows_path),
        "source_rows_sha256": source_rows_sha256,
        "source_token_identity": source_token_identity,
        "checkpoint_identities": checkpoint_identities,
        "runtime_identity": dict(context.runtime_identity),
        "producer_sha": context.producer_sha,
        "analysis_sha": context.analysis_sha,
        "notebook_revision": context.notebook_revision,
        "subset_fingerprint": subset_fingerprint,
        "source_metadata_path": str(context.metadata_path),
        "source_token_path": str(context.tokens_path),
        "full_score_reference_path": str(context.full_scores_path),
        "full_score_column": full_color_columns[0],
        "same_subset_for_every_config": True,
    }
    context.subset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    context.subset_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return PreparedSubset(
        metadata_path=subset_meta,
        full_scores_path=subset_full,
        raw_tokens_path=subset_raw,
        subset_fingerprint=subset_fingerprint,
        checkpoint_identities=checkpoint_identities,
        source_token_identity=source_token_identity,
        pool_counts=pool_counts,
    )


def build_shard_plan(
    stage_rows: int,
    shard_rows: int,
    global_batch_size: int,
    output_path: Optional[Path] = None,
) -> Sequence[Dict[str, int]]:
    if shard_rows % global_batch_size != 0 or stage_rows % global_batch_size != 0:
        raise ValueError("Stage and shard rows must be divisible by the global batch size")
    shards = []
    start = 0
    while start < stage_rows:
        rows = min(shard_rows, stage_rows - start)
        if rows % global_batch_size != 0:
            raise ValueError(f"Final shard is not batch aligned: start={start}, rows={rows}")
        shards.append(
            {
                "start": start,
                "end": start + rows,
                "rows": rows,
                "data_start_step": start // global_batch_size,
            }
        )
        start += rows
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(shards, indent=2, sort_keys=True) + "\n")
    return shards


def validate_runtime_records(
    records: Sequence[Mapping[str, Any]],
    global_batch_size: int,
    expected_microbatch: int,
) -> None:
    if not records:
        raise RuntimeError("No runtime completion records were found")
    saw_multi_batch = False
    for record in records:
        elapsed = positive_float(record.get("elapsed_seconds"))
        peak_memory = positive_float(record.get("peak_gpu_memory_mb"))
        if elapsed is None or peak_memory is None:
            raise RuntimeError("Incomplete required runtime metrics in completion markers")
        tokens_per_second = positive_float(record.get("tokens_per_second"))
        batches_per_second = positive_float(record.get("batches_per_second"))
        if (tokens_per_second is None) != (batches_per_second is None):
            raise RuntimeError("Partially recorded throughput metrics in completion markers")
        if int(record["rows"]) > global_batch_size:
            saw_multi_batch = True
            if tokens_per_second is None or batches_per_second is None:
                raise RuntimeError("Incomplete throughput metrics for a multi-batch shard")
        if int(record["microbatch"]) != expected_microbatch:
            raise RuntimeError("Inconsistent runtime microbatch records")
    if not saw_multi_batch:
        raise RuntimeError("At least one multi-batch runtime record is required")


def verify_bundle_archive(path: Path, expected_names: Optional[Iterable[str]] = None) -> Sequence[str]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Bundle archive is missing or empty: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"Bundle contains duplicate archive names: {path}")
        if expected_names is not None and set(names) != set(expected_names):
            expected = set(expected_names)
            actual = set(names)
            raise RuntimeError(f"Bundle member mismatch: missing={expected - actual}, extra={actual - expected}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Bundle CRC check failed for {bad_member}")
    return names


def create_verified_archive(
    files: Sequence[Tuple[Path, str]],
    manifest: Mapping[str, Any],
    archive_path: Path,
    drive_archive_path: Path,
) -> Tuple[Path, Path]:
    normalized = []
    for source, archive_name in files:
        source = Path(source)
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Required bundle artifact is missing or empty: {source}")
        normalized.append((source, str(archive_name)))
    archive_names = [name for _, name in normalized]
    if len(archive_names) != len(set(archive_names)):
        raise RuntimeError("Bundle contains duplicate archive names")

    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("bundle_manifest.json", payload)
        for source, archive_name in normalized:
            archive.write(source, archive_name)
    os.replace(temp_archive, archive_path)
    expected_names = ["bundle_manifest.json", *archive_names]
    verify_bundle_archive(archive_path, expected_names)

    drive_archive_path = Path(drive_archive_path)
    drive_archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_drive = drive_archive_path.with_suffix(drive_archive_path.suffix + ".tmp")
    shutil.copy2(archive_path, temp_drive)
    os.replace(temp_drive, drive_archive_path)
    if drive_archive_path.stat().st_size != archive_path.stat().st_size:
        raise RuntimeError("Drive bundle copy has a different byte size")
    verify_bundle_archive(drive_archive_path, expected_names)
    drive_archive_path.with_suffix(".manifest.json").write_text(payload)
    return archive_path, drive_archive_path


@dataclass(frozen=True)
class WorkflowContext:
    runner: TargetedDropoutRunner
    olmo_dir: Path
    stage_root: Path
    config_drive: Path
    report_drive: Path
    subset_metadata: Path
    subset_full_scores: Path
    subset_manifest: Path
    source_rows_path: Path
    target_configs: Sequence[Mapping[str, Any]]
    shards: Sequence[Mapping[str, int]]
    producer_sha: str
    analysis_sha: str
    notebook_revision: str
    run_stage: str
    stage_rows: int
    num_samples: int
    seed: int
    tau64_cutoff: float
    global_batch_size: int
    shard_rows: int
    microbatch: int


class TargetedDropoutWorkflow:
    EXPECTED_SELECTED_FILES = 64
    EXPECTED_PAIRWISE_ROWS = 96
    EXPECTED_FULL_POOL_ROWS = 64
    EXPECTED_OVERLAP_ROWS = 480

    def __init__(self, context: WorkflowContext):
        self.context = context

    @property
    def runner(self) -> TargetedDropoutRunner:
        return self.context.runner

    def score_dirs_for(self, config_id: str, model_id: str) -> Sequence[Path]:
        return [
            self.runner.shard_output_dir(config_id, model_id, shard) / "score" for shard in self.context.shards
        ]

    def run_zero_dropout_smoke(
        self,
        control_config: Mapping[str, Any],
        prior_checkpoint: Path,
        books_checkpoint: Path,
        smoke_rows: int,
    ) -> Dict[str, float]:
        import pandas as pd

        shard = {"start": 0, "end": smoke_rows, "rows": smoke_rows, "data_start_step": 0}
        smoke_root = self.context.stage_root / "_smoke_and_benchmark" / "smoke"
        prior_scores = self.runner.run_score_once(
            control_config,
            "prior",
            prior_checkpoint,
            smoke_root / "prior",
            shard,
            self.context.microbatch,
        )
        books_scores = self.runner.run_score_once(
            control_config,
            "books",
            books_checkpoint,
            smoke_root / "books",
            shard,
            self.context.microbatch,
        )
        analysis = smoke_root / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        self.runner.run_logged(
            [
                sys.executable,
                self.context.olmo_dir / "scripts/21_dropout_uncertainty_metrics.py",
                "--prior-score-dir",
                prior_scores,
                "--conditional-score-dir",
                books_scores,
                "--output-dir",
                analysis,
                "--config-id",
                control_config["config_id"],
                "--metadata",
                self.context.subset_metadata,
                "--full-scores",
                self.context.subset_full_scores,
                "--num-samples",
                self.context.num_samples,
                "--max-rows",
                smoke_rows,
                "--dropout-target",
                "none",
                "--attention-dropout",
                0.0,
                "--residual-dropout",
                0.0,
                "--embedding-dropout",
                0.0,
                "--seed",
                self.context.seed,
                "--compress-npz",
                "--skip-parquet",
            ],
            analysis / "aggregate.log",
        )
        with np.load(analysis / f"mc_samples_{control_config['config_id']}.npz") as smoke:
            color = smoke["color_samples"]
            full = smoke["full_color_score"]
        if color.shape != (smoke_rows, self.context.num_samples) or not np.isfinite(color).all():
            raise RuntimeError(f"Zero-dropout smoke output is invalid: {color.shape}")
        spearman = float(pd.Series(color.mean(axis=1)).rank().corr(pd.Series(full).rank()))
        mean_std = float(color.std(axis=1, ddof=1).mean())
        if spearman < 0.90 or mean_std > 1e-5:
            raise RuntimeError(
                "Zero-dropout smoke gate failed: " f"Spearman={spearman:.6f}, mean MC std={mean_std:.8f}"
            )
        return {"spearman": spearman, "mean_mc_std": mean_std}

    def analysis_context_payload(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "stage": self.context.run_stage,
            "subset_fingerprint": self.runner.context.subset_fingerprint,
            "config": dict(config),
            "seed": self.context.seed,
            "num_samples": self.context.num_samples,
            "microbatch": self.context.microbatch,
            "raw_shard_fingerprints": {
                model_id: [
                    self.runner.shard_experiment_fingerprint(config, model_id, shard, self.context.microbatch)
                    for shard in self.context.shards
                ]
                for model_id in ("prior", "books")
            },
        }

    def validate_raw_grid(self, config: Mapping[str, Any]) -> None:
        indexes = {}
        for model_id in ("prior", "books"):
            model_indexes = []
            for shard in self.context.shards:
                output_dir = self.runner.shard_output_dir(config["config_id"], model_id, shard)
                if not self.runner.valid_score_output(
                    output_dir, config, model_id, shard, self.context.microbatch
                ):
                    raise RuntimeError(f"Invalid raw shard: {output_dir}")
                index = np.memmap(output_dir / "score" / "mmap_index.npy", dtype=np.int64, mode="r")
                model_indexes.append(np.asarray(index, dtype=np.int64))
            combined = np.concatenate(model_indexes)
            if not np.array_equal(np.sort(combined), np.arange(self.context.stage_rows, dtype=np.int64)):
                raise RuntimeError(f"{config['config_id']} {model_id}: shard indexes do not cover the subset")
            indexes[model_id] = combined
        if not np.array_equal(indexes["prior"], indexes["books"]):
            raise RuntimeError(f"{config['config_id']}: prior and Books row order differs")

    def validate_analysis(self, config: Mapping[str, Any]) -> None:
        import pandas as pd

        config_id = config["config_id"]
        analysis = self.context.stage_root / config_id / "analysis"
        strategy = analysis / "strategy"
        paths = {
            "npz": analysis / f"mc_samples_{config_id}.npz",
            "parquet": analysis / f"mc_samples_{config_id}.parquet",
            "manifest": analysis / f"mc_samples_{config_id}_manifest.json",
            "context": analysis / "analysis_context.json",
            "summary": analysis / "color_distribution_summary.parquet",
            "metrics": strategy / "strategy_sweep_metrics.csv",
            "overlap": strategy / "strategy_selection_overlap.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise FileNotFoundError("Missing or empty analysis files: " + ", ".join(missing))
        expected_context = self.analysis_context_payload(config)
        stored_context = json.loads(paths["context"].read_text())
        if (
            stored_context.get("fingerprint") != canonical_sha256(expected_context)
            or stored_context.get("experiment") != expected_context
        ):
            raise ValueError(f"{config_id}: stale analysis context")
        manifest = json.loads(paths["manifest"].read_text())
        expected_manifest = {
            "config_id": config_id,
            "rows": self.context.stage_rows,
            "num_samples": self.context.num_samples,
            "dropout_target": config["dropout_target"],
            "seed": self.context.seed,
            "coupled_masks": True,
            "attention_dropout": config["attention_dropout"],
            "residual_dropout": config["residual_dropout"],
            "embedding_dropout": config["embedding_dropout"],
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise ValueError(f"{config_id}: manifest {key}={manifest.get(key)!r}, expected {expected!r}")
        required_arrays = {
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
        with np.load(paths["npz"], allow_pickle=False) as raw:
            if not required_arrays.issubset(raw.files):
                raise ValueError(f"{config_id}: NPZ missing {sorted(required_arrays.difference(raw.files))}")
            score_index = raw["score_index"].astype(np.int64)
            if not np.array_equal(np.sort(score_index), np.arange(self.context.stage_rows, dtype=np.int64)):
                raise ValueError(f"{config_id}: score_index is not a complete permutation")
            prior = raw["prior_losses"]
            conditional = raw["conditional_losses"]
            color = raw["color_samples"]
            expected_shape = (self.context.stage_rows, self.context.num_samples)
            if prior.shape != expected_shape or conditional.shape != prior.shape or color.shape != prior.shape:
                raise ValueError(f"{config_id}: invalid sample tensor shapes")
            if not all(np.isfinite(values).all() for values in (prior, conditional, color)):
                raise ValueError(f"{config_id}: non-finite MC samples")
            if not np.allclose(color, conditional - prior, rtol=1e-5, atol=1e-6):
                raise ValueError(f"{config_id}: color samples do not equal conditional-prior")
            seq_idx = raw["seq_idx"].astype(np.int64)
            if len(seq_idx) != self.context.stage_rows or len(np.unique(seq_idx)) != len(seq_idx):
                raise ValueError(f"{config_id}: seq_idx is incomplete or duplicated")
            if not np.isfinite(raw["full_color_score"]).all():
                raise ValueError(f"{config_id}: non-finite full-color reference")
        summary = pd.read_parquet(paths["summary"])
        summary_columns = {
            "seq_idx",
            "score_index",
            "pool_name",
            "mean_color",
            "std_color",
            "full_color_score",
        }
        if len(summary) != self.context.stage_rows or not summary_columns.issubset(summary.columns):
            raise ValueError(f"{config_id}: invalid summary schema or row count")
        if not np.array_equal(
            np.sort(summary["score_index"].to_numpy(dtype=np.int64)),
            np.arange(self.context.stage_rows),
        ):
            raise ValueError(f"{config_id}: summary score_index is incomplete")
        metrics = pd.read_csv(paths["metrics"])
        pairwise = metrics[metrics["metric_scope"] == "pairwise"]
        full_pool = metrics[metrics["metric_scope"] == "full_pool"]
        if len(pairwise) != self.EXPECTED_PAIRWISE_ROWS or len(full_pool) != self.EXPECTED_FULL_POOL_ROWS:
            raise ValueError(f"{config_id}: unexpected metric rows pairwise={len(pairwise)} full={len(full_pool)}")
        expected_tasks = {
            "hp_vs_hn",
            "hp_vs_rn",
            "hp_vs_tn",
            "rp_vs_hn",
            "rp_vs_rn",
            "rp_vs_tn",
        }
        if set(pairwise["task_id"]) != expected_tasks or metrics["strategy"].nunique() != 16:
            raise ValueError(f"{config_id}: incomplete task or strategy grid")
        if not np.isfinite(pairwise["roc_auc"]).all() or not np.isfinite(full_pool["recall_vs_full"]).all():
            raise ValueError(f"{config_id}: non-finite strategy metrics")
        hp_mean = pairwise[(pairwise["strategy"] == "mean") & (pairwise["task_id"] == "hp_vs_hn")]
        rate_mean = full_pool[(full_pool["strategy"] == "mean") & np.isclose(full_pool["selection_rate"], 1 / 64)]
        if len(hp_mean) != 1 or len(rate_mean) != 1:
            raise ValueError(f"{config_id}: required mean-strategy rows are missing or duplicated")
        overlap = pd.read_csv(paths["overlap"])
        if len(overlap) != self.EXPECTED_OVERLAP_ROWS or not np.isfinite(overlap["jaccard"]).all():
            raise ValueError(f"{config_id}: invalid strategy overlap table")
        selected = sorted((strategy / "strategy_selected_indices").glob("*.npy"))
        expected_sizes = set(full_pool["n_selected"].astype(int).tolist())
        if len(selected) != self.EXPECTED_SELECTED_FILES:
            raise ValueError(
                f"{config_id}: expected {self.EXPECTED_SELECTED_FILES} selected arrays, " f"found {len(selected)}"
            )
        for selected_path in selected:
            values = np.load(selected_path, allow_pickle=False)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{config_id}: invalid selected array {selected_path.name}")
            if len(values) not in expected_sizes or len(np.unique(values)) != len(values):
                raise ValueError(f"{config_id}: invalid selected IDs in {selected_path.name}")
            if not np.isin(values, seq_idx).all():
                raise ValueError(f"{config_id}: selected IDs outside subset in {selected_path.name}")

    def _remove_invalid_analysis(self, analysis: Path) -> None:
        analysis = Path(analysis).resolve()
        stage_root = self.context.stage_root.resolve()
        if analysis.name != "analysis" or analysis.parent.parent != stage_root:
            raise RuntimeError(f"Refusing to remove non-isolated analysis path: {analysis}")
        print("removing invalid isolated analysis:", analysis)
        shutil.rmtree(analysis)

    def analyze(self) -> None:
        for config in self.context.target_configs:
            self.validate_raw_grid(config)
        for config in self.context.target_configs:
            config_id = config["config_id"]
            analysis = self.context.stage_root / config_id / "analysis"
            strategy = analysis / "strategy"
            try:
                self.validate_analysis(config)
                print("skip validated analysis:", config_id)
                continue
            except Exception as exc:
                print("analysis requires repair:", config_id, type(exc).__name__, exc)
            if analysis.exists():
                self._remove_invalid_analysis(analysis)
            analysis.mkdir(parents=True, exist_ok=True)
            self.runner.run_logged(
                [
                    sys.executable,
                    self.context.olmo_dir / "scripts/21_dropout_uncertainty_metrics.py",
                    "--prior-score-dir",
                    *self.score_dirs_for(config_id, "prior"),
                    "--conditional-score-dir",
                    *self.score_dirs_for(config_id, "books"),
                    "--output-dir",
                    analysis,
                    "--config-id",
                    config_id,
                    "--metadata",
                    self.context.subset_metadata,
                    "--full-scores",
                    self.context.subset_full_scores,
                    "--num-samples",
                    self.context.num_samples,
                    "--max-rows",
                    self.context.stage_rows,
                    "--dropout-target",
                    config["dropout_target"],
                    "--attention-dropout",
                    config["attention_dropout"],
                    "--residual-dropout",
                    config["residual_dropout"],
                    "--embedding-dropout",
                    config["embedding_dropout"],
                    "--seed",
                    self.context.seed,
                    "--compress-npz",
                ],
                analysis / "aggregate.log",
            )
            self.runner.run_logged(
                [
                    sys.executable,
                    self.context.olmo_dir / "scripts/22_dropout_strategy_sweep.py",
                    "--mc-samples",
                    analysis / f"mc_samples_{config_id}.npz",
                    "--summary",
                    analysis / "color_distribution_summary.parquet",
                    "--output-dir",
                    strategy,
                    "--tau64-cutoff",
                    self.context.tau64_cutoff,
                ],
                analysis / "strategy.log",
            )
            experiment = self.analysis_context_payload(config)
            (analysis / "analysis_context.json").write_text(
                json.dumps(
                    {"fingerprint": canonical_sha256(experiment), "experiment": experiment},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            self.validate_analysis(config)
        print("all analyses passed content validation")

    def status(self) -> Sequence[Dict[str, Any]]:
        rows = []
        for config in self.context.target_configs:
            for model_id in ("prior", "books"):
                for shard in self.context.shards:
                    output_dir = self.runner.shard_output_dir(config["config_id"], model_id, shard)
                    rows.append(
                        {
                            "config_id": config["config_id"],
                            "model_id": model_id,
                            "start": int(shard["start"]),
                            "end": int(shard["end"]),
                            "rows": int(shard["rows"]),
                            "valid": self.runner.valid_score_output(
                                output_dir,
                                config,
                                model_id,
                                shard,
                                self.context.microbatch,
                            ),
                            "output_dir": str(output_dir),
                        }
                    )
        return rows

    @staticmethod
    def _selected_mask(scores: np.ndarray, count: int) -> np.ndarray:
        mask = np.zeros(len(scores), dtype=bool)
        count = min(max(1, int(count)), len(scores))
        mask[np.argpartition(scores, count - 1)[:count]] = True
        return mask

    def _uncertainty_error_ratio(
        self, mean_color: np.ndarray, std_color: np.ndarray, pool_name: np.ndarray
    ) -> Tuple[float, float, float]:
        import pandas as pd

        tasks = (
            ("hard_positive", "hard_negative"),
            ("hard_positive", "random_negative"),
            ("hard_positive", "tail_negative"),
            ("random_positive", "hard_negative"),
            ("random_positive", "random_negative"),
            ("random_positive", "tail_negative"),
        )
        low_rates, high_rates = [], []
        for positive_pool, negative_pool in tasks:
            task_mask = (pool_name == positive_pool) | (pool_name == negative_pool)
            labels = pool_name[task_mask] == positive_pool
            scores = mean_color[task_mask]
            uncertainty = std_color[task_mask]
            if not len(scores) or not labels.any():
                continue
            errors = self._selected_mask(scores, int(labels.sum())) != labels
            deciles = pd.qcut(uncertainty, 10, labels=False, duplicates="drop")
            if pd.isna(deciles).all():
                continue
            deciles = np.asarray(deciles, dtype=np.int64)
            low_rates.append(float(errors[deciles == deciles.min()].mean()))
            high_rates.append(float(errors[deciles == deciles.max()].mean()))
        low = float(np.mean(low_rates)) if low_rates else float("nan")
        high = float(np.mean(high_rates)) if high_rates else float("nan")
        ratio = high / low if low and np.isfinite(low) else float("nan")
        return low, high, ratio

    def _runtime_records(self, config_id: str) -> Sequence[Dict[str, Any]]:
        records = []
        for model_id in ("prior", "books"):
            for shard in self.context.shards:
                marker_path = self.runner.shard_output_dir(config_id, model_id, shard) / "completed.json"
                marker = json.loads(marker_path.read_text())
                runtime = marker.get("runtime_metrics", {})
                records.append(
                    {
                        "config_id": config_id,
                        "model_id": model_id,
                        "start": int(shard["start"]),
                        "end": int(shard["end"]),
                        "rows": int(shard["rows"]),
                        "elapsed_seconds": marker.get("elapsed_seconds"),
                        "tokens_per_second": runtime.get("tokens_per_second"),
                        "batches_per_second": runtime.get("batches_per_second"),
                        "peak_gpu_memory_mb": runtime.get("peak_gpu_memory_mb"),
                        "microbatch": marker.get("microbatch"),
                    }
                )
        validate_runtime_records(records, self.context.global_batch_size, self.context.microbatch)
        return records

    def build_report(self) -> Dict[str, Any]:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary_rows = []
        all_runtime_rows = []
        for config in self.context.target_configs:
            config_id = config["config_id"]
            self.validate_analysis(config)
            analysis = self.context.stage_root / config_id / "analysis"
            with np.load(analysis / f"mc_samples_{config_id}.npz") as raw:
                color = raw["color_samples"]
                score_index = raw["score_index"].astype(np.int64)
                full_color = raw["full_color_score"]
                pool_name = raw["pool_name"].astype(str)
            expected_shape = (self.context.stage_rows, self.context.num_samples)
            if color.shape != expected_shape or not np.isfinite(color).all():
                raise RuntimeError(f"{config_id}: invalid raw MC sample tensor {color.shape}")
            if not np.array_equal(np.sort(score_index), np.arange(self.context.stage_rows, dtype=np.int64)):
                raise RuntimeError(f"{config_id}: score_index does not cover the fixed subset")

            mean_color = color.mean(axis=1)
            std_color = color.std(axis=1, ddof=1)
            spearman = float(pd.Series(mean_color).rank().corr(pd.Series(full_color).rank()))
            pearson = float(np.corrcoef(mean_color, full_color)[0, 1])
            metrics = pd.read_csv(analysis / "strategy" / "strategy_sweep_metrics.csv")
            pair_mean = metrics[(metrics["metric_scope"] == "pairwise") & (metrics["strategy"] == "mean")]
            full_mean = metrics[(metrics["metric_scope"] == "full_pool") & (metrics["strategy"] == "mean")]
            mean_auc = float(pair_mean["roc_auc"].mean())
            hp_auc = float(pair_mean.loc[pair_mean["task_id"] == "hp_vs_hn", "roc_auc"].iloc[0])
            recall = float(
                full_mean.loc[np.isclose(full_mean["selection_rate"], 1 / 64), "recall_vs_full"].iloc[0]
            )
            low_error, high_error, error_ratio = self._uncertainty_error_ratio(mean_color, std_color, pool_name)
            q05 = np.quantile(color, 0.05, axis=1)
            q95 = np.quantile(color, 0.95, axis=1)
            uncertain_fraction = float(
                ((q05 <= self.context.tau64_cutoff) & (q95 > self.context.tau64_cutoff)).mean()
            )

            if spearman <= 0.20:
                decision, reason = "stop", "Spearman is at or below the 0.20 hard gate"
            elif recall <= 2 / 64:
                decision, reason = "stop", "1/64 recall is within 2x random overlap"
            elif mean_auc <= 0.55 and hp_auc <= 0.55:
                decision, reason = "stop", "pairwise AUC remains near chance"
            elif spearman >= 0.50 and (
                mean_auc >= 0.75 or hp_auc >= 0.60 or recall >= 0.10 or error_ratio >= 1.25
            ):
                decision, reason = (
                    "promote",
                    "rank is preserved and at least one promotion target is met",
                )
            else:
                decision, reason = (
                    "review",
                    "hard gates pass but promotion targets are not clearly met",
                )

            runtimes = pd.DataFrame(self._runtime_records(config_id))
            all_runtime_rows.extend(runtimes.to_dict("records"))
            summary_rows.append(
                {
                    "config_id": config_id,
                    "dropout_target": config["dropout_target"],
                    "attention_dropout": config["attention_dropout"],
                    "residual_dropout": config["residual_dropout"],
                    "embedding_dropout": config["embedding_dropout"],
                    "rows": self.context.stage_rows,
                    "K": self.context.num_samples,
                    "spearman_mean_vs_full": spearman,
                    "pearson_mean_vs_full": pearson,
                    "mean_pairwise_auc": mean_auc,
                    "hp_vs_hn_auc": hp_auc,
                    "recall_vs_full_1_64": recall,
                    "mean_mc_std": float(std_color.mean()),
                    "low_uncertainty_error_rate": low_error,
                    "high_uncertainty_error_rate": high_error,
                    "error_rate_ratio_high_vs_low": error_ratio,
                    "triage_uncertain_fraction": uncertain_fraction,
                    "runtime_seconds": float(runtimes["elapsed_seconds"].sum()),
                    "tokens_per_second_mean": float(runtimes["tokens_per_second"].dropna().mean()),
                    "batches_per_second_mean": float(runtimes["batches_per_second"].dropna().mean()),
                    "peak_gpu_memory_mb": float(runtimes["peak_gpu_memory_mb"].max()),
                    "global_batch_size": self.context.global_batch_size,
                    "shard_rows": self.context.shard_rows,
                    "microbatch": self.context.microbatch,
                    "shard_count": int(len(runtimes)),
                    "prior_shard_count": int((runtimes["model_id"] == "prior").sum()),
                    "conditional_shard_count": int((runtimes["model_id"] == "books").sum()),
                    "promotion_decision": decision,
                    "decision_reason": reason,
                }
            )

        stage_summary = pd.DataFrame(summary_rows)
        stage_summary.to_csv(self.context.stage_root / "stage_summary.csv", index=False)
        pd.DataFrame(all_runtime_rows).to_csv(self.context.stage_root / "runtime_summary.csv", index=False)
        promoted = stage_summary[stage_summary["promotion_decision"] == "promote"].sort_values(
            ["spearman_mean_vs_full", "mean_pairwise_auc", "recall_vs_full_1_64"],
            ascending=False,
        )
        promoted_ids = promoted["config_id"].tolist()
        recommended_stage_c = promoted_ids[:2] if self.context.run_stage == "stage_b_100k" else []
        revisions = {
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
        }
        acceptance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage_id": self.context.run_stage,
            "stage_rows": self.context.stage_rows,
            "num_samples": self.context.num_samples,
            "olmo_sha": self.context.producer_sha,
            **revisions,
            "subset_manifest": str(self.context.subset_manifest),
            "stage_passed": bool(promoted_ids),
            "promoted_config_ids": promoted_ids,
            "recommended_stage_c_config_ids": recommended_stage_c,
            "hard_gate_thresholds": {
                "spearman_gt": 0.20,
                "recall_1_64_gt": 2 / 64,
                "near_chance_auc_max": 0.55,
            },
            "config_decisions": stage_summary[["config_id", "promotion_decision", "decision_reason"]].to_dict(
                "records"
            ),
            "final_interpretation_status": (
                "ready_for_local_cross_baseline_report"
                if self.context.run_stage == "stage_c_500k"
                else "Stage C allowed only for recommended or explicitly promoted IDs"
            ),
        }
        (self.context.stage_root / "stage_acceptance.json").write_text(
            json.dumps(acceptance, indent=2, sort_keys=True) + "\n"
        )
        comparison = {
            "deterministic_full_scores": str(self.context.subset_full_scores),
            "targeted_stage_root": str(self.context.stage_root),
            "local_extract_root": (f"results/dropout-uncertainty-targeted/{self.runner.context.subset_id}"),
            "local_report_command": (
                "python scripts/22_targeted_dropout_cross_stage_report.py "
                "--stage-b-root results/dropout-uncertainty-targeted/stage_b_100k "
                + (
                    "--stage-c-root results/dropout-uncertainty-targeted/stage_c_500k "
                    if self.context.run_stage == "stage_c_500k"
                    else ""
                )
                + "--output-dir reports/dropout-uncertainty-targeted-ladder-final"
            ),
            "required_final_conclusions": [
                "standalone selection",
                "cascade candidate generation",
                "triage routing",
                "uncertainty diagnostics",
                "none of the above",
            ],
            **revisions,
        }
        (self.context.stage_root / "final_comparison_contract.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )

        self.context.report_drive.mkdir(parents=True, exist_ok=True)
        plot_frame = stage_summary.set_index("config_id")
        figure, axis = plt.subplots(figsize=(max(8, len(plot_frame) * 1.35), 5))
        plot_frame[
            [
                "spearman_mean_vs_full",
                "mean_pairwise_auc",
                "hp_vs_hn_auc",
                "recall_vs_full_1_64",
            ]
        ].plot(kind="bar", ax=axis)
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axis.set_ylabel("Metric value")
        axis.set_xlabel("")
        axis.set_title(f"{self.context.run_stage}: targeted-dropout quality metrics")
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        quality_figure = self.context.report_drive / "quality_metrics.png"
        figure.savefig(quality_figure, dpi=160)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        plot_frame["runtime_seconds"].div(3600).plot(kind="bar", ax=axes[0], color="#3b6ea8")
        plot_frame["tokens_per_second_mean"].plot(kind="bar", ax=axes[1], color="#c05a47")
        axes[0].set_title("Total scoring runtime")
        axes[0].set_ylabel("GPU hours")
        axes[1].set_title("Mean device throughput")
        axes[1].set_ylabel("Tokens / second")
        for axis in axes:
            axis.set_xlabel("")
        figure.tight_layout()
        runtime_figure = self.context.report_drive / "runtime_metrics.png"
        figure.savefig(runtime_figure, dpi=160)
        plt.close(figure)

        decisions = [
            f"- `{row.config_id}`: **{row.promotion_decision}** - {row.decision_reason}"
            for row in stage_summary.itertuples(index=False)
        ]
        report_md = self.context.report_drive / "stage_report.md"
        report_md.write_text(
            f"# Targeted Dropout Ladder: {self.context.run_stage}\n\n"
            f"- Rows: {self.context.stage_rows:,}\n- MC samples: {self.context.num_samples}\n"
            f"- Producer commit: `{self.context.producer_sha}`\n"
            f"- Analysis commit: `{self.context.analysis_sha}`\n"
            f"- Notebook revision: `{self.context.notebook_revision}`\n"
            f"- Stage passed: {acceptance['stage_passed']}\n\n## Decisions\n\n"
            + "\n".join(decisions)
            + "\n\n## Summary\n\n```csv\n"
            + stage_summary.to_csv(index=False, float_format="%.6g")
            + "```\n\n## Figures\n\n![Quality metrics](quality_metrics.png)\n\n"
            + "![Runtime metrics](runtime_metrics.png)\n\n## Local Handoff\n\n```bash\n"
            + comparison["local_report_command"]
            + "\n```\n",
            encoding="utf-8",
        )
        report_html = self.context.report_drive / "stage_report.html"
        report_html.write_text(
            '<!doctype html><meta charset="utf-8"><title>Targeted dropout stage report</title>'
            "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}"
            "table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border:1px solid #ccc;padding:5px}img{max-width:100%;height:auto}</style>"
            f"<h1>Targeted Dropout Ladder: {self.context.run_stage}</h1>"
            f"<p>Rows: {self.context.stage_rows:,}; MC samples: {self.context.num_samples}; "
            f"stage passed: {acceptance['stage_passed']}</p>"
            + stage_summary.to_html(index=False, float_format=lambda value: f"{value:.6g}")
            + '<h2>Quality</h2><img src="quality_metrics.png" alt="Quality metrics">'
            + '<h2>Runtime</h2><img src="runtime_metrics.png" alt="Runtime metrics">'
            + f"<h2>Local handoff</h2><pre>{comparison['local_report_command']}</pre>",
            encoding="utf-8",
        )
        report_manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage_id": self.context.run_stage,
            **revisions,
            "artifacts": [
                report_md.name,
                report_html.name,
                quality_figure.name,
                runtime_figure.name,
            ],
        }
        (self.context.report_drive / "report_manifest.json").write_text(
            json.dumps(report_manifest, indent=2, sort_keys=True) + "\n"
        )
        return {
            "stage_summary": stage_summary,
            "acceptance": acceptance,
            "comparison_contract": comparison,
        }

    def build_bundle(self, archive_path: Path, drive_archive_path: Path) -> Dict[str, Any]:
        files = []

        def add(path: Path, archive_name: str) -> None:
            files.append((Path(path), archive_name))

        for path, archive_name in (
            (self.context.subset_manifest, "subset_manifest.json"),
            (self.context.source_rows_path, "subset_source_rows.npy"),
            (self.runner.context.run_state_path, "run_state.json"),
            (self.context.stage_root / "shard_plan.json", "shard_plan.json"),
            (self.context.stage_root / "stage_summary.csv", "stage_summary.csv"),
            (self.context.stage_root / "runtime_summary.csv", "runtime_summary.csv"),
            (self.context.stage_root / "stage_acceptance.json", "stage_acceptance.json"),
            (
                self.context.stage_root / "final_comparison_contract.json",
                "final_comparison_contract.json",
            ),
        ):
            add(path, archive_name)
        for name in (
            "stage_report.md",
            "stage_report.html",
            "quality_metrics.png",
            "runtime_metrics.png",
            "report_manifest.json",
        ):
            add(self.context.report_drive / name, f"report/{name}")

        for config in self.context.target_configs:
            self.validate_analysis(config)
            config_id = config["config_id"]
            analysis = self.context.stage_root / config_id / "analysis"
            strategy = analysis / "strategy"
            base = f"analysis/{config_id}"
            for name in (
                "aggregate.log",
                "strategy.log",
                "analysis_context.json",
                "color_distribution_summary.parquet",
                f"mc_samples_{config_id}.npz",
                f"mc_samples_{config_id}.parquet",
                f"mc_samples_{config_id}_manifest.json",
            ):
                add(analysis / name, f"{base}/{name}")
            for name in ("strategy_sweep_metrics.csv", "strategy_selection_overlap.csv"):
                add(strategy / name, f"{base}/strategy/{name}")
            selected = sorted((strategy / "strategy_selected_indices").glob("*.npy"))
            if len(selected) != self.EXPECTED_SELECTED_FILES:
                raise RuntimeError(f"{config_id}: expected {self.EXPECTED_SELECTED_FILES} selected arrays")
            for selected_path in selected:
                add(
                    selected_path,
                    f"{base}/strategy/strategy_selected_indices/{selected_path.name}",
                )
            for model_id in ("prior", "books"):
                for shard in self.context.shards:
                    output_dir = self.runner.shard_output_dir(config_id, model_id, shard)
                    shard_name = f"shard_{int(shard['start']):06d}_{int(shard['end']):06d}"
                    raw_base = f"raw_score_provenance/{config_id}/{model_id}/{shard_name}"
                    add(output_dir / "completed.json", f"{raw_base}/completed.json")
                    add(output_dir.with_suffix(".log"), f"{raw_base}.log")
                    add(output_dir / "config.yaml", f"{raw_base}/config.yaml")
                    runtime_name = (
                        f"{config_id}_{model_id}_{int(shard['start']):06d}_" f"{int(shard['end']):06d}.yaml"
                    )
                    add(self.context.config_drive / runtime_name, f"runtime_configs/{runtime_name}")

        comparison = json.loads((self.context.stage_root / "final_comparison_contract.json").read_text())
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage_id": self.context.run_stage,
            "stage_rows": self.context.stage_rows,
            "num_samples": self.context.num_samples,
            "producer_sha": self.context.producer_sha,
            "analysis_sha": self.context.analysis_sha,
            "notebook_revision": self.context.notebook_revision,
            "target_configs": [dict(config) for config in self.context.target_configs],
            "stage_root": str(self.context.stage_root),
            "files": [
                {
                    "source": str(source),
                    "archive": archive_name,
                    "bytes": source.stat().st_size if source.exists() else None,
                }
                for source, archive_name in files
            ],
            "excludes": ["model checkpoints", "token arrays", "raw score memmaps"],
            "local_extract_root": comparison["local_extract_root"],
            "local_report_command": comparison["local_report_command"],
            "drive_fallback": str(drive_archive_path),
        }
        local_archive, drive_archive = create_verified_archive(files, manifest, archive_path, drive_archive_path)
        return {
            "archive_path": local_archive,
            "drive_archive_path": drive_archive,
            "file_count": len(files) + 1,
            "manifest": manifest,
        }


__all__ = [
    "PreparedSubset",
    "ScoringContext",
    "SubsetContext",
    "TargetedDropoutRunner",
    "TargetedDropoutWorkflow",
    "WorkflowContext",
    "build_shard_plan",
    "canonical_sha256",
    "create_verified_archive",
    "positive_float",
    "prepare_fixed_subset",
    "sampled_file_identity",
    "select_fastest_benchmark",
    "validate_runtime_records",
    "verify_bundle_archive",
]
