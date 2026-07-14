"""Runtime helpers for the targeted-dropout Stage B/C Colab notebook."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
    olmo_sha: str
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
            "schema_version": 3,
            "olmo_sha": self.context.olmo_sha,
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
            "olmo_sha": self.context.olmo_sha,
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
            "olmo_sha": self.context.olmo_sha,
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


__all__ = [
    "ScoringContext",
    "TargetedDropoutRunner",
    "canonical_sha256",
    "positive_float",
    "select_fastest_benchmark",
]
