#!/usr/bin/env python
"""Colab orchestration helpers for the 410M score-pool training experiment."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile


EXPECTED_ROWS = 100_000
SEQUENCE_LENGTH = 512
TOKENS_PER_RUN = 102_236_160
DEFAULT_GLOBAL_BATCH_SIZE = 256


@dataclass(frozen=True)
class BundleContext:
    experiment: str
    train_dataset: str
    run_ids: tuple[str, ...]
    train_data_drive: Path
    results_drive: Path
    reports_drive: Path
    runtime_config_dir: Path
    eval_manifest: Path
    analysis_helper: Path
    orchestration_helper: Path
    ablation_producer_sha: str
    olmo_producer_sha: str
    olmo_analysis_sha: str
    notebook_revision: str


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_runtime_configs(
    *,
    config_map: dict[str, Path],
    train_data_dir: Path,
    checkpoints_dir: Path,
    runtime_config_dir: Path,
    books_eval: Path,
    c4_eval: Path,
    device_eval_batch_size: int,
    eval_subset_num_batches: int,
) -> dict[str, Path]:
    from omegaconf import OmegaConf

    runtime_config_dir.mkdir(parents=True, exist_ok=True)

    def evaluator(label: str, path: Path) -> dict[str, object]:
        return {
            "label": label,
            "type": "lm",
            "data": {
                "paths": [str(path)],
                "drop_last": True,
                "num_workers": 2,
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "memmap_dtype": "uint16",
            },
            "device_eval_batch_size": device_eval_batch_size,
            "subset_num_batches": eval_subset_num_batches,
        }

    runtime_config_map: dict[str, Path] = {}
    for run_id, template_path in config_map.items():
        if not template_path.is_file():
            raise FileNotFoundError(template_path)
        train_tokens = train_data_dir / run_id / "train_tokens.npy"
        if not train_tokens.is_file():
            raise FileNotFoundError(train_tokens)
        cfg = OmegaConf.load(template_path)
        cfg.run_name = f"score_pool_410m_100m_{run_id.removesuffix('_100k')}"
        cfg.data.paths = [str(train_tokens)]
        cfg.save_folder = str(checkpoints_dir / run_id)
        cfg.global_train_batch_size = DEFAULT_GLOBAL_BATCH_SIZE
        cfg.device_train_batch_size = DEFAULT_GLOBAL_BATCH_SIZE
        cfg.max_duration = "2ep"
        cfg.save_interval = 390
        cfg.save_num_checkpoints_to_keep = 2
        cfg.save_num_unsharded_checkpoints_to_keep = 0
        cfg.eval_interval = 78
        cfg.eval_on_load = False
        cfg.device_eval_batch_size = device_eval_batch_size
        cfg.eval_subset_num_batches = eval_subset_num_batches
        cfg.evaluators = [
            evaluator("books_val", books_eval),
            evaluator("c4_val_proxy", c4_eval),
        ]
        out_path = runtime_config_dir / f"{run_id}.yaml"
        OmegaConf.save(cfg, out_path)
        runtime_config_map[run_id] = out_path
        print("wrote:", out_path)
    return runtime_config_map


def generate_seed_configs(
    *,
    runtime_config_map: dict[str, Path],
    seeds: Iterable[int],
    base_run_ids: Iterable[str],
    checkpoints_dir: Path,
    runtime_config_dir: Path,
) -> list[str]:
    from omegaconf import OmegaConf

    seed_run_ids = []
    for seed in seeds:
        for base_run_id in base_run_ids:
            run_id = base_run_id.removesuffix("_100k") + f"_seed{seed}_100k"
            cfg = OmegaConf.load(runtime_config_map[base_run_id])
            cfg.seed = seed
            cfg.run_name = f"{cfg.run_name}_seed{seed}"
            cfg.save_folder = str(checkpoints_dir / run_id)
            out_path = runtime_config_dir / f"{run_id}.yaml"
            OmegaConf.save(cfg, out_path)
            runtime_config_map[run_id] = out_path
            seed_run_ids.append(run_id)
            print("wrote:", out_path, "data:", cfg.data.paths[0], "seed:", cfg.seed)
    return seed_run_ids


def generate_smoke_configs(
    *,
    runtime_config_map: dict[str, Path],
    run_ids: Iterable[str],
    smoke_dir: Path,
    smoke_config_dir: Path,
) -> dict[str, Path]:
    from omegaconf import OmegaConf

    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_config_dir.mkdir(parents=True, exist_ok=True)
    smoke_config_map: dict[str, Path] = {}
    for run_id in run_ids:
        cfg = OmegaConf.load(runtime_config_map[run_id])
        cfg.run_name = f"smoke_{run_id}"
        cfg.save_folder = str(smoke_dir / run_id)
        cfg.max_duration = 5
        cfg.eval_interval = 5
        cfg.save_interval = 5
        cfg.console_log_interval = 1
        cfg.save_num_checkpoints_to_keep = 1
        for evaluator in cfg.evaluators:
            evaluator.subset_num_batches = 2
        out_path = smoke_config_dir / f"{run_id}_smoke.yaml"
        OmegaConf.save(cfg, out_path)
        smoke_config_map[run_id] = out_path
        print("wrote:", out_path)
    return smoke_config_map


def generate_microbatch_configs(
    *,
    base_config_path: Path,
    candidates: Iterable[int],
    benchmark_id: str,
    checkpoints_dir: Path,
    config_dir: Path,
) -> dict[int, Path]:
    from copy import deepcopy

    from omegaconf import OmegaConf

    config_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = OmegaConf.load(base_config_path)
    config_map: dict[int, Path] = {}
    for microbatch in candidates:
        cfg = deepcopy(base_cfg)
        cfg.run_name = f"microbatch_{benchmark_id}_{microbatch}"
        cfg.save_folder = str(checkpoints_dir / "benchmarks" / benchmark_id / f"microbatch_{microbatch}")
        cfg.max_duration = 20
        cfg.eval_interval = 100_000
        cfg.evaluators = []
        cfg.save_interval = 20
        cfg.save_num_checkpoints_to_keep = 1
        out_path = config_dir / f"microbatch_{benchmark_id}_{microbatch}.yaml"
        OmegaConf.save(cfg, out_path)
        config_map[microbatch] = out_path
        print("wrote:", out_path)
    return config_map


def stage_local_training_data(source_dir: Path, destination_dir: Path) -> None:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if destination_dir.parent != Path("/content"):
        raise ValueError(f"Local training-data destination must be directly under /content: {destination_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    staging_dir = destination_dir.with_name(f"{destination_dir.name}.partial")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    copied_bytes = 0
    last_heartbeat = time.monotonic()
    print(f"staging {len(files)} files ({total_bytes / 1_000_000_000:.2f} GB) to {destination_dir}")
    for index, source in enumerate(files, start=1):
        relative = source.relative_to(source_dir)
        destination = staging_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes += source.stat().st_size
        now = time.monotonic()
        if now - last_heartbeat >= 30 or index == len(files):
            print(
                f"staging progress: {index}/{len(files)} files; "
                f"{copied_bytes / 1_000_000_000:.2f}/{total_bytes / 1_000_000_000:.2f} GB",
                flush=True,
            )
            last_heartbeat = now
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    staging_dir.replace(destination_dir)
    print("local training data ready:", destination_dir)


def run_logged(
    cmd: Sequence[object],
    log_path: Path,
    *,
    cwd: Path,
    pythonpath: Path,
    append: bool = False,
    heartbeat_seconds: float = 30.0,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pythonpath)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    print("command:", " ".join(str(part) for part in cmd), flush=True)
    started = time.monotonic()
    with log_path.open(mode, encoding="utf-8") as log:
        if append:
            log.write("\n\n===== RESUMED RUN =====\n")
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines: Queue[str | None] = Queue()

        def drain_stdout() -> None:
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        Thread(target=drain_stdout, daemon=True).start()
        stream_closed = False
        while not stream_closed:
            try:
                line = lines.get(timeout=heartbeat_seconds)
            except Empty:
                elapsed = time.monotonic() - started
                print(
                    f"still running; granular progress unavailable; elapsed={elapsed:.1f}s",
                    flush=True,
                )
                continue
            if line is None:
                stream_closed = True
                continue
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = proc.wait()
    elapsed = time.monotonic() - started
    print(f"command complete; elapsed_seconds={elapsed:.2f}; log={log_path}", flush=True)
    if return_code != 0:
        tail = log_path.read_text(errors="ignore")[-4000:]
        raise RuntimeError(f"Command failed with exit code {return_code}. Log tail:\n{tail}")


def production_log_status(log_path: Path, expected_eval_points: int = 10) -> dict[str, object]:
    if not log_path.exists():
        return {
            "exists": False,
            "training_complete": False,
            "books_eval_points": 0,
            "c4_eval_points": 0,
            "has_required_eval_curve": False,
        }
    text = log_path.read_text(errors="ignore")
    books = text.count("eval/books_val/CrossEntropyLoss")
    c4 = text.count("eval/c4_val_proxy/CrossEntropyLoss")
    complete = "Training complete" in text[-30_000:]
    return {
        "exists": True,
        "training_complete": complete,
        "books_eval_points": books,
        "c4_eval_points": c4,
        "has_required_eval_curve": books >= expected_eval_points and c4 >= expected_eval_points,
    }


def run_training(
    *,
    run_id: str,
    config_path: Path,
    checkpoint_dir: Path,
    log_path: Path,
    olmo_dir: Path,
    microbatch: int,
    expected_eval_points: int = 10,
) -> str:
    status = production_log_status(log_path, expected_eval_points)
    if status["training_complete"]:
        if status["has_required_eval_curve"]:
            print(f"skipping completed run with eval curves: {run_id}")
            return "skipped"
        raise RuntimeError(
            f"{run_id} is complete but missing required eval curves: {status}. "
            "Use a fresh experiment/output directory for a full eval-enabled rerun."
        )

    args = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/train.py",
        str(config_path),
        f"--device_train_microbatch_size={microbatch}",
    ]
    checkpoint_dirs = sorted(checkpoint_dir.glob("step*")) if checkpoint_dir.exists() else []
    if checkpoint_dirs:
        args.append(f"--load_path=${{path.last_checkpoint:{checkpoint_dir}}}")
    elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise RuntimeError(
            f"{checkpoint_dir} exists but has no step checkpoints. Inspect before overwriting."
        )
    run_logged(
        args,
        log_path,
        cwd=olmo_dir,
        pythonpath=olmo_dir,
        append=log_path.exists(),
    )
    return "completed"


def assert_eval_curves_ready(
    run_ids: Iterable[str],
    results_dir: Path,
    expected_eval_points: int = 10,
) -> None:
    problems = []
    for run_id in run_ids:
        status = production_log_status(results_dir / f"{run_id}.log", expected_eval_points)
        ok = bool(status["training_complete"] and status["has_required_eval_curve"])
        print(run_id, status, "ok=", ok)
        if not ok:
            problems.append((run_id, status))
    if problems:
        raise RuntimeError(f"Missing required eval learning curves; rerun production. Problems: {problems}")


def validate_smoke_logs(run_ids: Iterable[str], smoke_dir: Path) -> None:
    required_markers = [
        "train/CrossEntropyLoss",
        "eval/books_val/CrossEntropyLoss",
        "eval/c4_val_proxy/CrossEntropyLoss",
        "Training complete",
    ]
    nonfinite_metric = re.compile(r"(?i)(?:train|eval)/[^=\s]+=(?:nan|[+-]?inf)(?:\s|$)")
    problems: list[tuple[str, object]] = []
    for run_id in run_ids:
        log_path = smoke_dir / f"{run_id}.log"
        if not log_path.is_file():
            problems.append((run_id, "missing log"))
            continue
        text = log_path.read_text(errors="ignore")
        missing = [marker for marker in required_markers if marker not in text]
        nonfinite = sorted(set(nonfinite_metric.findall(text)))
        if missing or nonfinite:
            problems.append((run_id, {"missing": missing, "nonfinite": nonfinite}))
            print(text[-4000:])
            continue
        print("smoke ok:", run_id, log_path.stat().st_size)
    if problems:
        raise RuntimeError(f"Smoke validation failed: {problems}")


def parse_microbatch_results(
    *,
    candidates: Iterable[int],
    benchmark_dir: Path,
    peak_limit_mb: float,
    total_runs: int,
) -> tuple[list[dict[str, object]], int]:
    token_pattern = re.compile(r"throughput/device/tokens_per_second=([0-9.,]+)")
    memory_pattern = re.compile(r"System/Peak GPU Memory \(MB\)=([0-9.,]+)")
    results: list[dict[str, object]] = []
    for microbatch in candidates:
        log_path = benchmark_dir / f"microbatch_{microbatch}.log"
        if not log_path.exists():
            print("missing:", log_path)
            continue
        text = log_path.read_text(errors="ignore")
        speeds = [float(match.group(1).replace(",", "")) for match in token_pattern.finditer(text)]
        peaks = [float(match.group(1).replace(",", "")) for match in memory_pattern.finditer(text)]
        if not speeds:
            print("no throughput parsed for", microbatch)
            continue
        median_tps = statistics.median(speeds)
        peak_mb = max(peaks) if peaks else float("nan")
        per_run_hours = TOKENS_PER_RUN / median_tps / 3600
        result = {
            "microbatch": microbatch,
            "completed": "Training complete" in text[-30_000:],
            "observed_rates": len(speeds),
            "median_tps": median_tps,
            "peak_mb": peak_mb,
            "eta_per_run_hours": per_run_hours,
            "eta_total_hours": total_runs * per_run_hours,
        }
        results.append(result)
        print(result)
    stable = [
        result
        for result in results
        if result["completed"]
        and result["observed_rates"] >= 2
        and float(result["peak_mb"]) <= peak_limit_mb
    ]
    recommended = int(max(stable, key=lambda result: float(result["median_tps"]))["microbatch"]) if stable else 32
    return results, recommended


def write_eval_data(
    *,
    books_eval: Path,
    c4_eval: Path,
    eval_manifest: Path,
    eval_sequences: int,
    sequence_length: int,
    device_eval_batch_size: int,
    eval_subset_num_batches: int,
) -> None:
    import numpy as np

    expected_bytes = eval_sequences * sequence_length * np.dtype(np.uint16).itemsize

    def write_first_chunks(src_path: Path, dst_path: Path) -> None:
        if dst_path.exists() and dst_path.stat().st_size == expected_bytes:
            print("exists:", dst_path)
            return
        raw = np.memmap(src_path, dtype=np.uint16, mode="r")
        needed = eval_sequences * sequence_length
        if raw.size < needed:
            raise RuntimeError(f"{src_path} has only {raw.size:,} uint16 tokens, need {needed:,}")
        output = np.memmap(dst_path, dtype=np.uint16, mode="w+", shape=(eval_sequences, sequence_length))
        output[:] = raw[:needed].reshape(eval_sequences, sequence_length)
        output.flush()
        print("wrote:", dst_path, dst_path.stat().st_size)

    if not books_eval.exists() or books_eval.stat().st_size != expected_bytes:
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id="hlzhang109/CoLoR-filter",
                repo_type="model",
                filename="downstream_data/books_val/books_val.npy",
            )
        )
        write_first_chunks(source, books_eval)
    else:
        print("exists:", books_eval)

    if not c4_eval.exists() or c4_eval.stat().st_size != expected_bytes:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("allenai/eleuther-ai-gpt-neox-20b-pii-special")
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        token_buffer: list[int] = []
        output = np.memmap(c4_eval, dtype=np.uint16, mode="w+", shape=(eval_sequences, sequence_length))
        row = 0
        stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for example in stream:
            text = example.get("text") or ""
            if not text.strip():
                continue
            token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            token_buffer.append(eos)
            while len(token_buffer) >= sequence_length and row < eval_sequences:
                chunk = token_buffer[:sequence_length]
                del token_buffer[:sequence_length]
                if max(chunk) >= 65536:
                    raise RuntimeError("Token id does not fit uint16")
                output[row] = np.asarray(chunk, dtype=np.uint16)
                row += 1
            if row >= eval_sequences:
                break
        if row != eval_sequences:
            raise RuntimeError(f"Only wrote {row} C4 eval rows")
        output.flush()
        print("wrote:", c4_eval, c4_eval.stat().st_size)
    else:
        print("exists:", c4_eval)

    manifest = {
        "sequence_length": sequence_length,
        "eval_sequences": eval_sequences,
        "device_eval_batch_size": device_eval_batch_size,
        "eval_subset_num_batches": eval_subset_num_batches,
        "books_val": {
            "label": "books_val",
            "source": "hlzhang109/CoLoR-filter:downstream_data/books_val/books_val.npy",
            "path": str(books_eval),
        },
        "c4_val_proxy": {
            "label": "c4_val_proxy",
            "source": "allenai/c4 en validation streaming split",
            "path": str(c4_eval),
        },
    }
    eval_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(eval_manifest.read_text())


def validate_eval_data(paths: Iterable[Path], eval_sequences: int, sequence_length: int) -> None:
    import numpy as np

    expected_bytes = eval_sequences * sequence_length * np.dtype(np.uint16).itemsize
    for path in paths:
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise AssertionError((path, path.stat().st_size if path.exists() else None, expected_bytes))
        array = np.memmap(path, dtype=np.uint16, mode="r", shape=(eval_sequences, sequence_length))
        if array.shape != (eval_sequences, sequence_length):
            raise AssertionError((path, array.shape))
        print("validated:", path, array.shape, array.dtype)


def _bundle_base_run_id(run_id: str) -> str:
    match = re.match(r"^(.+)_seed\d+_100k$", run_id)
    return f"{match.group(1)}_100k" if match else run_id


def build_bundle(context: BundleContext, zip_path: Path, drive_path: Path) -> dict[str, object]:
    figures_dir = context.reports_drive / "figures"
    required_files = [
        context.results_drive / "train_metrics_from_logs.csv",
        context.results_drive / "eval_metrics_from_logs.csv",
        context.results_drive / "checkpoint_save_times.csv",
        context.results_drive / "throughput_comparison.csv",
        context.results_drive / "selection_diagnostics.csv",
        context.results_drive / "overlap_jaccard.csv",
        context.results_drive / "checkpoint_manifest.json",
        context.reports_drive / "report.md",
        context.reports_drive / "report.html",
        figures_dir / "eval_loss_books_by_run.png",
        figures_dir / "eval_loss_books_hard_oracle_vs_cascade_seeds.png",
        context.eval_manifest,
    ]
    required_files.extend(context.results_drive / f"{run_id}.log" for run_id in context.run_ids)
    for run_id in sorted({_bundle_base_run_id(run_id) for run_id in context.run_ids}):
        required_files.extend(
            [
                context.train_data_drive / run_id / "train_meta.parquet",
                context.train_data_drive / run_id / "manifest.json",
            ]
        )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required bundle inputs:\n" + "\n".join(missing))

    files: list[dict[str, object]] = []

    def add_file(archive: ZipFile, source: Path, destination: str) -> None:
        archive.write(source, destination)
        files.append({"src": str(source), "dst": destination, "bytes": source.stat().st_size})

    def add_tree(archive: ZipFile, source_dir: Path, destination_dir: str, patterns: list[str]) -> None:
        if not source_dir.exists():
            return
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_dir).as_posix()
            if relative.endswith("train_tokens.npy"):
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                add_file(archive, source, f"{destination_dir}/{relative}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        add_tree(
            archive,
            context.results_drive,
            f"color-filter-ablation/results/{context.experiment}",
            ["*.log", "*.csv", "*.json", "*.jsonl"],
        )
        add_tree(
            archive,
            context.reports_drive,
            f"color-filter-ablation/reports/{context.experiment}",
            ["report.md", "report.html", "figures/*.png", "quick_checks/*.png"],
        )
        add_tree(
            archive,
            context.train_data_drive,
            f"color-filter-ablation/data/{context.train_dataset}",
            ["*.csv", "*.json", "*/train_meta.parquet", "*/train_meta.csv", "*/manifest.json"],
        )
        add_file(
            archive,
            context.eval_manifest,
            f"color-filter-ablation/data/eval/{context.experiment}/eval_manifest.json",
        )
        add_tree(
            archive,
            context.runtime_config_dir,
            f"color-filter-olmo/runtime_configs/{context.experiment}",
            ["*.yaml"],
        )
        add_file(
            archive,
            context.analysis_helper,
            "color-filter-olmo-analysis/scripts/score_pool_410m_report.py",
        )
        add_file(
            archive,
            context.orchestration_helper,
            "color-filter-olmo/scripts/score_pool_410m_colab.py",
        )
        manifest = {
            "experiment": context.experiment,
            "train_dataset": context.train_dataset,
            "run_ids": list(context.run_ids),
            "ablation_producer_sha": context.ablation_producer_sha,
            "olmo_producer_sha": context.olmo_producer_sha,
            "olmo_analysis_sha": context.olmo_analysis_sha,
            "notebook_revision": context.notebook_revision,
            "orchestration_helper_sha256": source_sha256(context.orchestration_helper),
            "files": files,
        }
        archive.writestr("score_pool_410m_figure_bundle_manifest.json", json.dumps(manifest, indent=2))

    verify_bundle(zip_path)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, drive_path)
    verify_bundle(drive_path)
    if drive_path.stat().st_size != zip_path.stat().st_size:
        raise AssertionError((zip_path.stat().st_size, drive_path.stat().st_size))
    print("files packaged:", len(files))
    print("zip:", zip_path, f"{zip_path.stat().st_size / 1_000_000:.1f} MB")
    print("Drive fallback:", drive_path)
    return manifest


def verify_bundle(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt bundle member: {bad_member}")
