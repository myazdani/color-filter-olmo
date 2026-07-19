from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_pool_410m_colab import (  # noqa: E402
    BundleContext,
    build_bundle,
    parse_microbatch_results,
    production_log_status,
    run_logged,
    validate_smoke_logs,
    verify_bundle,
)


def test_run_logged_captures_output(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    run_logged(
        [sys.executable, "-c", "print('done')"],
        log,
        cwd=tmp_path,
        pythonpath=tmp_path,
        heartbeat_seconds=0.01,
    )
    assert log.read_text() == "done\n"


def test_production_log_status_handles_missing_and_complete(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    assert production_log_status(log)["exists"] is False
    log.write_text(
        ("eval/books_val/CrossEntropyLoss=1\n" * 10)
        + ("eval/c4_val_proxy/CrossEntropyLoss=1\n" * 10)
        + "Training complete\n"
    )
    status = production_log_status(log)
    assert status["training_complete"] is True
    assert status["has_required_eval_curve"] is True


def test_microbatch_selection_uses_fastest_stable_candidate(tmp_path: Path) -> None:
    for microbatch, speed in [(32, 1000), (64, 1200), (128, 1100)]:
        (tmp_path / f"microbatch_{microbatch}.log").write_text(
            "Training complete\n"
            + f"throughput/device/tokens_per_second={speed}\n" * 2
            + "System/Peak GPU Memory (MB)=50000\n"
        )
    results, recommended = parse_microbatch_results(
        candidates=[32, 64, 128],
        benchmark_dir=tmp_path,
        peak_limit_mb=72_000,
        total_runs=10,
    )
    assert len(results) == 3
    assert recommended == 64


def test_smoke_validation_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "train/CrossEntropyLoss=nan\n"
        "eval/books_val/CrossEntropyLoss=1\n"
        "eval/c4_val_proxy/CrossEntropyLoss=1\n"
        "Training complete\n"
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_smoke_logs(["run"], tmp_path)


def test_smoke_validation_accepts_complete_finite_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "train/CrossEntropyLoss=2.0\n"
        "eval/books_val/CrossEntropyLoss=1.0\n"
        "eval/c4_val_proxy/CrossEntropyLoss=1.5\n"
        "Training complete\n"
    )
    validate_smoke_logs(["run"], tmp_path)


def _write(path: Path, value: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_bundle_is_verified_and_excludes_tokens(tmp_path: Path) -> None:
    train = tmp_path / "train"
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    configs = tmp_path / "configs"
    run_id = "example_100k"
    for name in [
        "train_metrics_from_logs.csv",
        "eval_metrics_from_logs.csv",
        "checkpoint_save_times.csv",
        "throughput_comparison.csv",
        "selection_diagnostics.csv",
        "overlap_jaccard.csv",
        "checkpoint_manifest.json",
        f"{run_id}.log",
    ]:
        _write(results / name)
    _write(reports / "report.md")
    _write(reports / "report.html")
    _write(reports / "figures/eval_loss_books_by_run.png")
    _write(reports / "figures/eval_loss_books_hard_oracle_vs_cascade_seeds.png")
    _write(train / run_id / "train_meta.parquet")
    _write(train / run_id / "manifest.json")
    _write(train / run_id / "train_tokens.npy")
    eval_manifest = tmp_path / "eval_manifest.json"
    _write(eval_manifest, json.dumps({"ok": True}))
    helper = tmp_path / "report.py"
    _write(helper)
    _write(configs / "run.yaml")
    context = BundleContext(
        experiment="experiment",
        train_dataset="dataset",
        run_ids=(run_id,),
        train_data_drive=train,
        results_drive=results,
        reports_drive=reports,
        runtime_config_dir=configs,
        eval_manifest=eval_manifest,
        analysis_helper=helper,
        orchestration_helper=SCRIPTS / "score_pool_410m_colab.py",
        ablation_producer_sha="a",
        olmo_producer_sha="b",
        olmo_analysis_sha="c",
        notebook_revision="d",
    )
    zip_path = tmp_path / "bundle.zip"
    drive_path = tmp_path / "drive/bundle.zip"
    build_bundle(context, zip_path, drive_path)
    verify_bundle(drive_path)
    with ZipFile(drive_path) as archive:
        names = archive.namelist()
    assert not any(name.endswith("train_tokens.npy") for name in names)


def test_corrupt_bundle_fails(tmp_path: Path) -> None:
    bundle = tmp_path / "bad.zip"
    bundle.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        verify_bundle(bundle)
