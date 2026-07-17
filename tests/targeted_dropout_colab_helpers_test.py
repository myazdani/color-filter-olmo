import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "targeted_dropout_colab_helpers.py"
SPEC = importlib.util.spec_from_file_location("targeted_dropout_colab_helpers", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def test_parse_score_log_metrics_uses_last_measurement(tmp_path):
    log_path = tmp_path / "score.log"
    log_path.write_text(
        "throughput/device/tokens_per_second=12,000\n"
        "throughput/device/batches_per_second=1.5\n"
        "System/Peak GPU Memory (MB)=3,216\n"
        "throughput/device/tokens_per_second=48,125\n"
    )

    metrics = HELPERS.TargetedDropoutRunner.parse_score_log_metrics(log_path)

    assert metrics == {
        "tokens_per_second": 48125.0,
        "batches_per_second": 1.5,
        "peak_gpu_memory_mb": 3216.0,
    }


def test_select_fastest_benchmark_ignores_failed_and_invalid_results():
    selected = HELPERS.select_fastest_benchmark(
        [
            {"microbatch": 8, "status": "failed: OOM", "tokens_per_second": 90000},
            {"microbatch": 16, "status": "ok", "tokens_per_second": 48000},
            {"microbatch": 24, "status": "ok", "tokens_per_second": None},
            {"microbatch": 32, "status": "ok", "tokens_per_second": 51000},
        ]
    )

    assert selected["microbatch"] == 32
    assert selected["tokens_per_second"] == 51000.0


def test_select_fastest_benchmark_requires_valid_throughput():
    with pytest.raises(RuntimeError, match="valid throughput"):
        HELPERS.select_fastest_benchmark([{"microbatch": 16, "status": "ok", "tokens_per_second": None}])


def test_score_config_removes_template_only_sweep_directives(tmp_path):
    class Config(dict):
        __getattr__ = dict.__getitem__

        def __setattr__(self, key, value):
            self[key] = value

    class FakeOmegaConf:
        @staticmethod
        def load(path):
            if Path(path).parent.name == "checkpoint":
                return Config(model=Config(name="checkpoint-model"), tokenizer=Config(name="tokenizer"))
            return Config(model=Config(name="template-model"), targeted_ladder=[], sweep=[])

    context = HELPERS.ScoringContext(
        olmo_dir=tmp_path,
        template_config=tmp_path / "template.yaml",
        runtime_checkpoint_dir=tmp_path,
        runtime_config_dir=tmp_path,
        config_drive=tmp_path,
        raw_score_drive=tmp_path,
        stage_root=tmp_path,
        subset_raw=tmp_path / "tokens.raw",
        run_state_path=tmp_path / "state.json",
        producer_sha="producer",
        analysis_sha="analysis",
        notebook_revision="test",
        run_stage="test",
        subset_id="test",
        subset_fingerprint="subset",
        runtime_identity={},
        checkpoint_identities={},
        seed=1,
        num_samples=1,
        global_batch_size=1,
        stage_rows=1,
        shard_rows=1,
        file_seqs=1,
    )
    runner = HELPERS.TargetedDropoutRunner(context)
    runner._config_types = lambda: (FakeOmegaConf, object)

    config = runner.load_checkpoint_score_config(tmp_path / "checkpoint")

    assert config.model.name == "checkpoint-model"
    assert config.tokenizer.name == "tokenizer"
    assert "targeted_ladder" not in config
    assert "sweep" not in config


def test_shard_payload_preserves_resume_contract(tmp_path):
    context = HELPERS.ScoringContext(
        olmo_dir=tmp_path,
        template_config=tmp_path / "template.yaml",
        runtime_checkpoint_dir=tmp_path / "checkpoints",
        runtime_config_dir=tmp_path / "configs",
        config_drive=tmp_path / "drive-configs",
        raw_score_drive=tmp_path / "scores",
        stage_root=tmp_path / "stage",
        subset_raw=tmp_path / "tokens.raw",
        run_state_path=tmp_path / "run-state.json",
        producer_sha="producer123",
        analysis_sha="analysis123",
        notebook_revision="stage-c-test",
        run_stage="stage_b_100k",
        subset_id="stage_b_100k",
        subset_fingerprint="subset-hash",
        runtime_identity={"python": "3.12.0", "torch": "2.11.0+cu128"},
        checkpoint_identities={"prior": {"model": "prior-hash"}},
        seed=1,
        num_samples=8,
        global_batch_size=32,
        stage_rows=100000,
        shard_rows=5120,
        file_seqs=100000,
    )
    runner = HELPERS.TargetedDropoutRunner(context)
    config = {
        "config_id": "dropout_attn_p001",
        "attention_dropout": 0.01,
        "residual_dropout": 0.0,
        "embedding_dropout": 0.0,
    }
    shard = {"start": 0, "end": 5120, "rows": 5120, "data_start_step": 0}

    payload = runner.shard_experiment_payload(config, "prior", shard, 16)

    assert payload["schema_version"] == 4
    assert payload["producer_sha"] == "producer123"
    assert payload["runtime_identity"]["torch"] == "2.11.0+cu128"
    assert payload["checkpoint_identity"] == {"model": "prior-hash"}
    assert payload["shard"] == shard
    assert runner.shard_experiment_fingerprint(config, "prior", shard, 16) == (HELPERS.canonical_sha256(payload))


def test_config_num_samples_override_is_part_of_resume_contract(tmp_path):
    context = HELPERS.ScoringContext(
        olmo_dir=tmp_path,
        template_config=tmp_path / "template.yaml",
        runtime_checkpoint_dir=tmp_path / "checkpoints",
        runtime_config_dir=tmp_path / "configs",
        config_drive=tmp_path / "drive-configs",
        raw_score_drive=tmp_path / "scores",
        stage_root=tmp_path / "stage",
        subset_raw=tmp_path / "tokens.raw",
        run_state_path=tmp_path / "run-state.json",
        producer_sha="producer123",
        analysis_sha="analysis123",
        notebook_revision="broad-sweep-test",
        run_stage="broad_rate_sweep_500k",
        subset_id="broad_rate_sweep_500k",
        subset_fingerprint="subset-hash",
        runtime_identity={"torch": "2.11.0+cu128"},
        checkpoint_identities={"prior": {"model": "prior-hash"}},
        seed=1,
        num_samples=8,
        global_batch_size=32,
        stage_rows=500000,
        shard_rows=24992,
        file_seqs=1048576,
    )
    runner = HELPERS.TargetedDropoutRunner(context)
    config = {
        "config_id": "dropout_broad_k1_p000",
        "num_samples": 1,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "embedding_dropout": 0.0,
    }
    shard = {"start": 0, "end": 32, "rows": 32, "data_start_step": 0}

    payload = runner.shard_experiment_payload(config, "prior", shard, 16)

    assert runner.config_num_samples(config) == 1
    assert payload["num_samples"] == 1
    assert payload["config"]["num_samples"] == 1


def test_config_num_samples_override_must_be_positive(tmp_path):
    context = HELPERS.ScoringContext(
        olmo_dir=tmp_path,
        template_config=tmp_path / "template.yaml",
        runtime_checkpoint_dir=tmp_path,
        runtime_config_dir=tmp_path,
        config_drive=tmp_path,
        raw_score_drive=tmp_path,
        stage_root=tmp_path,
        subset_raw=tmp_path / "tokens.raw",
        run_state_path=tmp_path / "state.json",
        producer_sha="producer",
        analysis_sha="analysis",
        notebook_revision="test",
        run_stage="test",
        subset_id="test",
        subset_fingerprint="subset",
        runtime_identity={},
        checkpoint_identities={},
        seed=1,
        num_samples=8,
        global_batch_size=32,
        stage_rows=32,
        shard_rows=32,
        file_seqs=32,
    )

    with pytest.raises(ValueError, match="must be positive"):
        HELPERS.TargetedDropoutRunner(context).config_num_samples({"config_id": "bad", "num_samples": 0})


def test_run_logged_streams_output_and_emits_heartbeat(tmp_path, capsys):
    context = HELPERS.ScoringContext(
        olmo_dir=tmp_path,
        template_config=tmp_path / "template.yaml",
        runtime_checkpoint_dir=tmp_path,
        runtime_config_dir=tmp_path,
        config_drive=tmp_path,
        raw_score_drive=tmp_path,
        stage_root=tmp_path,
        subset_raw=tmp_path / "tokens.raw",
        run_state_path=tmp_path / "state.json",
        producer_sha="producer",
        analysis_sha="analysis",
        notebook_revision="test",
        run_stage="test",
        subset_id="test",
        subset_fingerprint="subset",
        runtime_identity={},
        checkpoint_identities={},
        seed=1,
        num_samples=1,
        global_batch_size=1,
        stage_rows=1,
        shard_rows=1,
        file_seqs=1,
    )

    log_path = tmp_path / "command.log"
    HELPERS.TargetedDropoutRunner(context).run_logged(
        [sys.executable, "-u", "-c", "import time; print('child-ready'); time.sleep(0.08)"],
        log_path,
        heartbeat_seconds=0.01,
        label="test command",
    )

    captured = capsys.readouterr().out
    assert "child-ready" in captured
    assert "still running; granular progress unavailable" in captured
    assert "test command: complete" in captured
    assert "child-ready" in log_path.read_text()


def test_runtime_validation_allows_one_batch_tail_without_throughput():
    HELPERS.validate_runtime_records(
        [
            {
                "rows": 320,
                "elapsed_seconds": 10.0,
                "tokens_per_second": 48000.0,
                "batches_per_second": 2.9,
                "peak_gpu_memory_mb": 3216.0,
                "microbatch": 32,
            },
            {
                "rows": 32,
                "elapsed_seconds": 2.0,
                "tokens_per_second": None,
                "batches_per_second": None,
                "peak_gpu_memory_mb": 3216.0,
                "microbatch": 32,
            },
        ],
        global_batch_size=32,
        expected_microbatch=32,
    )


def test_runtime_validation_rejects_missing_multi_batch_throughput():
    with pytest.raises(RuntimeError, match="multi-batch"):
        HELPERS.validate_runtime_records(
            [
                {
                    "rows": 64,
                    "elapsed_seconds": 2.0,
                    "tokens_per_second": None,
                    "batches_per_second": None,
                    "peak_gpu_memory_mb": 3216.0,
                    "microbatch": 32,
                }
            ],
            global_batch_size=32,
            expected_microbatch=32,
        )


def test_create_verified_archive_copies_crc_checked_drive_fallback(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first\n")
    second.write_text("second\n")
    local_archive = tmp_path / "local" / "bundle.zip"
    drive_archive = tmp_path / "drive" / "bundle.zip"

    local, drive = HELPERS.create_verified_archive(
        [(first, "inputs/first.txt"), (second, "inputs/second.txt")],
        {"producer_sha": "producer", "analysis_sha": "analysis"},
        local_archive,
        drive_archive,
    )

    assert local == local_archive
    assert drive == drive_archive
    assert local.read_bytes() == drive.read_bytes()
    assert drive.with_suffix(".manifest.json").is_file()
    assert set(HELPERS.verify_bundle_archive(drive)) == {
        "bundle_manifest.json",
        "inputs/first.txt",
        "inputs/second.txt",
    }
