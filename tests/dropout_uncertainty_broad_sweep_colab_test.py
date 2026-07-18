import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "dropout_uncertainty_broad_sweep_colab.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "dropout_uncertainty_broad_rate_sweep_colab.ipynb"
PINNED_PRODUCER_SHA = "7d19d836bc48a6ca76621558d5a339ad030af284"
PINNED_ANALYSIS_SHA = "afe9db7b62bf17dab38ee4a51395e40adcb2dfea"
SPEC = importlib.util.spec_from_file_location("dropout_uncertainty_broad_sweep_colab", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


def sweep_configs():
    return [
        {
            "config_id": config_id,
            "dropout_rate": rate,
            "num_samples": samples,
            "dropout_target": "attention+residual+embedding",
            "attention_dropout": rate,
            "residual_dropout": rate,
            "embedding_dropout": rate,
        }
        for config_id, rate, samples in (
            ("dropout_broad_k1_p000", 0.0, 1),
            ("dropout_broad_k8_p000001", 0.00001, 8),
            ("dropout_broad_k8_p0001", 0.001, 8),
            ("dropout_broad_k8_p001", 0.01, 8),
        )
    ]


def test_validate_sweep_configs_accepts_requested_grid():
    HELPER.validate_sweep_configs(sweep_configs())


def test_validate_sweep_configs_rejects_non_broad_or_wrong_k():
    configs = sweep_configs()
    configs[0]["num_samples"] = 8
    with pytest.raises(ValueError, match="requires K=1"):
        HELPER.validate_sweep_configs(configs)
    configs = sweep_configs()
    configs[-1]["embedding_dropout"] = 0.0
    with pytest.raises(ValueError, match="embedding_dropout"):
        HELPER.validate_sweep_configs(configs)


def test_finite_sample_std_handles_k1_without_nan():
    samples = np.asarray([[1.0], [2.0]], dtype=np.float32)
    assert np.array_equal(HELPER.finite_sample_std(samples), np.zeros(2, dtype=np.float32))


def test_reference_alignment_contracts_accept_both_corrected_formats():
    assert HELPER.FIXED_REFERENCE_ALIGNMENT_CONTRACTS == {
        "metadata_and_full_scores_indexed_by_score_index",
        "legacy_reference_arrays_indexed_by_score_index",
    }


def test_validate_raw_grid_requires_matching_complete_permutations(tmp_path):
    configs = sweep_configs()
    shards = [
        {"start": 0, "end": 4, "rows": 4, "data_start_step": 0},
        {"start": 4, "end": 8, "rows": 4, "data_start_step": 1},
    ]

    class FakeRunner:
        def shard_output_dir(self, config_id, model_id, shard):
            return tmp_path / config_id / model_id / f"{shard['start']}_{shard['end']}"

        def valid_score_output(self, output, config, model_id, shard, microbatch):
            return (output / "score" / "mmap_index.npy").is_file()

    runner = FakeRunner()
    for config in configs:
        for model_id in ("prior", "books"):
            for shard in shards:
                output = runner.shard_output_dir(config["config_id"], model_id, shard) / "score"
                output.mkdir(parents=True)
                index = np.memmap(output / "mmap_index.npy", dtype=np.int64, mode="w+", shape=(4,))
                index[:] = np.arange(shard["start"], shard["end"])
                index.flush()
    context = HELPER.BroadSweepContext(
        runner=runner,
        analysis_olmo_dir=tmp_path,
        configs=configs,
        shards=shards,
        prior_checkpoint=tmp_path,
        books_checkpoint=tmp_path,
        metadata_path=tmp_path,
        full_scores_path=tmp_path,
        stage_root=tmp_path,
        report_dir=tmp_path,
        config_drive=tmp_path,
        run_state_path=tmp_path,
        shard_plan_path=tmp_path,
        subset_manifest_path=tmp_path,
        producer_sha="producer",
        analysis_sha="analysis",
        notebook_revision="test",
        rows=8,
        seq_len=512,
        tau64_cutoff=0.0,
        global_batch_size=4,
        shard_rows=4,
        microbatch=4,
    )
    workflow = HELPER.BroadDropoutSweep(context)

    workflow.validate_raw_grid(microbatch=4)

    bad_path = runner.shard_output_dir(configs[0]["config_id"], "books", shards[0]) / "score/mmap_index.npy"
    bad_index = np.memmap(bad_path, dtype=np.int64, mode="r+")
    bad_index[:] = np.asarray([1, 0, 2, 3])
    bad_index.flush()
    with pytest.raises(RuntimeError, match="prior and Books score-index order differs"):
        workflow.validate_raw_grid(microbatch=4)


def test_notebook_compiles_and_declares_hardened_sweep_contract():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None
    for config in sweep_configs():
        assert config["config_id"] in source
    assert f"PRODUCER_SHA = '{PINNED_PRODUCER_SHA}'" in source
    assert f"ANALYSIS_SHA = '{PINNED_ANALYSIS_SHA}'" in source
    assert "checkout_exact(PRODUCER_DIR, PRODUCER_SHA)" in source
    assert "checkout_exact(ANALYSIS_DIR, ANALYSIS_SHA)" in source
    assert "analysis_olmo_dir=ANALYSIS_DIR" in source
    assert "'gpu_name': torch.cuda.get_device_name(0)" in source
    assert "'gpu_total_memory_bytes': int(gpu.total_memory)" in source
    assert "pythonpath_root=self.context.analysis_olmo_dir" in HELPER_PATH.read_text()
    assert 'self.analysis_script_dir / "21_dropout_uncertainty_metrics.py"' in HELPER_PATH.read_text()
    assert "metadata_and_full_scores_indexed_by_score_index" in source
    assert "WORKFLOW.run_production()" in source
    assert "WORKFLOW.raw_status()" in source
    assert "WORKFLOW.analyze()" in source
    assert "AUTO_DOWNLOAD = globals().get('AUTO_DOWNLOAD', True)" in source
    assert "files.download(str(DRIVE_ARCHIVE))" in "".join(code_cells[-1]["source"])
    assert "build_bundle" not in "".join(code_cells[-1]["source"])


def test_long_running_cells_delegate_progress_to_helper():
    helper_source = HELPER_PATH.read_text()
    runner_source = (ROOT / "scripts" / "targeted_dropout_colab_helpers.py").read_text()
    assert "Conservative production estimate" in helper_source
    assert "conservative remaining ETA" in helper_source
    assert "observed ETA" in helper_source
    assert "still running; granular progress unavailable" in runner_source
    assert "flush=True" in runner_source
    assert "run_logged" in helper_source
