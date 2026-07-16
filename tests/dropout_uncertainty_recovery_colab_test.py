import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dropout_uncertainty_recovery_colab import (
    DropoutUncertaintyRecovery,
    RecoveryContext,
    load_npz,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POOLS = (
    "random_positive",
    "hard_positive",
    "hard_negative",
    "random_negative",
    "tail_negative",
)


def make_legacy_artifact(root: Path, rows_per_pool: int = 100):
    legacy_dir = root / "legacy"
    corrected_dir = root / "corrected"
    report_dir = root / "report"
    legacy_dir.mkdir(parents=True)

    rows = rows_per_pool * len(POOLS)
    rng = np.random.default_rng(20260715)
    pool_name = np.repeat(np.asarray(POOLS), rows_per_pool)
    pool_centers = {
        "random_positive": 0.12,
        "hard_positive": 0.28,
        "hard_negative": 0.45,
        "random_negative": 0.66,
        "tail_negative": 0.88,
    }
    source_color = np.asarray([pool_centers[name] for name in pool_name])
    source_color += rng.normal(0.0, 0.045, size=rows)
    full_color = source_color + rng.normal(0.0, 0.002, size=rows)
    score_index = rng.permutation(rows).astype(np.int64)
    sample_noise = rng.normal(0.0, 0.006, size=(rows, 8))
    color_samples = (source_color[score_index, None] + sample_noise).astype(np.float32)
    prior_losses = np.full_like(color_samples, 3.0)
    conditional_losses = prior_losses + color_samples

    metadata = json.dumps({"config_id": "dropout_k8_p005", "legacy": True}, sort_keys=True)
    arrays = {
        "seq_idx": np.arange(rows, dtype=np.int64),
        "score_index": score_index,
        "pool_name": pool_name,
        "prior_losses": prior_losses,
        "conditional_losses": conditional_losses,
        "color_samples": color_samples,
        "utility_samples": -color_samples,
        "metadata_json": np.asarray(metadata, dtype=f"U{len(metadata)}"),
        "full_color_score": full_color.astype(np.float32),
    }
    npz_path = legacy_dir / "mc_samples_dropout_k8_p005.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest_path = legacy_dir / "mc_samples_dropout_k8_p005_manifest.json"
    manifest_path.write_text(
        json.dumps({"config_id": "dropout_k8_p005", "rows": rows, "num_samples": 8}),
        encoding="utf-8",
    )
    context = RecoveryContext(
        olmo_dir=REPO_ROOT,
        legacy_analysis_dir=legacy_dir,
        corrected_analysis_dir=corrected_dir,
        report_dir=report_dir,
        raw_score_root=root / "unused_raw_scores",
        metadata_path=root / "unused_metadata.csv",
        full_scores_path=root / "unused_full_scores.csv",
        producer_sha="legacy-unrecorded-broad-dropout-k8-p005",
        analysis_sha="test-analysis-sha",
        notebook_revision="test-revision",
    )
    return DropoutUncertaintyRecovery(context), arrays, npz_path


def test_repair_preserves_samples_and_realigns_reference_arrays(tmp_path):
    workflow, source, legacy_npz = make_legacy_artifact(tmp_path)
    source_hash = sha256_file(legacy_npz)

    preflight = workflow.preflight()
    assert preflight["legacy"]["raw_spearman_mean_vs_full"] < 0.2
    assert preflight["legacy"]["candidate_repaired_spearman_mean_vs_full"] > 0.9

    result = workflow.repair_compact(write_parquet=False)
    corrected = load_npz(workflow.corrected_npz)
    permutation = source["score_index"]

    assert result["status"] == "repaired"
    assert sha256_file(legacy_npz) == source_hash
    for name in ("prior_losses", "conditional_losses", "color_samples", "utility_samples", "score_index"):
        np.testing.assert_array_equal(corrected[name], source[name])
    np.testing.assert_array_equal(corrected["seq_idx"], source["seq_idx"][permutation])
    np.testing.assert_array_equal(corrected["pool_name"], source["pool_name"][permutation])
    np.testing.assert_array_equal(corrected["full_color_score"], source["full_color_score"][permutation])

    manifest = json.loads(workflow.corrected_manifest.read_text(encoding="utf-8"))
    assert manifest["row_alignment"] == "legacy_reference_arrays_indexed_by_score_index"
    assert manifest["legacy_npz_sha256"] == source_hash
    assert manifest["alignment_repaired"] is True

    strategy = workflow.run_strategy_sweep()
    assert strategy["mean_pairwise_auc"] > 0.9
    assert strategy["recall_vs_full_1_64"] > 0.8
    report = workflow.build_report()
    assert report["summary"]["spearman_mean_vs_full"] > 0.9
    assert report["report"].is_file()
    assert workflow.status()["validation"]["rows"] == len(permutation)

    reused = workflow.repair_compact(write_parquet=False)
    assert reused["status"] == "reused"


def test_repair_rejects_non_source_ordered_legacy_ids(tmp_path):
    workflow, source, legacy_npz = make_legacy_artifact(tmp_path)
    source["seq_idx"] = source["score_index"].copy()
    np.savez_compressed(legacy_npz, **source)

    with pytest.raises(ValueError, match="automatic repair would be ambiguous"):
        workflow.preflight()
