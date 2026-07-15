import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "dropout_uncertainty_targeted_ladder_stage_c_colab.ipynb"
)
PRODUCER_SHA = "f7c8efad718ce65f0681b1f67e6016c2b4f00f7a"
ANALYSIS_SHA = "1adafd63779821c4803eb90b958cf6640a30f6fd"


def load_notebook():
    return json.loads(NOTEBOOK.read_text())


def code_sources(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_stage_c_notebook_is_pinned_and_pre_gated():
    notebook = load_notebook()
    source = "\n".join(code_sources(notebook))

    assert f"PRODUCER_SHA = '{PRODUCER_SHA}'" in source
    assert f"ANALYSIS_SHA = '{ANALYSIS_SHA}'" in source
    assert "NOTEBOOK_REVISION = 'stage-c-v3-2026-07-15'" in source
    assert "checkout', '--detach', ANALYSIS_SHA" in source
    assert "RUN_STAGE = 'stage_c_500k'" in source
    assert "ENABLE_STAGE_C = True" in source
    assert "STAGE_C_CONFIG_IDS = ['dropout_embed_p0005', 'dropout_resid_p001']" in source
    assert "stage_b_gate.get('stage_passed')" in source
    assert "stage_b_gate.get('olmo_sha') != PRODUCER_SHA" in source


def test_stage_c_shard_plan_is_batch_aligned():
    stage_rows = 500_000
    shard_rows = 24_992
    global_batch_size = 32
    shards = []
    start = 0
    while start < stage_rows:
        rows = min(shard_rows, stage_rows - start)
        assert rows % global_batch_size == 0
        shards.append(rows)
        start += rows

    assert len(shards) == 21
    assert shards[-1] == 160


def test_stage_c_notebook_compiles_and_download_is_separate():
    notebook = load_notebook()
    code = code_sources(notebook)
    for index, source in enumerate(code):
        compile(source, f"{NOTEBOOK.name}:code-cell-{index}", "exec")

    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "WORKFLOW.build_bundle" in code[-2]
    assert "files.download" not in code[-2]
    assert "DRIVE_ARCHIVE_PATH = STAGE_ROOT / 'bundles' / ARCHIVE_NAME" in code[-2]
    assert "AUTO_DOWNLOAD = globals().get('AUTO_DOWNLOAD', True)" in code[-1]
    assert "files.download" in code[-1]
    assert "verify_bundle_archive(download_path)" in code[-1]
    assert "Drive/manual-download fallback" in code[-1]
    assert "exec(compile" not in "\n".join(code)


def test_stage_c_notebook_delegates_nontrivial_work_to_helper():
    notebook = load_notebook()
    source = "\n".join(code_sources(notebook))

    assert "prepare_fixed_subset(SubsetContext(" in source
    assert "WORKFLOW.analyze()" in source
    assert "WORKFLOW.build_report()" in source
    assert "WORKFLOW.build_bundle" in source
    assert "def validate_analysis" not in source
    assert "def require_bundle_file" not in source
    assert "zipfile.ZipFile" not in source
    assert "shutil.rmtree" not in source

    exempt_long_cells = {7, 10, 15}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and index not in exempt_long_cells:
            assert len("".join(cell["source"]).splitlines()) <= 40, index
