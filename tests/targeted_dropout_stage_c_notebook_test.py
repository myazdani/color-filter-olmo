import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "dropout_uncertainty_targeted_ladder_stage_c_colab.ipynb"
)
PRODUCER_SHA = "f7c8efad718ce65f0681b1f67e6016c2b4f00f7a"


def load_notebook():
    return json.loads(NOTEBOOK.read_text())


def code_sources(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_stage_c_notebook_is_pinned_and_pre_gated():
    notebook = load_notebook()
    source = "\n".join(code_sources(notebook))

    assert f"OLMO_SHA = '{PRODUCER_SHA}'" in source
    assert "RUN_STAGE = 'stage_c_500k'" in source
    assert "ENABLE_STAGE_C = True" in source
    assert "STAGE_C_CONFIG_IDS = ['dropout_embed_p0005', 'dropout_resid_p001']" in source
    assert "stage_b_gate.get('stage_passed')" in source
    assert "stage_b_gate.get('olmo_sha') != OLMO_SHA" in source


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
    assert "zipfile.ZipFile" in code[-2]
    assert "files.download" not in code[-2]
    assert "AUTO_DOWNLOAD = globals().get('AUTO_DOWNLOAD', True)" in code[-1]
    assert "files.download" in code[-1]
    assert "exec(compile" not in "\n".join(code)


def test_stage_c_scoring_gate_does_not_require_report_only_baselines():
    source = "\n".join(code_sources(load_notebook()))
    required_inputs_block = source.split("required_inputs = [", 1)[1].split("]", 1)[0]

    assert "TOKENS_DRIVE" in required_inputs_block
    assert "META_DRIVE" in required_inputs_block
    assert "FULL_SCORES_DRIVE" in required_inputs_block
    assert "PAIR_MID2_DRIVE" not in required_inputs_block
    assert "BROAD_DROPOUT_ANALYSIS" not in required_inputs_block
