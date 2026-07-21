from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "train_410m_score_pool_mini_universes_colab.ipynb"
)


def test_report_state_is_defined_before_training_section() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]

    setup_index = next(index for index, source in enumerate(code) if "generate_seed_configs(" in source)
    training_index = next(index for index, source in enumerate(code) if "def run_training(" in source)
    report_index = next(index for index, source in enumerate(code) if "def run_report_helper(" in source)

    assert setup_index < training_index < report_index
    assert "base_production_order =" in code[setup_index]
    assert "production_order =" in code[setup_index]
    assert "EXPECTED_EVAL_POINTS = 10" in code[setup_index]
