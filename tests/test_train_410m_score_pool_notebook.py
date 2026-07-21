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
    assert code[setup_index].index("base_production_order =") < code[setup_index].index(
        "generate_seed_configs("
    )
    assert "production_order = list(runtime_config_map)" in code[setup_index]
    assert "len(production_order) == len(set(production_order))" in code[setup_index]
    assert "EXPECTED_EVAL_POINTS = 10" in code[setup_index]

    base_run_ids = [f"base_{index}_100k" for index in range(8)] + [
        "hard_positive_oracle_100k",
        "hard_pair_cascade_100k",
    ]
    runtime_config_map = {run_id: Path(f"/{run_id}.yaml") for run_id in base_run_ids}

    class FakeHelper:
        @staticmethod
        def generate_seed_configs(**kwargs) -> list[str]:
            seed_run_ids = []
            for seed in kwargs["seeds"]:
                for base_run_id in kwargs["base_run_ids"]:
                    run_id = base_run_id.removesuffix("_100k") + f"_seed{seed}_100k"
                    kwargs["runtime_config_map"][run_id] = Path(f"/{run_id}.yaml")
                    seed_run_ids.append(run_id)
            return seed_run_ids

    namespace = {
        "runtime_config_map": runtime_config_map,
        "SCORE_POOL_COLAB": FakeHelper,
        "CHECKPOINTS_DRIVE": Path("/checkpoints"),
        "RUNTIME_CONFIG_DIR": Path("/configs"),
    }
    exec(code[setup_index], namespace)

    assert namespace["base_production_order"] == base_run_ids
    assert len(namespace["production_order"]) == len(set(namespace["production_order"])) == 14
