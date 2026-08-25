from pathlib import Path

from core_tfm.robustness_runner import (
    code_cells_through_shard_12e,
    load_notebook,
    patch_q1_notebook,
)


ROOT = Path(__file__).resolve().parents[1]


def test_valid_bang_comparison_is_not_treated_as_shell_escape():
    nb = {
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "x = (\n",
                    "    1\n",
                    "    != 2\n",
                    ")\n",
                    "# Cell 12E — synthetic terminator\n",
                ],
            }
        ],
    }
    sources = code_cells_through_shard_12e(nb)
    assert len(sources) == 1
    ns = {}
    exec(compile(sources[0], "<test>", "exec"), ns, ns)
    assert ns["x"] is True


def test_actual_q1_template_patches_and_compiles_through_12e():
    template = load_notebook(
        ROOT / "notebooks" / "CoRe_TFM_Q1_FAST_COMPLETE_256_Colab.ipynb"
    )
    patched = patch_q1_notebook(
        template,
        run_id="_static_test/seed_11",
        seed=11,
        train_limit=256,
        test_limit=128,
        drive_base="/tmp/core_tfm_robustness_test",
        session_minutes=25,
        shard_minutes=25,
        disable_controlled_replications=True,
        disable_selection_ablations=True,
        disable_validation_sensitivity=True,
    )
    sources = code_cells_through_shard_12e(patched)
    assert len(sources) >= 10
    assert any("Cell 12E" in src for src in sources)
