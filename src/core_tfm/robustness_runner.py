"""Utilities for executing parameterized robustness variants of the frozen Q1 notebook.

The goal is to reuse the original inference implementation rather than maintain a
second benchmark stack. A caller loads the original notebook, patches only
predeclared protocol constants (run id, sampling seed, context limit, output
location, time budget, and optional expensive post-processing flags), and then
executes code cells through shard 12E.

The original frozen result directory is never targeted by these helpers.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

EXPECTED_METHODS = 8
EXPECTED_DATASETS = 10
EXPECTED_MODELS = 3  # two TFMs + CatBoost boundary baseline, matching Q1 notebook
EXPECTED_FOLDS = 5
EXPECTED_ROWS = EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_FOLDS * EXPECTED_METHODS


def load_notebook(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _replace_once(text: str, old: str, new: str, *, required: bool = True) -> str:
    count = text.count(old)
    if required and count == 0:
        raise ValueError(f"Required notebook patch target not found: {old!r}")
    if count > 1:
        raise ValueError(f"Notebook patch target is ambiguous ({count} matches): {old!r}")
    return text.replace(old, new, 1) if count else text


def patch_q1_notebook(
    notebook: dict,
    *,
    run_id: str,
    seed: int,
    train_limit: int,
    test_limit: int = 128,
    drive_base: str = "/content/drive/MyDrive/CoRe_TFM_Q1/core_tfm_jmlr_robustness_v2",
    session_minutes: int = 720,
    disable_controlled_replications: bool = True,
    disable_selection_ablations: bool = True,
    disable_validation_sensitivity: bool = True,
) -> dict:
    """Return a patched copy of the original Q1 notebook.

    Only protocol constants are changed. The inference/data/reconciliation code
    remains the original notebook implementation.
    """
    if train_limit <= 0 or test_limit <= 0:
        raise ValueError("train_limit and test_limit must be positive")
    if session_minutes <= 0:
        raise ValueError("session_minutes must be positive")

    nb = deepcopy(notebook)
    all_source = "\n".join(
        "".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"
    )

    replacements = [
        ('RUN_ID = "core_tfm_q1_fast_complete_256_v1"', f'RUN_ID = "{run_id}"'),
        (
            'PROTOCOL_REVISION = "v7_bounded_256x128_complete_2026-08-23"',
            f'PROTOCOL_REVISION = "jmlr_robustness_v2_seed{seed}_train{train_limit}"',
        ),
        (
            'DRIVE_BASE = Path("/content/drive/MyDrive/CoRe_TFM_Q1")',
            f'DRIVE_BASE = Path("{drive_base}")',
        ),
        (
            'MODEL_CACHE = DRIVE_BASE / "model_cache"',
            'MODEL_CACHE = Path("/content/drive/MyDrive/CoRe_TFM_Q1/model_cache")',
        ),
        ('SEED = 42', f'SEED = {int(seed)}'),
        ('SESSION_TIME_BUDGET_MINUTES = 30', f'SESSION_TIME_BUDGET_MINUTES = {int(session_minutes)}'),
        ('MAX_TRAIN_ROWS = 256 if FULL_Q1_RUN else 192', f'MAX_TRAIN_ROWS = {int(train_limit)}'),
        ('MAX_TEST_ROWS = 128 if FULL_Q1_RUN else 96', f'MAX_TEST_ROWS = {int(test_limit)}'),
    ]
    if disable_controlled_replications:
        replacements.append(('RUN_CONTROLLED_REPLICATIONS = FULL_Q1_RUN', 'RUN_CONTROLLED_REPLICATIONS = False'))
    if disable_selection_ablations:
        replacements.append(('RUN_SELECTION_ABLATIONS = FULL_Q1_RUN', 'RUN_SELECTION_ABLATIONS = False'))
    if disable_validation_sensitivity:
        replacements.append(('RUN_VALIDATION_SENSITIVITY = FULL_Q1_RUN', 'RUN_VALIDATION_SENSITIVITY = False'))

    for old, _ in replacements:
        if old not in all_source:
            raise ValueError(f"Template drift detected; patch target missing: {old}")

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for old, new in replacements:
            if old in src:
                src = _replace_once(src, old, new)
        cell["source"] = src.splitlines(keepends=True)
    return nb


def code_cells_through_shard_12e(notebook: dict) -> list[str]:
    """Return code sources from setup through the final 12E inference shard.

    Stops before the original notebook's post-processing/manuscript cells.
    """
    out: list[str] = []
    found_12e = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        out.append(src)
        first = src.lstrip().splitlines()[0] if src.strip() else ""
        if "Cell 12E" in first or "12E" in first and "Cell" in first:
            found_12e = True
            break
    if not found_12e:
        raise ValueError("Could not locate Q1 shard 12E; template structure changed")
    return out


def fold_result_status(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    path = run / "fold_results.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {"complete": False, "reason": "fold_results.csv missing or empty", "rows": 0}
    df = pd.read_csv(path)
    required = {"dataset", "model", "fold", "method", "joint_nll"}
    missing = required - set(df.columns)
    if missing:
        return {"complete": False, "reason": f"missing columns: {sorted(missing)}", "rows": int(len(df))}
    method_counts = df.groupby(["dataset", "model", "fold"])["method"].nunique()
    complete = (
        len(df) == EXPECTED_ROWS
        and df["dataset"].nunique() == EXPECTED_DATASETS
        and df["model"].nunique() == EXPECTED_MODELS
        and df[["dataset", "model", "fold"]].drop_duplicates().shape[0]
        == EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_FOLDS
        and int(method_counts.min()) == EXPECTED_METHODS
        and int(method_counts.max()) == EXPECTED_METHODS
    )
    return {
        "complete": bool(complete),
        "rows": int(len(df)),
        "expected_rows": EXPECTED_ROWS,
        "datasets": int(df["dataset"].nunique()),
        "models": int(df["model"].nunique()),
        "fold_cells": int(df[["dataset", "model", "fold"]].drop_duplicates().shape[0]),
        "methods_per_fold_min": int(method_counts.min()),
        "methods_per_fold_max": int(method_counts.max()),
    }


def write_complete_marker(run_dir: str | Path, payload: dict) -> Path:
    run = Path(run_dir)
    status = fold_result_status(run)
    if not status.get("complete"):
        raise RuntimeError(f"Refusing COMPLETE marker for incomplete run: {status}")
    marker = run / "COMPLETE.json"
    marker.write_text(json.dumps({**payload, "fold_result_status": status}, indent=2), encoding="utf-8")
    return marker
