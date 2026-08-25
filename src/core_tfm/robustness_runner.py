"""Utilities for executing parameterized robustness variants of the Q1 notebook.

The robustness workflow deliberately reuses the original Q1 inference stack. A
caller patches only prespecified protocol constants and executes the original
cells through shard 12E. These helpers never target the frozen Q1 result folder.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

EXPECTED_METHODS = 8
EXPECTED_DATASETS = 10
EXPECTED_MODELS = 3  # two TFMs + CatBoost boundary baseline, matching Q1
EXPECTED_FOLDS = 5
EXPECTED_ROWS = EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_FOLDS * EXPECTED_METHODS


def load_notebook(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
        raise ValueError(f"Not a valid nbformat-4 notebook: {path}")
    return nb


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
    drive_base: str = "/content/drive/MyDrive/CoRe_TFM_Q1/core_tfm_jmlr_robustness_v2_1",
    session_minutes: int = 120,
    shard_minutes: int | None = None,
    disable_controlled_replications: bool = True,
    disable_selection_ablations: bool = True,
    disable_validation_sensitivity: bool = True,
) -> dict:
    """Return a protocol-patched copy of the original Q1 notebook.

    Only run/output/sampling/time-budget constants are changed. Inference,
    dataset preparation, reconciliation, validation selection, and scoring remain
    the Q1 implementation.
    """
    if not run_id or run_id.startswith("/") or ".." in Path(run_id).parts:
        raise ValueError("run_id must be a non-empty relative path without '..'")
    if train_limit <= 0 or test_limit <= 0:
        raise ValueError("train_limit and test_limit must be positive")
    if session_minutes <= 0:
        raise ValueError("session_minutes must be positive")
    if shard_minutes is None:
        shard_minutes = session_minutes
    if shard_minutes <= 0:
        raise ValueError("shard_minutes must be positive")

    nb = deepcopy(notebook)
    code_sources = [
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    ]
    all_source = "\n".join(code_sources)

    replacements: list[tuple[str, str]] = [
        ('RUN_ID = "core_tfm_q1_fast_complete_256_v1"', f'RUN_ID = "{run_id}"'),
        (
            'PROTOCOL_REVISION = "v7_bounded_256x128_complete_2026-08-23"',
            f'PROTOCOL_REVISION = "jmlr_robustness_v2_1_seed{int(seed)}_train{int(train_limit)}"',
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
        ('SHARD_TIME_BUDGET_MINUTES = 30', f'SHARD_TIME_BUDGET_MINUTES = {int(shard_minutes)}'),
        ('MAX_TRAIN_ROWS = 256 if FULL_Q1_RUN else 192', f'MAX_TRAIN_ROWS = {int(train_limit)}'),
        ('MAX_TEST_ROWS = 128 if FULL_Q1_RUN else 96', f'MAX_TEST_ROWS = {int(test_limit)}'),
        (
            '"bounded_context_protocol": {"max_train_rows": 256, "max_test_rows": 128}',
            f'"bounded_context_protocol": {{"max_train_rows": {int(train_limit)}, "max_test_rows": {int(test_limit)}}}',
        ),
    ]
    if disable_controlled_replications:
        replacements.append(('RUN_CONTROLLED_REPLICATIONS = FULL_Q1_RUN', 'RUN_CONTROLLED_REPLICATIONS = False'))
    if disable_selection_ablations:
        replacements.append(('RUN_SELECTION_ABLATIONS = FULL_Q1_RUN', 'RUN_SELECTION_ABLATIONS = False'))
    if disable_validation_sensitivity:
        replacements.append(('RUN_VALIDATION_SENSITIVITY = FULL_Q1_RUN', 'RUN_VALIDATION_SENSITIVITY = False'))

    # Require one global occurrence for every patch target. This catches template
    # drift before any GPU inference starts.
    for old, _ in replacements:
        count = all_source.count(old)
        if count != 1:
            raise ValueError(
                f"Template drift detected for {old!r}: expected exactly 1 occurrence, found {count}"
            )

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
    """Return executable Python cell sources through the final 12E shard.

    Python's compiler is the authoritative syntax check. This intentionally does
    not classify lines by their first character: valid continuation lines such as
    ``!= (...)`` can begin with ``!`` after indentation is stripped. True IPython
    magics/shell escapes fail normal Python compilation and are reported with the
    originating engine-cell index.
    """
    out: list[str] = []
    found_12e = False
    engine_index = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        engine_index += 1
        try:
            compile(src, f"<q1_engine_static_check_{engine_index}>", "exec")
        except SyntaxError as exc:
            first = src.lstrip().splitlines()[0] if src.strip() else "<empty>"
            raise ValueError(
                f"Q1 engine cell {engine_index} is not valid regular Python before 12E "
                f"({first!r}): {exc.msg} at line {exc.lineno}"
            ) from exc
        out.append(src)
        first = src.lstrip().splitlines()[0]
        if "Cell 12E" in first or ("12E" in first and "Cell" in first):
            found_12e = True
            break
    if not found_12e:
        raise ValueError("Could not locate Q1 shard 12E; template structure changed")
    return out


def _failure_summary(run: Path) -> tuple[int, dict | None]:
    path = run / "fold_failures.json"
    if not path.exists() or path.stat().st_size == 0:
        return 0, None
    try:
        failures = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return -1, {"error": "fold_failures.json is not valid JSON"}
    if not isinstance(failures, list):
        return -1, {"error": "fold_failures.json must contain a list"}
    return len(failures), (failures[-1] if failures else None)


def fold_result_status(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    path = run / "fold_results.csv"
    failure_count, last_failure = _failure_summary(run)
    base = {
        "path": str(run),
        "failure_count": failure_count,
        "last_failure": last_failure,
    }
    if not path.exists() or path.stat().st_size == 0:
        return {**base, "complete": False, "reason": "fold_results.csv missing or empty", "rows": 0, "fold_cells": 0}
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {**base, "complete": False, "reason": "fold_results.csv is empty or unparseable", "rows": 0, "fold_cells": 0}
    required = {"dataset", "model", "fold", "method", "joint_nll"}
    missing = required - set(df.columns)
    if missing:
        return {**base, "complete": False, "reason": f"missing columns: {sorted(missing)}", "rows": int(len(df)), "fold_cells": 0}
    if df.empty:
        return {**base, "complete": False, "reason": "fold_results.csv contains headers but zero rows", "rows": 0, "fold_cells": 0}

    method_counts = df.groupby(["dataset", "model", "fold"])["method"].nunique()
    fold_cells = int(df[["dataset", "model", "fold"]].drop_duplicates().shape[0])
    complete = (
        len(df) == EXPECTED_ROWS
        and df["dataset"].nunique() == EXPECTED_DATASETS
        and df["model"].nunique() == EXPECTED_MODELS
        and fold_cells == EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_FOLDS
        and int(method_counts.min()) == EXPECTED_METHODS
        and int(method_counts.max()) == EXPECTED_METHODS
    )
    return {
        **base,
        "complete": bool(complete),
        "rows": int(len(df)),
        "expected_rows": EXPECTED_ROWS,
        "datasets": int(df["dataset"].nunique()),
        "models": int(df["model"].nunique()),
        "fold_cells": fold_cells,
        "expected_fold_cells": EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_FOLDS,
        "methods_per_fold_min": int(method_counts.min()),
        "methods_per_fold_max": int(method_counts.max()),
    }


def write_complete_marker(run_dir: str | Path, payload: dict) -> Path:
    run = Path(run_dir)
    status = fold_result_status(run)
    if not status.get("complete"):
        raise RuntimeError(f"Refusing COMPLETE marker for incomplete run: {status}")
    marker = run / "COMPLETE.json"
    marker.write_text(
        json.dumps({"complete": True, **payload, "fold_result_status": status}, indent=2),
        encoding="utf-8",
    )
    return marker
