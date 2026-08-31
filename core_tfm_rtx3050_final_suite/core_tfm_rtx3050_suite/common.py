from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    repo: Path
    suite_dir: Path
    config_path: Path
    output_root: Path


def repo_root_from_arg(repo: str | Path) -> Path:
    p = Path(repo).expanduser().resolve()
    if not (p / ".git").exists():
        raise FileNotFoundError(f"Not a git repository: {p}")
    if not (p / "src" / "core_tfm").exists():
        raise FileNotFoundError(f"This does not look like the core-tfm repository: {p}")
    return p


def suite_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).resolve() if path else suite_dir() / "config" / "final_suite.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {p}")
    return data


def resolve_paths(repo: str | Path, config_path: str | Path | None = None) -> Paths:
    r = repo_root_from_arg(repo)
    s = suite_dir()
    c = Path(config_path).resolve() if config_path else s / "config" / "final_suite.yaml"
    cfg = load_config(c)
    out = Path(cfg["output_root"])
    if not out.is_absolute():
        out = (r / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return Paths(repo=r, suite_dir=s, config_path=c, output_root=out)


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def safe_read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def set_low_vram_environment() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")


def variant_id(group: str, seed: int, train_limit: int) -> str:
    return f"{group}/seed_{int(seed)}_train_{int(train_limit)}"


def parse_variant_from_dir(run_dir: Path) -> tuple[int | None, int | None]:
    seed = train = None
    for part in run_dir.parts:
        if part.startswith("seed_") and "_train_" in part:
            left, right = part.split("_train_", 1)
            try:
                seed = int(left.replace("seed_", ""))
                train = int(right)
            except ValueError:
                pass
    return seed, train
