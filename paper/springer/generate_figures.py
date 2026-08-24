"""Compatibility entry point for synchronized manuscript figures."""

from pathlib import Path
import runpy


runpy.run_path(
    Path(__file__).resolve().parents[1] / "generate_benchmark_figures.py",
    run_name="__main__",
)
