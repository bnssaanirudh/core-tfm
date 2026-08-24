from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q1_fast_complete_256_v1"
JMLR = ROOT / "paper" / "jmlr"
SPRINGER = ROOT / "paper" / "springer" / "figures"


def save(fig: plt.Figure, name: str) -> None:
    for directory in (JMLR, SPRINGER):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)


def exact_view_regret() -> None:
    data = pd.read_csv(RESULTS / "controlled_replications" / "exact_view_summary.csv")
    methods = ["selected_order", "arithmetic", "hard_core", "adaptive_soft"]
    labels = {
        "selected_order": "Selected order",
        "arithmetic": "Arithmetic",
        "hard_core": "Hard CoRe",
        "adaptive_soft": "Adaptive Soft",
    }
    regimes = [
        "uniform_low_noise", "marginals_reliable", "conditionals_reliable",
        "j1_reliable", "j2_reliable", "all_noisy",
    ]
    regime_labels = [
        "Low noise", "Marginals reliable", "Conditionals reliable",
        "$J_1$ reliable", "$J_2$ reliable", "All noisy",
    ]
    pivot = data.pivot(index="regime", columns="method", values="expected_nll_mean")
    regret = pivot.sub(pivot.min(axis=1), axis=0)
    x = np.arange(len(regimes))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, regret.loc[regimes, method], width, label=labels[method])
    ax.set_ylabel("Expected NLL regret")
    ax.set_xticks(x, regime_labels, rotation=18, ha="right")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7)
    fig.tight_layout()
    save(fig, "Fig1_exact_view_regret.pdf")


def mixture_gain() -> None:
    data = pd.read_csv(RESULTS / "controlled_replications" / "selective_mixture.csv")
    delta = np.sort(data["nll_selective_core"] - data["nll_arithmetic"])
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    colors = np.where(delta < 0, "#2a6fbb", "#c44e52")
    ax.bar(np.arange(1, len(delta) + 1), delta, color=colors, width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Task (sorted)", ylabel="Selective minus arithmetic NLL")
    fig.tight_layout()
    save(fig, "Fig2_selective_task_gain.pdf")


def validation_size() -> None:
    data = pd.read_csv(RESULTS / "controlled_replications" / "validation_size_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(data["n_val"], data["mean_full_oracle_regret"], marker="o", linewidth=2)
    ax.set(xlabel="Validation observations", ylabel="Mean full-oracle regret")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "Fig3_validation_size.pdf")


def real_dataset_effects() -> None:
    data = pd.read_csv(RESULTS / "primary_effects_by_dataset.csv").sort_values(
        "selective_minus_arithmetic"
    )
    y = np.arange(len(data))
    values = data["selective_minus_arithmetic"].to_numpy()
    colors = np.where(values < 0, "#2a6fbb", "#c44e52")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, data["dataset"].str.title())
    ax.set_xlabel("Selective minus arithmetic joint NLL")
    fig.tight_layout()
    save(fig, "Fig4_real_dataset_effects.pdf")


if __name__ == "__main__":
    exact_view_regret()
    mixture_gain()
    validation_size()
    real_dataset_effects()

