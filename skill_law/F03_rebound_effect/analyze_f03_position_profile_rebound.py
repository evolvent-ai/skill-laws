from __future__ import annotations
import os


import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path


DATA_DIR = finding_data_path("F03", "data", "main")
OUT_DIR = finding_path("F03", "analysis")
FIGURE_DIR = finding_path("figures", "exports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


def model_from_path(path: Path) -> str:
    stem = path.stem
    return stem.replace("v2_comprehensive_results_", "")


def add_profile_row(rows: list[dict], model: str, k: int, ordered: list[float]) -> None:
    if len(ordered) != k:
        return
    clipped = np.clip(np.asarray(ordered, dtype=float), EPS, 1.0)
    first = float(clipped[0])
    product_profile = float(np.prod(clipped))
    flat_first = float(first**k)
    mean_step = float(np.mean(clipped))
    flat_mean = float(mean_step**k)
    log_penalty_first = float(-math.log(product_profile / flat_first)) if flat_first > 0 else np.nan
    log_penalty_mean = float(-math.log(product_profile / flat_mean)) if flat_mean > 0 else np.nan
    gamma_profile_first = (
        float(-np.sum(np.log(clipped)) / (-k * math.log(first)))
        if 0 < first < 1
        else np.nan
    )
    gamma_profile_mean = (
        float(-np.sum(np.log(clipped)) / (-k * math.log(mean_step)))
        if 0 < mean_step < 1
        else np.nan
    )
    rows.append(
        {
            "model": model,
            "K": k,
            "first_step_accuracy": first,
            "mean_step_accuracy": mean_step,
            "min_step_accuracy": float(np.min(clipped)),
            "product_position_profile": product_profile,
            "flat_first_step_baseline": flat_first,
            "flat_mean_step_baseline": flat_mean,
            "log_penalty_vs_first_step": log_penalty_first,
            "log_penalty_vs_mean_step": log_penalty_mean,
            "gamma_profile_vs_first_step": gamma_profile_first,
            "gamma_profile_vs_mean_step": gamma_profile_mean,
            "relative_profile_to_first_baseline": product_profile / flat_first,
            "relative_profile_to_mean_baseline": product_profile / flat_mean,
        }
    )


def load_rows() -> pd.DataFrame:
    rows = []
    archive_dir = finding_path("F03", "data", "archive")
    current_path = archive_dir / "step_position_decay.json"
    if current_path.exists():
        data = json.loads(current_path.read_text())
        for k_str, steps in data.items():
            k = int(k_str)
            ordered = [float(steps[str(i)]) for i in range(k) if str(i) in steps]
            add_profile_row(rows, "gpt-5.4-mini", k, ordered)
    for path in sorted(DATA_DIR.glob("v2_comprehensive_results_*.json")):
        if path.name == "v2_comprehensive_results.json":
            continue
        model = model_from_path(path)
        data = json.loads(path.read_text())
        for k_str, steps in data["step_position"].items():
            k = int(k_str)
            ordered = [steps[str(i)]["acc"] for i in sorted(map(int, steps.keys()))]
            add_profile_row(rows, model, k, ordered)
    return pd.DataFrame(rows).sort_values(["model", "K"])


def summarize(rows: pd.DataFrame) -> dict:
    by_k = (
        rows.groupby("K", as_index=False)
        .agg(
            n_models=("model", "nunique"),
            first_step_accuracy_mean=("first_step_accuracy", "mean"),
            mean_step_accuracy_mean=("mean_step_accuracy", "mean"),
            product_profile_median=("product_position_profile", "median"),
            relative_profile_to_first_baseline_median=("relative_profile_to_first_baseline", "median"),
            relative_profile_to_mean_baseline_median=("relative_profile_to_mean_baseline", "median"),
            log_penalty_vs_first_step_mean=("log_penalty_vs_first_step", "mean"),
            log_penalty_vs_first_step_se=("log_penalty_vs_first_step", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            log_penalty_vs_mean_step_mean=("log_penalty_vs_mean_step", "mean"),
            log_penalty_vs_mean_step_se=("log_penalty_vs_mean_step", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            gamma_profile_vs_first_step_mean=("gamma_profile_vs_first_step", "mean"),
            gamma_profile_vs_first_step_se=("gamma_profile_vs_first_step", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            gamma_profile_vs_mean_step_mean=("gamma_profile_vs_mean_step", "mean"),
            gamma_profile_vs_mean_step_se=("gamma_profile_vs_mean_step", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .sort_values("K")
    )
    by_model = (
        rows.groupby("model", as_index=False)
        .agg(
            n_depths=("K", "nunique"),
            log_penalty_vs_mean_step_mean=("log_penalty_vs_mean_step", "mean"),
            gamma_profile_vs_mean_step_mean=("gamma_profile_vs_mean_step", "mean"),
        )
        .sort_values("log_penalty_vs_mean_step_mean", ascending=False)
    )
    summary = {
        "scope": (
            "Position-profile diagnostic only. The source logs lack chain IDs, so the "
            "product of position-wise accuracies is not an observed all-correct rate."
        ),
        "n_models": int(rows["model"].nunique()),
        "n_model_depths": int(len(rows)),
        "by_k": by_k.to_dict(orient="records"),
        "by_model": by_model.to_dict(orient="records"),
    }
    rows.to_csv(OUT_DIR / "F03_position_profile_penalty_rows.csv", index=False)
    by_k.to_csv(OUT_DIR / "F03_position_profile_penalty_by_k.csv", index=False)
    by_model.to_csv(OUT_DIR / "F03_position_profile_penalty_by_model.csv", index=False)
    (OUT_DIR / "F03_position_profile_penalty_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plot(rows: pd.DataFrame, summary: dict) -> None:
    by_k = pd.DataFrame(summary["by_k"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    ax = axes[0]
    for model, group in rows.groupby("model"):
        group = group.sort_values("K")
        ax.plot(group["K"], group["relative_profile_to_mean_baseline"], color="#777777", alpha=0.28, linewidth=0.8)
    ax.plot(
        by_k["K"],
        by_k["relative_profile_to_mean_baseline_median"],
        color="#9A3324",
        marker="o",
        linewidth=2.2,
        label="median across models",
    )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Pipeline length K")
    ax.set_ylabel("Profile product / flat mean-step baseline")
    ax.set_title("Position dispersion penalty")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    x = np.arange(len(by_k))
    y = by_k["gamma_profile_vs_mean_step_mean"]
    err = 1.96 * by_k["gamma_profile_vs_mean_step_se"].fillna(0)
    ax.bar(x, y, yerr=err, color="#3A6EA5", alpha=0.85, capsize=3)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, by_k["K"])
    ax.set_xlabel("Pipeline length K")
    ax.set_ylabel(r"Profile $\gamma$ vs. mean-step baseline")
    ax.set_title("Profile-level penalty exponent")

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("F03 diagnostic: position-profile compounding is below a flat first-step baseline", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_F03_position_profile_penalty.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "F03_position_profile_penalty.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    plot(rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {FIGURE_DIR / 'fig_F03_position_profile_penalty.pdf'}")


if __name__ == "__main__":
    main()
