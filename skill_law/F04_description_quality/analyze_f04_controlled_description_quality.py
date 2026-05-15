from __future__ import annotations
import os


import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path


DATA_DIR = finding_data_path("F04", "data", "controlled_description_quality")
OUT_DIR = finding_path("F04", "analysis")
FIGURE_DIR = finding_path("figures", "exports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


LEVEL_ORDER = ["L1_name", "L4_examples", "L5_no_boundary_matched", "L5_boundary"]
LEVEL_LABELS = {
    "L1_name": "L1 name",
    "L4_examples": "L4 examples",
    "L5_no_boundary_matched": "L5 length-matched",
    "L5_boundary": "L5 boundary",
}


def load_valid_runs() -> tuple[pd.DataFrame, list[dict]]:
    candidates = sorted(DATA_DIR.glob("controlled_description_quality_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No controlled description-quality summaries in {DATA_DIR}")
    valid_by_model: dict[str, tuple[int, str, Path, dict]] = {}
    for summary_path in candidates:
        summary = json.loads(summary_path.read_text())
        if summary.get("n_rows", 0) <= 0:
            print(f"Skipping empty run {summary_path.name}: {summary.get('validity_note', 'empty')}")
            continue
        tag = summary_path.name.removeprefix("controlled_description_quality_summary_").removesuffix(".json")
        model = str(summary.get("model", "unknown"))
        candidate = (int(summary.get("n_rows", 0)), tag, summary_path, summary)
        if model not in valid_by_model or candidate[:2] > valid_by_model[model][:2]:
            valid_by_model[model] = candidate

    frames = []
    summaries = []
    for _, tag, _, summary in sorted(valid_by_model.values(), key=lambda item: item[1]):
        raw_path = DATA_DIR / f"controlled_description_quality_raw_{tag}.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        df["run_tag"] = tag
        df["model"] = summary.get("model", df.get("model", pd.Series(["unknown"])).iloc[0])
        frames.append(df)
        summaries.append({**summary, "run_tag": tag})
    if frames:
        return pd.concat(frames, ignore_index=True), summaries
    raise RuntimeError("No valid controlled description-quality run found.")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - radius, center + radius


def summarize(df: pd.DataFrame, run_summaries: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    usable = df[df["valid_output"]].copy()
    if usable.empty:
        summary = {
            "models": sorted(df["model"].dropna().unique().tolist()) if "model" in df else [],
            "n_valid_runs": len(run_summaries),
            "n_rows": int(len(df)),
            "n_usable": 0,
            "valid_output_rate": 0.0,
            "error_rate": float((df["chosen"] == "ERROR").mean()) if len(df) and "chosen" in df else np.nan,
            "library_sizes": sorted([int(v) for v in df["library_size"].dropna().unique()]) if "library_size" in df else [],
            "levels": sorted(df["level"].dropna().unique().tolist()) if "level" in df else [],
            "valid_for_analysis": False,
            "interpretation": "diagnostic run only; no parseable routing outputs were available",
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary
    grouped_rows = []
    for (model, level, n), group in usable.groupby(["model", "level", "library_size"]):
        correct = int(group["is_correct"].sum())
        total = int(len(group))
        lo, hi = wilson_ci(correct, total)
        grouped_rows.append(
            {
                "model": model,
                "level": level,
                "level_label": LEVEL_LABELS.get(level, level),
                "library_size": int(n),
                "n": total,
                "accuracy": correct / total if total else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_desc_chars": float(group["target_desc_chars"].mean()),
            }
        )
    by_level_n = pd.DataFrame(grouped_rows)
    by_level_n["level_order"] = by_level_n["level"].map({level: i for i, level in enumerate(LEVEL_ORDER)})
    by_level_n = by_level_n.sort_values(["model", "library_size", "level_order"])

    pooled_rows = []
    for (level, n), group in usable.groupby(["level", "library_size"]):
        correct = int(group["is_correct"].sum())
        total = int(len(group))
        lo, hi = wilson_ci(correct, total)
        pooled_rows.append(
            {
                "model": "pooled",
                "level": level,
                "level_label": LEVEL_LABELS.get(level, level),
                "library_size": int(n),
                "n": total,
                "accuracy": correct / total if total else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_desc_chars": float(group["target_desc_chars"].mean()),
            }
        )
    pooled_by_level_n = pd.DataFrame(pooled_rows)
    pooled_by_level_n["level_order"] = pooled_by_level_n["level"].map({level: i for i, level in enumerate(LEVEL_ORDER)})
    pooled_by_level_n = pooled_by_level_n.sort_values(["library_size", "level_order"])

    contrast_rows = []
    contrast_groupers = [(["model", "library_size"], "model"), (["library_size"], "pooled")]
    for group_cols, mode in contrast_groupers:
        for key, group in usable.groupby(group_cols):
            if mode == "model":
                model, n = key
            else:
                model = "pooled"
                n = key[0] if isinstance(key, tuple) else key
            pivot = group.pivot_table(index=["run_tag", "task_idx"], columns="level", values="is_correct", aggfunc="mean")
            for lhs, rhs, name in [
                ("L4_examples", "L1_name", "L4-L1"),
                ("L5_no_boundary_matched", "L4_examples", "L5matched-L4"),
                ("L5_boundary", "L5_no_boundary_matched", "Boundary-L5matched"),
                ("L5_boundary", "L4_examples", "Boundary-L4"),
            ]:
                if lhs not in pivot or rhs not in pivot:
                    continue
                diff = (pivot[lhs] - pivot[rhs]).dropna()
                contrast_rows.append(
                    {
                        "model": model,
                        "library_size": int(n),
                        "contrast": name,
                        "n_pairs": int(diff.size),
                        "mean_delta": float(diff.mean()) if diff.size else np.nan,
                        "se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else np.nan,
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)

    models = sorted(usable["model"].dropna().unique().tolist())
    summary = {
        "models": models,
        "n_valid_runs": len(run_summaries),
        "n_rows": int(len(df)),
        "n_usable": int(len(usable)),
        "valid_output_rate": float(len(usable) / len(df)) if len(df) else 0.0,
        "error_rate": float((df["chosen"] == "ERROR").mean()) if len(df) else np.nan,
        "library_sizes": sorted([int(v) for v in usable["library_size"].unique()]),
        "levels": [level for level in LEVEL_ORDER if level in set(usable["level"])],
        "interpretation": (
            "controlled same-task same-library description-level intervention; target "
            "description is the manipulated variable"
        ),
    }
    if not contrasts.empty:
        boundary = contrasts[(contrasts["contrast"] == "Boundary-L5matched") & (contrasts["model"] == "pooled")]
        summary["boundary_minus_length_matched_mean"] = float(boundary["mean_delta"].mean()) if len(boundary) else None
    return by_level_n, pooled_by_level_n, contrasts, summary


def plot(by_level_n: pd.DataFrame, pooled_by_level_n: pd.DataFrame, contrasts: pd.DataFrame, summary: dict) -> None:
    if by_level_n.empty and pooled_by_level_n.empty:
        return
    models = summary["models"]
    n_cols = max(2, len(models) + 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.4), squeeze=False)
    axes = axes[0]
    plot_panels = [(model, by_level_n[by_level_n["model"] == model]) for model in models]
    plot_panels.append(("pooled", pooled_by_level_n))
    for ax, (panel_name, panel_df) in zip(axes[:-1], plot_panels):
        for level in LEVEL_ORDER:
            group = panel_df[panel_df["level"] == level]
            if group.empty:
                continue
            yerr = np.vstack([
                group["accuracy"] - group["ci95_low"],
                group["ci95_high"] - group["accuracy"],
            ])
            ax.errorbar(
                group["library_size"],
                group["accuracy"],
                yerr=yerr,
                marker="o",
                linewidth=1.7,
                capsize=3,
                label=LEVEL_LABELS[level],
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Library size N")
        ax.set_title(panel_name)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Routing accuracy")
    axes[0].legend(frameon=False, fontsize=6)

    ax = axes[-1]
    pooled_contrasts = contrasts[contrasts["model"] == "pooled"].copy()
    if not pooled_contrasts.empty:
        plot_df = pooled_contrasts.copy()
        x_labels = [f"N={n}\n{c}" for n, c in zip(plot_df["library_size"], plot_df["contrast"])]
        colors = ["#5875A4" if "Boundary" not in c else "#C85C4A" for c in plot_df["contrast"]]
        ax.bar(np.arange(len(plot_df)), plot_df["mean_delta"], color=colors, alpha=0.86)
        ax.axhline(0, color="black", linewidth=1, linestyle=":")
        ax.set_xticks(np.arange(len(plot_df)), x_labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Paired accuracy delta")
    ax.set_title("pooled contrasts")
    ax.grid(axis="y", alpha=0.22)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Controlled description-quality intervention", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_F04_controlled_description_quality.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "F04_controlled_description_quality.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_legacy_single(by_level_n: pd.DataFrame, contrasts: pd.DataFrame, summary: dict) -> None:
    if not summary["models"]:
        return
    model = summary["models"][0]
    by_level_n = by_level_n[by_level_n["model"] == model]
    contrasts = contrasts[contrasts["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    ax = axes[0]
    for level in LEVEL_ORDER:
        group = by_level_n[by_level_n["level"] == level]
        if group.empty:
            continue
        yerr = np.vstack([
            group["accuracy"] - group["ci95_low"],
            group["ci95_high"] - group["accuracy"],
        ])
        ax.errorbar(
            group["library_size"],
            group["accuracy"],
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=LEVEL_LABELS[level],
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Library size N")
    ax.set_ylabel("Routing accuracy")
    ax.set_title("Controlled target-description levels")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    if not contrasts.empty:
        plot_df = contrasts.copy()
        x_labels = [f"N={n}\n{c}" for n, c in zip(plot_df["library_size"], plot_df["contrast"])]
        colors = ["#5875A4" if "Boundary" not in c else "#C85C4A" for c in plot_df["contrast"]]
        ax.bar(np.arange(len(plot_df)), plot_df["mean_delta"], color=colors, alpha=0.86)
        ax.axhline(0, color="black", linewidth=1, linestyle=":")
        ax.set_xticks(np.arange(len(plot_df)), x_labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Paired accuracy delta")
    ax.set_title("Within-task contrasts")
    ax.grid(axis="y", alpha=0.22)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(f"Controlled description-quality intervention ({model})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F04_controlled_description_quality_single_model.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        df, run_summaries = load_valid_runs()
    except (FileNotFoundError, RuntimeError) as exc:
        summary = {
            "valid_for_analysis": False,
            "note": str(exc),
            "data_dir": str(DATA_DIR),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "F04_controlled_description_quality_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    by_level_n, pooled_by_level_n, contrasts, summary = summarize(df, run_summaries)
    by_level_n.to_csv(OUT_DIR / "F04_controlled_description_quality_by_level_n.csv", index=False)
    pooled_by_level_n.to_csv(OUT_DIR / "F04_controlled_description_quality_pooled_by_level_n.csv", index=False)
    contrasts.to_csv(OUT_DIR / "F04_controlled_description_quality_contrasts.csv", index=False)
    (OUT_DIR / "F04_controlled_description_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not by_level_n.empty or not pooled_by_level_n.empty:
        plot(by_level_n, pooled_by_level_n, contrasts, summary)
        plot_legacy_single(by_level_n, contrasts, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {FIGURE_DIR / 'fig_F04_controlled_description_quality.pdf'}")


if __name__ == "__main__":
    main()
