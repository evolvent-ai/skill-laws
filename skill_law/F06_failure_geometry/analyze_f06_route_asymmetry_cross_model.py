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


DATA_DIR = finding_data_path("F06", "data")
OUT_DIR = finding_path("F06", "analysis")
FIGURE_DIR = finding_path("figures", "exports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def model_name(path: Path) -> str:
    stem = path.stem.replace("descriptor_group_experiment", "")
    return stem.strip("_") or "gpt-5.4-mini"


def append_pair_row(rows: list[dict], model: str, pair_key: str, group_key: str, comparison: str, fwd_comp: float, rev_comp: float, fwd_correct: float, rev_correct: float, condition: str = "before") -> None:
    a_route = fwd_comp - rev_comp
    total_confusion = fwd_comp + rev_comp
    mean_acc = (fwd_correct + rev_correct) / 2.0
    mean_pair_confusion = total_confusion / 2.0
    off_family_miss = max(0.0, 1.0 - mean_acc - mean_pair_confusion)
    abs_a = abs(a_route)
    if abs_a < 0.05 and total_confusion >= 0.30:
        regime = "symmetric_interference"
    elif abs_a < 0.05:
        regime = "low_confusion_symmetric"
    elif a_route > 0:
        regime = "forward_dominant"
    else:
        regime = "reverse_dominant"
    rows.append(
        {
            "model": model,
            "condition": condition,
            "pair_key": pair_key,
            "group_key": group_key,
            "comparison": comparison,
            "a_route": a_route,
            "abs_a_route": abs_a,
            "total_confusion": total_confusion,
            "forward_competition": fwd_comp,
            "reverse_competition": rev_comp,
            "forward_correct": fwd_correct,
            "reverse_correct": rev_correct,
            "mean_accuracy": mean_acc,
            "mean_pair_confusion": mean_pair_confusion,
            "off_family_miss": off_family_miss,
            "regime": regime,
        }
    )


def load_current_jsonl(rows: list[dict]) -> None:
    for path in sorted(DATA_DIR.glob("F06_structured_descriptor_intervention_*.jsonl")):
        model = path.name.replace("F06_structured_descriptor_intervention_", "").split("_T", 1)[0]
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        grouped = {}
        for row in records:
            key = (row["pair_key"], row.get("group_key", ""), row.get("comparison", ""), row.get("condition", "before"))
            grouped.setdefault(key, {}).setdefault(row["direction"], []).append(row)
        for (pair_key, group_key, comparison, condition), by_direction in grouped.items():
            if "forward" not in by_direction or "reverse" not in by_direction:
                continue
            fwd = by_direction["forward"]
            rev = by_direction["reverse"]
            append_pair_row(
                rows,
                model,
                pair_key,
                group_key,
                comparison,
                float(np.mean([r["is_pair_confusion"] for r in fwd])),
                float(np.mean([r["is_pair_confusion"] for r in rev])),
                float(np.mean([r["is_correct"] for r in fwd])),
                float(np.mean([r["is_correct"] for r in rev])),
                condition,
            )


def load_pairs() -> pd.DataFrame:
    rows = []
    load_current_jsonl(rows)
    for path in sorted(DATA_DIR.glob("descriptor_group_experiment*.json")):
        with path.open() as f:
            data = json.load(f)
        model = data.get("model") or model_name(path)
        for p in data["pair_results"]:
            fwd_comp = p["forward"]["competition_rate"]
            rev_comp = p["reverse"]["competition_rate"]
            fwd_correct = p["forward"]["correct_rate"]
            rev_correct = p["reverse"]["correct_rate"]
            append_pair_row(rows, model, p["pair_key"], p["group_key"], p["comparison"], fwd_comp, rev_comp, fwd_correct, rev_correct)
    if not rows:
        raise RuntimeError("No F06 route-asymmetry result files found")
    return pd.DataFrame(rows)


def mean_ci(x: pd.Series) -> tuple[float, float]:
    vals = x.dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return float(vals.mean()) if len(vals) else float("nan"), 0.0
    return float(vals.mean()), float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_model = (
        df.groupby(["model", "regime"], as_index=False)
        .agg(
            n_pairs=("pair_key", "count"),
            mean_accuracy=("mean_accuracy", "mean"),
            total_confusion=("total_confusion", "mean"),
            mean_pair_confusion=("mean_pair_confusion", "mean"),
            off_family_miss=("off_family_miss", "mean"),
            abs_a_route=("abs_a_route", "mean"),
        )
    )
    rows = []
    for regime, g in per_model.groupby("regime"):
        row = {"regime": regime, "models": int(g["model"].nunique()), "n_model_regime": int(len(g))}
        for col in ["mean_accuracy", "total_confusion", "mean_pair_confusion", "off_family_miss", "abs_a_route"]:
            m, ci = mean_ci(g[col])
            row[f"{col}_mean"] = m
            row[f"{col}_ci95"] = ci
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("regime")

    corr_rows = []
    for model, g in df.groupby("model"):
        if len(g) > 2:
            corr = np.corrcoef(g["total_confusion"], g["mean_accuracy"])[0, 1]
            corr_abs = np.corrcoef(g["abs_a_route"], g["mean_accuracy"])[0, 1]
            corr_rows.append((corr, corr_abs))
    corr = np.array(corr_rows)
    headline = {
        "n_models": int(df["model"].nunique()),
        "n_model_pairs": int(len(df)),
        "corr_total_confusion_accuracy_mean": float(np.nanmean(corr[:, 0])),
        "corr_abs_asymmetry_accuracy_mean": float(np.nanmean(corr[:, 1])),
    }
    for regime in ["symmetric_interference", "forward_dominant", "reverse_dominant", "low_confusion_symmetric"]:
        sub = summary[summary["regime"] == regime]
        if not sub.empty:
            headline[f"{regime}_accuracy_mean"] = float(sub.iloc[0]["mean_accuracy_mean"])
            headline[f"{regime}_accuracy_ci95"] = float(sub.iloc[0]["mean_accuracy_ci95"])
            headline[f"{regime}_total_confusion_mean"] = float(sub.iloc[0]["total_confusion_mean"])
            headline[f"{regime}_pair_confusion_mean"] = float(sub.iloc[0]["mean_pair_confusion_mean"])
            headline[f"{regime}_off_family_miss_mean"] = float(sub.iloc[0]["off_family_miss_mean"])
    return summary, headline


def plot(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    regimes = ["symmetric_interference", "forward_dominant", "reverse_dominant", "low_confusion_symmetric"]
    labels = ["Symmetric\ninterference", "Forward\n dominant", "Reverse\n dominant", "Low-confusion\nsymmetric"]
    colors = {
        "symmetric_interference": "#C85C4A",
        "forward_dominant": "#5B8DB8",
        "reverse_dominant": "#7BAF6A",
        "low_confusion_symmetric": "#AAAAAA",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))

    ax = axes[0]
    for i, regime in enumerate(regimes):
        vals = df[df["regime"] == regime]
        if vals.empty:
            continue
        rng = np.random.default_rng(i + 11)
        ax.scatter(rng.normal(i, 0.055, len(vals)), vals["mean_accuracy"] * 100, s=11, alpha=0.22, color=colors[regime])
        per_model = vals.groupby("model")["mean_accuracy"].mean()
        m, ci = mean_ci(per_model)
        ax.errorbar(i, m * 100, yerr=ci * 100, fmt="o", color="black", capsize=3, ms=4)
    ax.set_xticks(range(len(regimes)), labels)
    ax.set_ylabel("Mean routing accuracy (%)")
    ax.set_title("Observed geometry predicts accuracy")

    ax = axes[1]
    per_regime = (
        df.groupby("regime", as_index=False)
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            mean_pair_confusion=("mean_pair_confusion", "mean"),
            off_family_miss=("off_family_miss", "mean"),
        )
        .set_index("regime")
        .reindex(regimes)
    )
    x = np.arange(len(regimes))
    correct = per_regime["mean_accuracy"].fillna(0).to_numpy() * 100
    pair_conf = per_regime["mean_pair_confusion"].fillna(0).to_numpy() * 100
    off_family = per_regime["off_family_miss"].fillna(0).to_numpy() * 100
    ax.bar(x, correct, color="#4E8F5A", label="Correct")
    ax.bar(x, pair_conf, bottom=correct, color="#C85C4A", label="Paired competitor")
    ax.bar(x, off_family, bottom=correct + pair_conf, color="#9A9A9A", label="Other / outside pair")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Outcome share (%)")
    ax.set_title("Failure destination")
    ax.legend(frameon=False, fontsize=6, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_F06_route_asymmetry.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "F06_route_asymmetry_cross_model.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        df = load_pairs()
    except RuntimeError as exc:
        headline = {
            "valid_for_analysis": False,
            "note": str(exc),
            "data_dir": str(DATA_DIR),
        }
        with (OUT_DIR / "F06_route_asymmetry_headline.json").open("w") as f:
            json.dump(headline, f, indent=2)
        print(json.dumps(headline, indent=2))
        return
    summary, headline = summarize(df)
    df.to_csv(OUT_DIR / "F06_route_asymmetry_pairs.csv", index=False)
    summary.to_csv(OUT_DIR / "F06_route_asymmetry_summary.csv", index=False)
    with (OUT_DIR / "F06_route_asymmetry_headline.json").open("w") as f:
        json.dump(headline, f, indent=2)
    plot(df, summary)
    print(json.dumps(headline, indent=2))
    print(f"Wrote {FIGURE_DIR / 'fig_F06_route_asymmetry.pdf'}")


if __name__ == "__main__":
    main()
