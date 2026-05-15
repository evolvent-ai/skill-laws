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


DATA_DIR = finding_data_path("F12", "data")
OUT_DIR = finding_path("F12", "analysis")
FIG_DIR = finding_path("F12", "figures")
FIGURE_DIR = finding_path("figures", "exports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def model_name(path: Path) -> str:
    return path.stem.replace("F12_raw_", "")


def load_all() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("F12_raw_*.jsonl")):
        model = model_name(path)
        seen = set()
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (model, rec["a_id"], rec["b_id"], rec["dependency"], rec["trial"])
                if key in seen:
                    continue
                seen.add(key)
                rec["model"] = model
                rows.append(rec)
    if not rows:
        raise RuntimeError(f"No F12_raw_*.jsonl files found in {DATA_DIR}")
    df = pd.DataFrame(rows)
    df["gap"] = (df["s_a"] - df["s_b"]).abs()
    df["s_product"] = df["s_a"] * df["s_b"]
    df["s_avg"] = (df["s_a"] + df["s_b"]) / 2.0
    df["s_weak"] = df[["s_a", "s_b"]].min(axis=1)
    df["s_strong"] = df[["s_a", "s_b"]].max(axis=1)
    df["sx"] = df["s_joint"] - df["s_product"]
    df["old_arithmetic_residual"] = df["s_joint"] - df["s_avg"]
    df["below_weak"] = df["s_joint"] < df["s_weak"]
    denom = ((df["out_a_len"] + df["out_b_len"]) / 2.0).clip(lower=1)
    df["len_ratio"] = df["out_joint_len"] / denom
    df["compressed"] = df["len_ratio"] < 0.7
    return df


def gap_regime(gap: float) -> str:
    if gap < 0.10:
        return "small"
    if gap <= 0.30:
        return "medium"
    return "large"


def mean_ci(values: pd.Series) -> pd.Series:
    x = values.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return pd.Series({"mean": np.nan, "ci95": np.nan})
    if len(x) == 1:
        return pd.Series({"mean": float(x[0]), "ci95": 0.0})
    return pd.Series(
        {
            "mean": float(np.mean(x)),
            "ci95": float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))),
        }
    )


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pair_cols = ["model", "a_id", "b_id", "dependency"]
    pairs = (
        df.groupby(pair_cols, as_index=False)
        .agg(
            n_trials=("trial", "count"),
            gap=("gap", "mean"),
            s_a=("s_a", "mean"),
            s_b=("s_b", "mean"),
            s_joint=("s_joint", "mean"),
            s_product=("s_product", "mean"),
            s_weak=("s_weak", "mean"),
            sx=("sx", "mean"),
            old_arithmetic_residual=("old_arithmetic_residual", "mean"),
            below_weak_rate=("below_weak", "mean"),
            len_ratio=("len_ratio", "mean"),
            compressed_rate=("compressed", "mean"),
        )
    )
    pairs["gap_regime"] = pairs["gap"].map(gap_regime)

    model_regime = (
        pairs.groupby(["model", "gap_regime"], as_index=False)
        .agg(
            n_pairs=("sx", "count"),
            sx=("sx", "mean"),
            old_arithmetic_residual=("old_arithmetic_residual", "mean"),
            below_weak_rate=("below_weak_rate", "mean"),
            compressed_rate=("compressed_rate", "mean"),
            len_ratio=("len_ratio", "mean"),
        )
    )

    regime_rows = []
    for regime, g in model_regime.groupby("gap_regime"):
        row = {"gap_regime": regime, "models": int(g["model"].nunique())}
        for col in ["sx", "old_arithmetic_residual", "below_weak_rate", "compressed_rate", "len_ratio"]:
            stats = mean_ci(g[col])
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_ci95"] = stats["ci95"]
        row["sx_positive_models"] = int((g["sx"] > 0).sum())
        row["sx_negative_models"] = int((g["sx"] < 0).sum())
        row["sx_zero_models"] = int((g["sx"] == 0).sum())
        regime_rows.append(row)
    regime_summary = pd.DataFrame(regime_rows)

    dep_gap = (
        pairs.groupby(["model", "dependency", "gap_regime"], as_index=False)
        .agg(
            n_pairs=("sx", "count"),
            sx=("sx", "mean"),
            below_weak_rate=("below_weak_rate", "mean"),
            compressed_rate=("compressed_rate", "mean"),
        )
    )

    dep_rows = []
    for (dep, regime), g in dep_gap.groupby(["dependency", "gap_regime"]):
        row = {"dependency": dep, "gap_regime": regime, "models": int(g["model"].nunique())}
        for col in ["sx", "below_weak_rate", "compressed_rate"]:
            stats = mean_ci(g[col])
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_ci95"] = stats["ci95"]
        dep_rows.append(row)
    dep_summary = pd.DataFrame(dep_rows)

    by_regime = {str(row["gap_regime"]): row for _, row in regime_summary.iterrows()}

    def value(regime: str, key: str, default: float = float("nan")) -> float:
        row = by_regime.get(regime)
        if row is None:
            return default
        return float(row.get(key, default))

    def count(regime: str, key: str) -> int:
        row = by_regime.get(regime)
        if row is None:
            return 0
        return int(row.get(key, 0))

    headline = {
        "n_models": int(df["model"].nunique()),
        "n_trials": int(len(df)),
        "n_model_pairs": int(len(pairs)),
        "large_gap_product_synergy_mean": value("large", "sx_mean"),
        "large_gap_product_synergy_ci95": value("large", "sx_ci95"),
        "small_gap_product_synergy_mean": value("small", "sx_mean"),
        "small_gap_product_synergy_ci95": value("small", "sx_ci95"),
        "small_gap_below_weak_rate_mean": value("small", "below_weak_rate_mean"),
        "small_gap_below_weak_rate_ci95": value("small", "below_weak_rate_ci95"),
        "small_gap_old_arithmetic_residual_mean": value("small", "old_arithmetic_residual_mean"),
        "large_gap_old_arithmetic_residual_mean": value("large", "old_arithmetic_residual_mean"),
        "large_gap_sx_positive_models": count("large", "sx_positive_models"),
        "large_gap_sx_negative_models": count("large", "sx_negative_models"),
        "small_gap_sx_positive_models": count("small", "sx_positive_models"),
        "small_gap_sx_negative_models": count("small", "sx_negative_models"),
        "small_gap_sx_zero_models": count("small", "sx_zero_models"),
    }
    return pairs, pd.concat([regime_summary.assign(kind="gap"), dep_summary.assign(kind="dep_gap")], ignore_index=True), headline


def plot(pairs: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = ["small", "medium", "large"]
    labels = ["Small\nG<0.10", "Medium\n0.10-0.30", "Large\nG>0.30"]
    colors = {"small": "#5B8DB8", "medium": "#7BAF6A", "large": "#C85C4A"}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))

    ax = axes[0]
    rng = np.random.default_rng(7)
    for i, regime in enumerate(order):
        g = pairs[pairs["gap_regime"] == regime]
        x = rng.normal(i, 0.055, size=len(g))
        ax.scatter(x, g["sx"], s=12, alpha=0.23, color=colors[regime], linewidths=0)
        model_stats = (
            g.groupby("model")["sx"].mean().reset_index()["sx"]
        )
        stats = mean_ci(model_stats)
        ax.errorbar(i, stats["mean"], yerr=stats["ci95"], fmt="o", color="black", capsize=3, ms=4)
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel(r"Product residual $S_\times$")
    ax.set_title("Independence baseline")

    ax = axes[1]
    for i, regime in enumerate(order):
        g = pairs[pairs["gap_regime"] == regime]
        model_stats = g.groupby("model")["below_weak_rate"].mean().reset_index()["below_weak_rate"]
        stats = mean_ci(model_stats)
        ax.bar(i, stats["mean"] * 100, color=colors[regime], alpha=0.85)
        ax.errorbar(i, stats["mean"] * 100, yerr=stats["ci95"] * 100, fmt="none", color="black", capsize=3)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Below-weak rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Crowding risk")

    ax = axes[2]
    for dep, marker in [("tight", "o"), ("loose", "s"), ("independent", "^")]:
        g = pairs[pairs["dependency"] == dep]
        if g.empty:
            continue
        model_dep = g.groupby(["model", "gap_regime"], as_index=False)["sx"].mean()
        ys = []
        es = []
        for regime in order:
            vals = model_dep[model_dep["gap_regime"] == regime]["sx"]
            stats = mean_ci(vals)
            ys.append(stats["mean"])
            es.append(stats["ci95"])
        ax.errorbar(range(3), ys, yerr=es, marker=marker, lw=1.3, capsize=3, label=dep)
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel(r"$S_\times$ by dependency")
    ax.set_title("Mechanism stratification")
    ax.legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)

    fig.tight_layout()
    for out in [
        FIG_DIR / "F12_product_baseline_cross_model.pdf",
        FIGURE_DIR / "fig_F12_product_baseline.pdf",
    ]:
        fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG_DIR / "F12_product_baseline_cross_model.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        df = load_all()
    except RuntimeError as exc:
        headline = {
            "valid_for_analysis": False,
            "note": str(exc),
            "data_dir": str(DATA_DIR),
        }
        with (OUT_DIR / "F12_product_headline.json").open("w") as f:
            json.dump(headline, f, indent=2)
        print(json.dumps(headline, indent=2))
        return
    pairs, summary, headline = summarize(df)
    pairs.to_csv(OUT_DIR / "F12_product_pair_summary.csv", index=False)
    summary.to_csv(OUT_DIR / "F12_product_summary.csv", index=False)
    with (OUT_DIR / "F12_product_headline.json").open("w") as f:
        json.dump(headline, f, indent=2)
    plot(pairs, summary)

    print(json.dumps(headline, indent=2))
    print(f"Wrote {OUT_DIR / 'F12_product_pair_summary.csv'}")
    print(f"Wrote {FIGURE_DIR / 'fig_F12_product_baseline.pdf'}")


if __name__ == "__main__":
    main()
