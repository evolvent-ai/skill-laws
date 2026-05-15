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


DATA_DIR = finding_data_path("F09", "data", "processed")
OUT_DIR = finding_path("F09", "analysis")
FIGURE_DIR = finding_path("figures", "exports")
N_BOOT = int(os.environ.get("SKILL_LAW_BOOTSTRAPS", "5000"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def model_from_path(path: Path) -> str:
    return path.stem.replace("cooperation_large_scale_", "")


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("cooperation_large_scale_*.json")):
        model = model_from_path(path)
        with path.open() as f:
            data = json.load(f)
        for r in data:
            rows.append(
                {
                    "model": model,
                    "pair_id": f"{r['skill_a']}__{r['skill_b']}",
                    "skill_a": r["skill_a"],
                    "skill_b": r["skill_b"],
                    "p_a": float(r["p_a"]),
                    "p_b": float(r["p_b"]),
                    "p_joint": float(r["p_c"]),
                    "p_product": float(r["p_a"]) * float(r["p_b"]),
                    "delta": float(r["p_c"]) - float(r["p_a"]) * float(r["p_b"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No cooperation_large_scale_*.json files found in {DATA_DIR}")
    df = pd.DataFrame(rows)
    df["valid_product"] = df["p_product"] > 0
    df["degenerate"] = (df["p_a"].isin([0.0, 1.0])) & (df["p_b"].isin([0.0, 1.0]))
    return df


def ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def bootstrap_cluster(df: pd.DataFrame, cluster_col: str, n_boot: int | None = None, seed: int = 17) -> dict:
    n_boot = N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].drop_duplicates().to_numpy()
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([df[df[cluster_col] == c] for c in sampled], ignore_index=True)
        means.append(float(boot["delta"].mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(df["delta"].mean()), "ci_low": float(lo), "ci_high": float(hi), "n_clusters": int(len(clusters))}


def summarize(df: pd.DataFrame) -> dict:
    by_model = df.groupby("model")["delta"].mean()
    by_pair = df.groupby("pair_id")["delta"].mean()
    model_mean, model_ci = ci(by_model.to_numpy())
    pair_mean, pair_ci = ci(by_pair.to_numpy())
    nondeg = df[~df["degenerate"]].copy()
    valid = df[df["valid_product"]].copy()
    leave_one_model = []
    for model in sorted(df["model"].unique()):
        subset = df[df["model"] != model]
        leave_one_model.append(
            {
                "held_out_model": model,
                "n_models": int(subset["model"].nunique()),
                "delta_mean": float(subset["delta"].mean()),
                "model_cluster_mean": float(subset.groupby("model")["delta"].mean().mean()),
            }
        )
    loo_df = pd.DataFrame(leave_one_model)
    most_negative = str(by_model.idxmin())
    without_most_negative = df[df["model"] != most_negative]

    return {
        "n_models": int(df["model"].nunique()),
        "n_pairs": int(df["pair_id"].nunique()),
        "n_model_pairs": int(len(df)),
        "raw_delta_mean": float(df["delta"].mean()),
        "model_cluster_mean": model_mean,
        "model_cluster_ci95": model_ci,
        "pair_cluster_mean": pair_mean,
        "pair_cluster_ci95": pair_ci,
        "pair_bootstrap": bootstrap_cluster(df, "pair_id"),
        "model_bootstrap": bootstrap_cluster(df, "model"),
        "nondegenerate_n": int(len(nondeg)),
        "nondegenerate_delta_mean": float(nondeg["delta"].mean()) if len(nondeg) else None,
        "valid_product_n": int(len(valid)),
        "valid_product_delta_mean": float(valid["delta"].mean()) if len(valid) else None,
        "most_negative_model": most_negative,
        "delta_without_most_negative_model": float(without_most_negative["delta"].mean()),
        "model_cluster_ci95_without_most_negative_model": ci(
            without_most_negative.groupby("model")["delta"].mean().to_numpy()
        )[1],
        "leave_one_model_out": leave_one_model,
        "model_delta": {k: float(v) for k, v in by_model.items()},
    }


def plot(df: pd.DataFrame) -> None:
    model_stats = df.groupby("model", as_index=False).agg(delta=("delta", "mean"), n=("delta", "count"))
    model_stats = model_stats.sort_values("delta")
    pair_stats = df.groupby("pair_id", as_index=False).agg(delta=("delta", "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    ax = axes[0]
    y = np.arange(len(model_stats))
    ax.scatter(model_stats["delta"], y, color="#3A6EA5", s=28)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y, model_stats["model"], fontsize=6)
    ax.set_xlabel(r"Mean $\Delta=P(A,B)-P(A)P(B)$")
    ax.set_title("Cluster by model")

    ax = axes[1]
    ax.hist(pair_stats["delta"], bins=24, color="#6AA56A", alpha=0.8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(pair_stats["delta"].mean(), color="#9A3324", lw=1.2)
    ax.set_xlabel(r"Pair-cluster mean $\Delta$")
    ax.set_ylabel("Pairs")
    ax.set_title("Cluster by pair")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_F09_clustered_independence.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "F09_clustered_independence.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load()
    summary = summarize(df)
    df.to_csv(OUT_DIR / "F09_clustered_independence_rows.csv", index=False)
    pd.DataFrame(summary["leave_one_model_out"]).to_csv(
        OUT_DIR / "F09_clustered_independence_leave_one_model_out.csv", index=False
    )
    with (OUT_DIR / "F09_clustered_independence_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    plot(df)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {FIGURE_DIR / 'fig_F09_clustered_independence.pdf'}")


if __name__ == "__main__":
    main()
