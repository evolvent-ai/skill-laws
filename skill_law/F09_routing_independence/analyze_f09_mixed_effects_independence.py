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

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

STRICT_MODELS = {"gpt-5.4", "gpt-5.4-mini"}


def jaccard(a: str, b: str) -> float:
    ta = set(a.replace("-", " ").lower().split())
    tb = set(b.replace("-", " ").lower().split())
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("cooperation_large_scale_*.json")):
        model = path.stem.replace("cooperation_large_scale_", "")
        with path.open() as f:
            data = json.load(f)
        for r in data:
            rows.append({
                "model": model,
                "pair_id": f"{r['skill_a']}__{r['skill_b']}",
                "skill_a": r["skill_a"],
                "skill_b": r["skill_b"],
                "p_a": float(r["p_a"]),
                "p_b": float(r["p_b"]),
                "p_joint": float(r["p_c"]),
                "delta": float(r["p_c"]) - float(r["p_a"]) * float(r["p_b"]),
                "coherence": jaccard(r["skill_a"], r["skill_b"]),
                "strict_format": int(model in STRICT_MODELS),
            })
    df = pd.DataFrame(rows)
    coh_std = max(float(df["coherence"].std()), 1e-9)
    df["coherence_z"] = (df["coherence"] - df["coherence"].mean()) / coh_std
    return df


def fit_mixed_effects(df: pd.DataFrame) -> dict:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not available"}

    try:
        md1 = smf.mixedlm("delta ~ coherence_z + strict_format", df, groups=df["model"])
        mdf1 = md1.fit(reml=True, method="lbfgs")
        coef_coherence = float(mdf1.params["coherence_z"])
        se_coherence = float(mdf1.bse["coherence_z"])
        coef_format = float(mdf1.params["strict_format"])
        se_format = float(mdf1.bse["strict_format"])
        coef_intercept = float(mdf1.params["Intercept"])
        model1_ok = True
    except Exception as e:
        coef_coherence = se_coherence = coef_format = se_format = coef_intercept = float("nan")
        model1_ok = False
        model1_err = str(e)

    try:
        md2 = smf.mixedlm("delta ~ coherence_z + strict_format", df, groups=df["pair_id"])
        mdf2 = md2.fit(reml=True, method="lbfgs")
        coef_coherence_pair = float(mdf2.params["coherence_z"])
        se_coherence_pair = float(mdf2.bse["coherence_z"])
        model2_ok = True
    except Exception as e:
        coef_coherence_pair = se_coherence_pair = float("nan")
        model2_ok = False

    result = {
        "n_obs": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_pairs": int(df["pair_id"].nunique()),
        "coherence_mean": float(df["coherence"].mean()),
        "coherence_std": float(df["coherence"].std()),
        "strict_format_n": int(df[df["strict_format"] == 1]["model"].nunique()),
        "model1_random_by_model": {
            "ok": model1_ok,
            "intercept": coef_intercept,
            "coherence_z_coef": coef_coherence,
            "coherence_z_se": se_coherence,
            "coherence_z_ci95_lo": coef_coherence - 1.96 * se_coherence,
            "coherence_z_ci95_hi": coef_coherence + 1.96 * se_coherence,
            "strict_format_coef": coef_format,
            "strict_format_se": se_format,
            "strict_format_ci95_lo": coef_format - 1.96 * se_format,
            "strict_format_ci95_hi": coef_format + 1.96 * se_format,
        },
        "model2_random_by_pair": {
            "ok": model2_ok,
            "coherence_z_coef": coef_coherence_pair,
            "coherence_z_se": se_coherence_pair,
        },
    }
    if not model1_ok:
        result["model1_random_by_model"]["error"] = model1_err
    return result


def plot_coherence_delta(df: pd.DataFrame, result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    ax = axes[0]
    rng = np.random.default_rng(42)
    ax.scatter(
        df["coherence"] + rng.normal(0, 0.005, len(df)),
        df["delta"],
        s=8, alpha=0.18, color="#5B8DB8", linewidths=0,
    )
    model_means = df.groupby("model")[["coherence", "delta"]].mean()
    ax.scatter(model_means["coherence"], model_means["delta"],
               s=40, color="#C85C4A", zorder=5, label="model mean")
    m1 = result.get("model1_random_by_model", {})
    if m1.get("ok"):
        coh_std = df["coherence"].std()
        coh_mean = df["coherence"].mean()
        xs = np.linspace(df["coherence"].min(), df["coherence"].max(), 100)
        xs_z = (xs - coh_mean) / max(coh_std, 1e-9)
        ys = m1["intercept"] + m1["coherence_z_coef"] * xs_z
        ax.plot(xs, ys, color="#333333", lw=1.5, ls="--",
                label=f"ME slope={m1['coherence_z_coef']:.3f}")
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Coherence (Jaccard)")
    ax.set_ylabel(r"$\Delta = P(A,B) - P(A)P(B)$")
    ax.set_title("Coherence vs. Independence Residual")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    for i, (label, mask) in enumerate([
        ("Plain-text", df["strict_format"] == 0),
        ("Strict FC", df["strict_format"] == 1),
    ]):
        vals = df[mask].groupby("model")["delta"].mean()
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(i, mean, color=["#5B8DB8", "#C85C4A"][i], alpha=0.85)
        ax.errorbar(i, mean, yerr=1.96 * se, fmt="none", color="black", capsize=4)
        ax.text(i, mean + 0.005, f"n={len(vals)}m", ha="center", fontsize=7)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xticks([0, 1], ["Plain-text\nrouting", "Strict\nfunction call"])
    ax.set_ylabel(r"Mean $\Delta$ (model-clustered)")
    ax.set_title("Format Protocol Effect")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for out in [
        OUT_DIR / "F09_mixed_effects_coherence.pdf",
        FIGURE_DIR / "fig_F09_mixed_effects.pdf",
    ]:
        fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT_DIR / "F09_mixed_effects_coherence.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load()
    result = fit_mixed_effects(df)

    out_path = OUT_DIR / "F09_mixed_effects_result.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    plot_coherence_delta(df, result)
    print(f"\nWrote {out_path}")
    print(f"Wrote {FIGURE_DIR / 'fig_F09_mixed_effects.pdf'}")


if __name__ == "__main__":
    main()
