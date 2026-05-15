from __future__ import annotations
import os


import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

TRANSITION_DIR = finding_data_path("F02", "data", "transition_audit")
OUT_DIR = finding_path("F02", "analysis")
FIGURE_DIR = finding_path("figures", "exports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

A_MEAN = 0.834
B_MEAN = 0.042
GAMMA_MEAN = 1.41


def p_single(n: int) -> float:
    return max(A_MEAN - B_MEAN * np.log(n), 0.01)


def p_ind(n: int, k: int) -> float:
    return p_single(n) ** k


def p_gamma(n: int, k: int) -> float:
    return p_single(n) ** (GAMMA_MEAN * k)


def p_markov(p_cc: float, p_cw: float, k: int, p0: float) -> float:
    if k <= 0:
        return 1.0
    return p0 * (p_cc ** (k - 1))


def load_summaries() -> list[dict]:
    rows = []
    candidates = list(TRANSITION_DIR.glob("transition_cascade_summary_*.json"))
    candidates.extend(finding_path("F02", "data", "transition_audit").glob("transition_cascade_summary_*.json"))
    for path in sorted(set(candidates)):
        with path.open() as f:
            d = json.load(f)
        rows.append(d)
    return rows


def compute_cascade_penalty(summaries: list[dict]) -> list[dict]:
    results = []
    for s in summaries:
        n = s["library_size"]
        p_cc = s["p_correct_after_correct"]
        p_cw = s["p_correct_after_wrong"]
        if p_cc is None:
            p_cc = p_single(n)
        if p_cw is None:
            p_cw = p_single(n)
        p0 = p_single(n)

        for k_str, byk in s.get("by_k", {}).items():
            k = int(k_str)
            p_emp_ind = p_ind(n, k)
            p_emp_gamma = p_gamma(n, k)
            p_emp_markov = p_markov(p_cc, p_cw, k, p0)

            penalty_gamma = np.log(max(p_emp_ind, 1e-9)) - np.log(max(p_emp_gamma, 1e-9))
            penalty_markov = np.log(max(p_emp_ind, 1e-9)) - np.log(max(p_emp_markov, 1e-9))

            results.append({
                "model": s["model"],
                "N": n,
                "K": k,
                "p_single": p0,
                "p_ind": p_emp_ind,
                "p_gamma": p_emp_gamma,
                "p_markov": p_emp_markov,
                "penalty_gamma": penalty_gamma,
                "penalty_markov": penalty_markov,
                "rho_c": byk["rho_c"],
                "p_cc": byk["p_correct_after_correct"],
                "p_cw": byk["p_correct_after_wrong"],
            })
    return results


def plot_cascade_penalty(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    ax = axes[0]
    for n_val, grp in df.groupby("N"):
        grp_k = grp.groupby("K")[["penalty_gamma", "penalty_markov"]].mean().reset_index()
        ax.plot(grp_k["K"], grp_k["penalty_gamma"], "o-", label=f"N={n_val} (γ-fit)", lw=1.5)
        ax.plot(grp_k["K"], grp_k["penalty_markov"], "s--", label=f"N={n_val} (Markov)", lw=1.0, alpha=0.7)
    ax.set_xlabel("Pipeline depth K")
    ax.set_ylabel(r"$\log P_{\mathrm{ind}} - \log P$")
    ax.set_title("Cascade Penalty vs. Depth")
    ax.legend(frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.scatter(df["penalty_markov"], df["penalty_gamma"], s=20, alpha=0.6, color="#5B8DB8")
    lo = min(df["penalty_markov"].min(), df["penalty_gamma"].min()) - 0.05
    hi = max(df["penalty_markov"].max(), df["penalty_gamma"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("Markov-chain penalty")
    ax.set_ylabel("γ-fit penalty")
    ax.set_title("Markov vs. γ-Fit Penalty")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.scatter(df["rho_c"], df["penalty_gamma"], s=20, alpha=0.6, color="#C85C4A")
    ax.set_xlabel(r"Transition contrast $\rho_c$")
    ax.set_ylabel("γ-fit cascade penalty")
    ax.set_title(r"$\rho_c$ vs. Cascade Penalty")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for out in [
        OUT_DIR / "F02_cascade_penalty_decomposition.pdf",
        FIGURE_DIR / "fig_F02_cascade_penalty.pdf",
    ]:
        fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT_DIR / "F02_cascade_penalty_decomposition.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} transition summaries")

    rows = compute_cascade_penalty(summaries)
    print(f"Computed {len(rows)} (N, K) penalty rows")

    out_path = OUT_DIR / "F02_cascade_penalty_result.json"
    if not rows:
        result = {
            "n_summaries": len(summaries),
            "n_rows": 0,
            "valid_for_analysis": False,
            "note": "No transition summaries contained enough valid rows for penalty decomposition.",
        }
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"No analyzable rows; wrote {out_path}")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\nCascade penalty by K (mean across N and models):")
    print(df.groupby("K")[["penalty_gamma", "penalty_markov", "rho_c"]].mean().round(4).to_string())

    print("\nCascade penalty by N (mean across K and models):")
    print(df.groupby("N")[["penalty_gamma", "penalty_markov"]].mean().round(4).to_string())

    corr = df["penalty_markov"].corr(df["penalty_gamma"])
    print(f"\nCorrelation(Markov penalty, γ-fit penalty) = {corr:.3f}")

    ratio = (df["penalty_markov"] / df["penalty_gamma"].clip(lower=1e-9)).median()
    print(f"Median Markov/γ-fit ratio = {ratio:.3f}")

    result = {
        "n_summaries": len(summaries),
        "n_rows": len(rows),
        "penalty_by_k": df.groupby("K")[["penalty_gamma", "penalty_markov", "rho_c"]].mean().round(4).to_dict(),
        "penalty_by_n": df.groupby("N")[["penalty_gamma", "penalty_markov"]].mean().round(4).to_dict(),
        "markov_gamma_corr": round(corr, 4),
        "median_markov_gamma_ratio": round(float(ratio), 4),
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    plot_cascade_penalty(rows)
    print(f"\nWrote {out_path}")
    print(f"Wrote {FIGURE_DIR / 'fig_F02_cascade_penalty.pdf'}")


if __name__ == "__main__":
    main()
