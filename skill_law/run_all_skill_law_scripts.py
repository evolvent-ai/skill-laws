from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from skill_law.demo_data import ensure_demo_data


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class ScriptRun:
    path: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None


SCRIPT_RUNS = [
    ScriptRun("F01_logarithmic_decay/run_f01_cross_model_scaling_probe.py", ("--model", MODEL, "--demo", "--limit-cases", "5")),
    ScriptRun("F01_logarithmic_decay/run_f01_route_decay_sweep.py", ("--model", MODEL, "--demo", "--limit-cases", "5")),
    ScriptRun("F02_pipeline_compounding/run_f02_transition_cascade_audit.py", ("--model", MODEL, "--limit-per-k", "1", "--k-values", "4", "--workers", "1")),
    ScriptRun("F02_pipeline_compounding/analyze_f02_cascade_penalty_decomposition.py"),
    ScriptRun("F03_rebound_effect/run_f03_step_position_rebound_probe.py", ("--model", MODEL, "--k-values", "4", "--limit-per-k", "1", "--workers", "1")),
    ScriptRun("F03_rebound_effect/analyze_f03_position_profile_rebound.py"),
    ScriptRun("F04_description_quality/run_f04_controlled_description_quality.py", ("--model", MODEL, "--limit", "1", "--library-sizes", "30", "--workers", "1")),
    ScriptRun("F04_description_quality/analyze_f04_controlled_description_quality.py"),
    ScriptRun("F05_local_competition/run_f05_similarity_competition_stress.py", ("--model", MODEL), {"MAX_WORKERS": "1", "NUM_TEST_TASKS": "5", "SKILL_LAW_LEXICAL_EMBEDDINGS": "1"}),
    ScriptRun("F05_local_competition/analyze_f05_local_competition_index.py", (), {"SKILL_LAW_ANALYSIS_RECORD_LIMIT": "200"}),
    ScriptRun("F06_failure_geometry/run_f06_structured_boundary_rewrite.py", ("--model", MODEL, "--limit", "1", "--trials", "1", "--workers", "1", "--no-quality-gate")),
    ScriptRun("F06_failure_geometry/analyze_f06_route_asymmetry_cross_model.py"),
    ScriptRun("F07_anchor_removal_black_hole/run_f07_query_anchor_ablation.py", ("--model", MODEL), {"MAX_WORKERS": "1", "TRIALS": "1", "TARGET_LIMIT": "1", "LIBRARY_SIZES": "100"}),
    ScriptRun("F07_anchor_removal_black_hole/run_f07_real_skills_anchor_stress_n100.py", ("--model", MODEL), {"MAX_WORKERS": "1", "TRIALS": "1", "N_PER_BIN": "1", "N_BINS": "1"}),
    ScriptRun("F08_dual_trigger_protocol/run_f08_dual_trigger_hijack_validation.py", ("--model", MODEL), {"MAX_WORKERS": "1", "TRIALS": "1", "TARGETS_PER_BIN": "1", "N_BINS": "1", "LIBRARY_SIZES": "150"}),
    ScriptRun("F08_dual_trigger_protocol/run_f08_protocol_ablation.py", ("--model", MODEL), {"MAX_WORKERS": "1", "ABLATION_TRIALS": "1", "LIBRARY_SIZES": "150"}),
    ScriptRun("F09_routing_independence/analyze_f09_clustered_routing_independence.py", (), {"SKILL_LAW_BOOTSTRAPS": "100"}),
    ScriptRun("F09_routing_independence/analyze_f09_mixed_effects_independence.py"),
    ScriptRun("F10_execution_rescue/run_f10_conditional_transfer_rescue.py", ("--model", MODEL), {"MAX_WORKERS": "1", "TRIALS": "1", "TASK_LIMIT": "1"}),
    ScriptRun("F10_execution_rescue/run_f10_mass_combo_rescue_probe.py", (), {"PAIRS_PER_BIN": "1", "TRIALS": "1", "N_SIZE": "30", "SKILL_LAW_LEXICAL_EMBEDDINGS": "1"}),
    ScriptRun("F11_context_recovery/run_f11_quality_propagation_v4.py", ("--model", MODEL), {"N_TRIALS": "1", "PAIR_LIMIT": "1"}),
    ScriptRun("F11_context_recovery/run_f11_self_repair_baselines.py", ("--model", MODEL, "--limit-per-dependency", "1", "--trials", "1", "--conditions", "no_upstream_task", "--workers", "1", "--max-retries", "1")),
    ScriptRun("F12_strong_tow_crowding/run_f12_strong_tow_product_baseline.py", ("--model", MODEL), {"N_TRIALS": "1", "PAIR_LIMIT": "1"}),
    ScriptRun("F12_strong_tow_crowding/analyze_f12_strong_tow_crowding.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every F01-F12 skill_law entry script once.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser.parse_args()


def tail(text: str, lines: int = 8) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    selected = text.strip().splitlines()[-lines:]
    return "\n".join(selected)


def main() -> int:
    args = parse_args()
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT / "src")
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    ensure_demo_data()
    failures = 0

    for idx, item in enumerate(SCRIPT_RUNS, start=1):
        script = ROOT / item.path
        env = base_env.copy()
        if item.env:
            env.update(item.env)
        command = [sys.executable, str(script), *item.args]
        print(f"\n[{idx:02d}/{len(SCRIPT_RUNS)}] RUN {item.path} {' '.join(item.args)}", flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            failures += 1
            print("TIMEOUT")
            if exc.stdout:
                print(tail(exc.stdout))
            if exc.stderr:
                print(tail(exc.stderr))
            if args.stop_on_fail:
                return 1
            continue

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode == 0:
            print("PASS")
            if output.strip():
                print(tail(output))
        else:
            failures += 1
            print(f"FAIL returncode={result.returncode}")
            if output.strip():
                print(tail(output))
            if args.stop_on_fail:
                return result.returncode

    passed = len(SCRIPT_RUNS) - failures
    print(f"\nSUMMARY passed={passed} failed={failures} total={len(SCRIPT_RUNS)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
