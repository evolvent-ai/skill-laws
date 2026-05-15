from __future__ import annotations
import os


import argparse
import csv
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path


from skill_law.runtime import (
    build_skill_to_task,
    choose_task_desc,
    load_env,
    load_skill_specs,
    parse_skill_id,
)
from skill_law.runtime import StrictLLMRouter as LLMRouter
from skill_law.runtime import LibrarySpec
from skill_law.runtime import TaskSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path
from skill_law.runtime import load_env


TASKS_FILE = finding_data_path("F01", "data", "benchmark_tasks.json")
TARGETS_FILE = finding_data_path("F06", "analysis", "F06_counterexample_intervention_targets.csv")
OUT_DIR = finding_path("F06", "data")

DESCRIPTOR_RULES = {
    "fast": "Use when the task emphasizes speed, low latency, quick turnaround, or optimized execution.",
    "quick": "Use when the task asks for a quick, lightweight, first-pass, or minimal-effort operation.",
    "smart": "Use when the task needs adaptive judgment, parallel comparison, or choosing among alternatives.",
    "advanced": "Use when the task asks for a complex, high-capability, feature-rich, or expert-level operation.",
    "deep": "Use when the task asks for thorough analysis, exhaustive search, multi-pass inspection, or depth.",
    "secure": "Use when the task emphasizes security, permissions, privacy, validation, or risk controls.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--total-n", type=int, default=48)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--min-valid-rate", type=float, default=0.95)
    parser.add_argument("--no-quality-gate", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "::".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def load_targets(limit: int) -> list[dict]:
    with TARGETS_FILE.open() as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def descriptor_rule(skill_id: str) -> str:
    descriptor = parse_skill_id(skill_id)["descriptor"]
    return DESCRIPTOR_RULES.get(
        descriptor,
        f"Use when the task explicitly matches the '{descriptor}' descriptor in the tool ID.",
    )


def rewrite_description(skill, competitor_id: str) -> str:
    parsed = parse_skill_id(skill.id)
    competitor = parse_skill_id(competitor_id)
    function_root = parsed["function_root"].replace("-", " ")
    competitor_descriptor = competitor["descriptor"]
    own_descriptor = parsed["descriptor"]
    return f"""---
name: {skill.id}
description: {function_root} tool with explicit descriptor-gated routing rules.
---

Primary function: {function_root}.
Positive selection rule for {skill.id}: {descriptor_rule(skill.id)}
Boundary against {competitor_id}: do not select {skill.id} when the task is better
matched by the '{competitor_descriptor}' descriptor; select {competitor_id} for those
cases. Compare descriptor intent before comparing generic function overlap.
Example positive cue for {skill.id}: a task that asks for {own_descriptor} {function_root}.
Example excluded cue: a task that asks for {competitor_descriptor} {function_root}.
"""


def boundary_description(skill, competitor_id: str) -> str:
    return (
        skill.description
        + "\n\nBoundary counterexample: do not use this skill for tasks that are better "
        f"matched by {competitor_id}. If the task names or implies {competitor_id}'s narrower "
        f"operation, select {competitor_id} instead."
    )


def make_library(skills_map: dict, skill_ids: list[str], condition: str, first: str, second: str) -> LibrarySpec:
    skills = []
    for sid in skill_ids:
        skill = skills_map[sid]
        if condition == "boundary":
            if sid == first:
                skill = replace(skill, description=boundary_description(skill, second))
            elif sid == second:
                skill = replace(skill, description=boundary_description(skill, first))
        elif condition == "structured":
            if sid == first:
                skill = replace(skill, description=rewrite_description(skill, second))
            elif sid == second:
                skill = replace(skill, description=rewrite_description(skill, first))
        skills.append(skill)
    random.shuffle(skills)
    return LibrarySpec(id=f"structured_{condition}_{first}_{second}", skills=skills)


def run_trial(job: dict) -> dict:
    router = LLMRouter(model_name=job["model"])
    chosen, meta = router.route(job["task_spec"], job["library"])
    valid_ids = {s.id for s in job["library"].skills}
    return {
        "model": job["model"],
        "pair_key": job["pair_key"],
        "group_key": job["group_key"],
        "comparison": job["comparison"],
        "condition": job["condition"],
        "direction": job["direction"],
        "trial": job["trial"],
        "target": job["target"],
        "competitor": job["competitor"],
        "chosen": chosen,
        "is_correct": chosen == job["target"],
        "is_pair_confusion": chosen == job["competitor"],
        "is_outside_miss": chosen not in {job["target"], job["competitor"]},
        "is_valid_choice": chosen in valid_ids,
        "task_desc": job["task_spec"].instruction,
        "router_meta": meta,
        "timestamp": time.time(),
    }


def main() -> None:
    load_env()
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"F06_structured_descriptor_intervention_{args.model}_T{args.trials}_P{args.limit}.jsonl"

    skills_map = load_skill_specs()
    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    task_map = build_skill_to_task(tasks)
    targets = load_targets(args.limit)

    excluded = {sid for row in targets for sid in row["pair_key"].split("__")}
    filler_pool = sorted(sid for sid in skills_map if sid not in excluded)

    jobs = []
    for row in targets:
        first, second = row["pair_key"].split("__")
        if first not in skills_map or second not in skills_map:
            continue
        pair_ids = [first, second]
        for direction, target, competitor in [
            ("forward", first, second),
            ("reverse", second, first),
        ]:
            task_desc = choose_task_desc(target, task_map, skills_map, use_function_keywords=True)
            rng = random.Random(stable_seed(args.seed, row["pair_key"], direction))
            for trial in range(args.trials):
                fillers = rng.sample(filler_pool, args.total_n - 2)
                skill_ids = pair_ids + fillers
                for condition in ["before", "boundary", "structured"]:
                    library = make_library(skills_map, skill_ids, condition, first, second)
                    task = TaskSpec(
                        id=f"{row['pair_key']}_{direction}_{condition}_{trial}",
                        instruction=task_desc,
                        required_skills=[target],
                        gold_trace=[],
                    )
                    jobs.append(
                        {
                            "model": args.model,
                            "pair_key": row["pair_key"],
                            "group_key": row["group_key"],
                            "comparison": row["comparison"],
                            "condition": condition,
                            "direction": direction,
                            "trial": trial,
                            "target": target,
                            "competitor": competitor,
                            "task_spec": task,
                            "library": library,
                        }
                    )

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_trial, job): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"job={idx}/{len(jobs)} {row['pair_key']} {row['condition']} "
                f"{row['direction']} valid={row['is_valid_choice']} "
                f"correct={row['is_correct']} pair_conf={row['is_pair_confusion']}"
            )

    rows = sorted(rows, key=lambda r: (r["pair_key"], r["direction"], r["trial"], r["condition"]))
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid_rate = sum(row["is_valid_choice"] for row in rows) / len(rows) if rows else 0.0
    print(f"Wrote {out_path}")
    print(f"valid_choice_rate={valid_rate:.4f}")
    if not args.no_quality_gate and valid_rate < args.min_valid_rate:
        raise SystemExit(
            f"Structured descriptor intervention failed quality gate: "
            f"valid_choice_rate={valid_rate:.4f} < {args.min_valid_rate:.4f}"
        )


if __name__ == "__main__":
    main()
