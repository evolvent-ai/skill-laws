

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import dotenv
import numpy as np
import openai


from skill_law.runtime import load_skills
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path


LEVELS = ["L1_name", "L2_oneline", "L3_full", "L4_examples", "L5_no_boundary_matched", "L5_boundary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--library-sizes", nargs="+", type=int, default=[30, 80, 150])
    parser.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--limit", type=int, default=40, help="Number of target tasks.")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out-dir", default=str(finding_path("F04", "data", "controlled_description_quality")))
    return parser.parse_args()


def get_cluster(skill_id: str) -> str:
    match = re.search(
        r"(?:auto-|pro-|fast-|advanced-|batch-|bulk-|quick-|secure-|smart-|deep-)?"
        r"([a-zA-Z-]+?)(?:-\d+)?$",
        skill_id,
    )
    return match.group(1) if match else "unknown"


def normalize_id(skill_id: str) -> str:
    return skill_id.strip().strip('"')


def stable_seed(*parts: object) -> int:
    payload = "::".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def build_library_ids(target_id: str, all_skills: list, skills_by_id: dict, skills_by_cluster: dict, n: int, rng) -> list[str]:
    target = skills_by_id.get(target_id)
    if target is None:
        return []
    cluster = get_cluster(target_id)
    same = [s.id for s in skills_by_cluster.get(cluster, []) if s.id != target_id]
    others = [s.id for s in all_skills if s.id != target_id and get_cluster(s.id) != cluster]
    library_ids = [target_id]
    n_same = min(n - 1, len(same))
    if n_same:
        library_ids.extend(list(rng.choice(same, n_same, replace=False)))
    if len(library_ids) < n:
        library_ids.extend(list(rng.choice(others, n - len(library_ids), replace=False)))
    rng.shuffle(library_ids)
    return library_ids


def description_for_level(target, level: str, competitor_ids: list[str]) -> str:
    base = target.description.strip()
    cluster = get_cluster(target.id).replace("-", " ")
    competitors = ", ".join(competitor_ids[:4]) if competitor_ids else "nearby same-family tools"
    if level == "L1_name":
        return target.id
    if level == "L2_oneline":
        first_sentence = base.split(".")[0] + "." if "." in base else base
        return f"{target.id}: {first_sentence}"
    if level == "L3_full":
        return base
    if level == "L4_examples":
        return (
            f"{base} Use this tool for {cluster} requests when the requested action matches "
            f"the tool name and its listed triggers. Example: choose {target.id} when the "
            f"task asks for its exact operation, inputs, or output behavior."
        )
    if level == "L5_no_boundary_matched":
        return (
            f"{base} Use this tool for {cluster} requests when the requested action matches "
            f"the tool name and its listed triggers. Example: choose {target.id} when the "
            f"task asks for its exact operation, inputs, or output behavior. Prefer exact "
            f"semantic matches, preserve the requested object type, and compare all tool IDs "
            f"before selecting."
        )
    if level == "L5_boundary":
        return (
            f"{base} Use this tool for {cluster} requests when the requested action matches "
            f"the tool name and its listed triggers. Example: choose {target.id} when the "
            f"task asks for its exact operation, inputs, or output behavior. Boundary rule: "
            f"do not use {target.id} for tasks better matched by {competitors}; route those "
            f"excluded cases to the named competing tool instead."
        )
    raise ValueError(f"Unknown level: {level}")


def build_library(skill_ids: list[str], target_id: str, level: str, skills_by_id: dict) -> list:
    same_cluster_competitors = [
        sid for sid in skill_ids if sid != target_id and get_cluster(sid) == get_cluster(target_id)
    ]
    library = []
    for sid in skill_ids:
        skill = skills_by_id[sid]
        if sid == target_id:
            desc = description_for_level(skill, level, same_cluster_competitors)
            library.append(replace(skill, description=desc))
        else:
            library.append(skill)
    return library


def route(client, model: str, task_desc: str, library: list, max_retries: int) -> tuple[str, str]:
    tool_ids = [s.id for s in library]
    tools_desc = "\n".join([f"- {s.id}: {s.description[:900]}" for s in library])
    system_prompt = (
        "You are a tool routing agent. Select the single best tool for the task.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"Valid IDs: {', '.join(tool_ids)}\n"
        "Respond with ONLY the exact tool ID, nothing else."
    )
    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task_desc}\n\nTool ID:"},
                ],
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            match = re.search(r"[a-zA-Z0-9_.:-]+", raw)
            return (match.group(0) if match else raw), raw
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return "ERROR", last_error


def run_trial(trial: dict, args: argparse.Namespace, skills_by_id: dict) -> dict:
    client = openai.OpenAI(timeout=120.0)
    library = build_library(trial["library_ids"], trial["target"], trial["level"], skills_by_id)
    chosen, raw = route(client, args.model, trial["task_desc"], library, args.max_retries)
    target_desc = next(s.description for s in library if s.id == trial["target"])
    return {
        **trial,
        "model": args.model,
        "chosen": chosen,
        "raw": raw,
        "is_correct": chosen == trial["target"],
        "target_desc_chars": len(target_desc),
        "valid_output": chosen in set(trial["library_ids"]),
    }


def summarize(rows: list[dict]) -> dict:
    n_error = sum(1 for row in rows if row.get("chosen") == "ERROR")
    valid_rows = [row for row in rows if row.get("valid_output")]
    by_level_n = {}
    for key in sorted({(row["level"], row["library_size"]) for row in rows}):
        subset = [row for row in rows if (row["level"], row["library_size"]) == key]
        usable = [row for row in subset if row.get("valid_output")]
        by_level_n[f"{key[0]}_N{key[1]}"] = {
            "n": len(subset),
            "usable_n": len(usable),
            "accuracy": float(np.mean([row["is_correct"] for row in usable])) if usable else None,
            "invalid_rate": 1 - len(usable) / len(subset) if subset else None,
            "mean_target_desc_chars": float(np.mean([row["target_desc_chars"] for row in subset])) if subset else None,
        }
    return {
        "n_rows": len(rows),
        "n_error_rows": n_error,
        "error_rate": n_error / len(rows) if rows else 1.0,
        "valid_output_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "valid_for_analysis": bool(rows and n_error / len(rows) < 0.05 and len(valid_rows) / len(rows) >= 0.9),
        "validity_note": (
            "valid"
            if rows and n_error / len(rows) < 0.05 and len(valid_rows) / len(rows) >= 0.9
            else "invalid_or_diagnostic_only: too many API errors or unparsable outputs"
        ),
        "by_level_n": by_level_n,
    }


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.model}_N{'-'.join(map(str, args.library_sizes))}_L{args.limit}"
    raw_path = out_dir / f"controlled_description_quality_raw_{run_tag}.jsonl"
    summary_path = out_dir / f"controlled_description_quality_summary_{run_tag}.json"

    all_skills = [s for s in load_skills() if s.description and s.description != "|" and '"' not in s.id]
    skills_by_id = {s.id: s for s in all_skills}
    skills_by_cluster = defaultdict(list)
    for skill in all_skills:
        skills_by_cluster[get_cluster(skill.id)].append(skill)

    tasks = json.loads(finding_data_path("F01", "data", "benchmark_tasks.json").read_text())
    single_step = [
        t for t in tasks
        if int(t["required_steps"]) == 1 and t["required_skills"][0]["id"] in skills_by_id
    ]
    rng = np.random.default_rng(args.seed)
    sampled = list(rng.choice(single_step, size=min(args.limit, len(single_step)), replace=False))

    trials = []
    for task_idx, task in enumerate(sampled):
        target = task["required_skills"][0]["id"]
        for n in args.library_sizes:
            lib_rng = np.random.default_rng(stable_seed(task["idx"], target, n, args.seed))
            library_ids = build_library_ids(target, all_skills, skills_by_id, skills_by_cluster, n, lib_rng)
            if not library_ids:
                continue
            for level in args.levels:
                trials.append(
                    {
                        "trial_id": f"{task_idx}_{target}_N{n}_{level}",
                        "task_idx": task["idx"],
                        "task_desc": task["task_desc"],
                        "target": target,
                        "cluster": get_cluster(target),
                        "library_size": n,
                        "level": level,
                        "library_ids": library_ids,
                    }
                )

    rows = []
    if args.workers <= 1:
        for idx, trial in enumerate(trials):
            row = run_trial(trial, args, skills_by_id)
            rows.append(row)
            print(f"trial={idx + 1}/{len(trials)} level={row['level']} N={row['library_size']} correct={row['is_correct']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_trial, trial, args, skills_by_id): trial for trial in trials}
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"trial={idx}/{len(trials)} level={row['level']} N={row['library_size']} correct={row['is_correct']}")

    rows = sorted(rows, key=lambda r: (r["task_idx"], r["library_size"], r["level"]))
    with raw_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "library_sizes": args.library_sizes,
            "levels": args.levels,
            "limit": args.limit,
            "workers": args.workers,
            "run_tag": run_tag,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
