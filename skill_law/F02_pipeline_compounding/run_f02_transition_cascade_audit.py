

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import numpy as np
import openai


from skill_law.runtime import load_skills
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--library-size", type=int, default=40)
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--limit-per-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out-dir", default=str(finding_path("F02", "data", "transition_audit")))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def get_cluster(skill_id: str) -> str:
    match = re.search(
        r"(?:auto-|pro-|fast-|advanced-|batch-|bulk-|quick-|secure-|smart-|deep-)?"
        r"([a-zA-Z-]+?)(?:-\d+)?$",
        skill_id,
    )
    return match.group(1) if match else "unknown"


def build_library(target_id: str, all_skills, skill_by_id, skills_by_cluster, n: int, rng) -> list:
    target = skill_by_id.get(target_id)
    if target is None:
        return []
    cluster = get_cluster(target_id)
    same = [s for s in skills_by_cluster.get(cluster, []) if s.id != target_id]
    others = [s for s in all_skills if s.id != target_id and get_cluster(s.id) != cluster]
    n_same = min(n - 1, len(same))
    library = [target]
    if n_same:
        library.extend(list(rng.choice(same, n_same, replace=False)))
    if len(library) < n:
        library.extend(list(rng.choice(others, n - len(library), replace=False)))
    rng.shuffle(library)
    return library


def route(
    client,
    model: str,
    task_desc: str,
    step_idx: int,
    previous_summary: str,
    library: list,
    max_retries: int,
) -> tuple[str, str]:
    tool_ids = [s.id for s in library]
    tools_desc = "\n".join([f"- {s.id}: {s.description[:700]}" for s in library])
    system_prompt = (
        "You are a tool routing agent. Select the single best NEXT tool for the requested "
        "workflow step.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"Valid IDs: {', '.join(tool_ids)}\n"
        "Respond with ONLY the exact tool ID, nothing else."
    )
    user_prompt = (
        f"Full task: {task_desc}\n\n"
        f"Current step number: {step_idx + 1}\n"
        f"Previous execution summary: {previous_summary}\n\n"
        "What is the exact ID of the NEXT tool to use?"
    )
    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            match = re.search(r"[a-zA-Z0-9_.:-]+", raw)
            chosen = match.group(0) if match else raw
            return chosen, raw
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return "ERROR", last_error


def summarize(raw_rows: list[dict]) -> dict:
    transitions = []
    by_chain = defaultdict(list)
    for row in raw_rows:
        by_chain[row["chain_id"]].append(row)
    for chain_id, rows in by_chain.items():
        rows = sorted(rows, key=lambda r: r["step_idx"])
        for prev, curr in zip(rows, rows[1:]):
            transitions.append(
                {
                    "chain_id": chain_id,
                    "K": curr["K"],
                    "prev_correct": prev["is_correct"],
                    "curr_correct": curr["is_correct"],
                }
            )

    def rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return float(np.mean([t["curr_correct"] for t in subset]))

    correct_prev = [t for t in transitions if t["prev_correct"]]
    wrong_prev = [t for t in transitions if not t["prev_correct"]]
    p_after_correct = rate(correct_prev)
    p_after_wrong = rate(wrong_prev)

    by_k = {}
    for k in sorted({t["K"] for t in transitions}):
        ts = [t for t in transitions if t["K"] == k]
        pc = rate([t for t in ts if t["prev_correct"]])
        pw = rate([t for t in ts if not t["prev_correct"]])
        by_k[str(k)] = {
            "n_transitions": len(ts),
            "p_correct_after_correct": pc,
            "p_correct_after_wrong": pw,
            "rho_c": None if pc is None or pw is None else pc - pw,
        }

    n_error = sum(1 for row in raw_rows if row.get("chosen") == "ERROR")
    error_rate = n_error / len(raw_rows) if raw_rows else 1.0
    return {
        "n_chains": len(by_chain),
        "n_rows": len(raw_rows),
        "n_error_rows": int(n_error),
        "error_rate": float(error_rate),
        "valid_for_analysis": bool(error_rate < 0.05 and len(raw_rows) > 0),
        "validity_note": (
            "valid" if error_rate < 0.05 and len(raw_rows) > 0
            else "invalid_or_diagnostic_only: too many API ERROR rows"
        ),
        "n_transitions": len(transitions),
        "p_correct_after_correct": p_after_correct,
        "p_correct_after_wrong": p_after_wrong,
        "rho_c": None if p_after_correct is None or p_after_wrong is None else p_after_correct - p_after_wrong,
        "by_k": by_k,
    }


def run_chain(
    chain_idx: int,
    task: dict,
    args: argparse.Namespace,
    all_skills: list,
    skill_by_id: dict,
    skills_by_cluster: dict,
) -> list[dict]:
    client = openai.OpenAI(timeout=120.0)
    chain_id = f"{task['idx']}__K{task['required_steps']}__{chain_idx}"
    previous_summary = "No previous steps have been executed."
    rows = []
    for step_idx, skill in enumerate(task["required_skills"]):
        target_id = skill["id"]
        step_rng = np.random.default_rng(abs(hash((chain_id, step_idx, args.seed))) % (2**32))
        library = build_library(
            target_id,
            all_skills=all_skills,
            skill_by_id=skill_by_id,
            skills_by_cluster=skills_by_cluster,
            n=args.library_size,
            rng=step_rng,
        )
        if not library:
            continue
        chosen, raw = route(
            client=client,
            model=args.model,
            task_desc=task["task_desc"],
            step_idx=step_idx,
            previous_summary=previous_summary,
            library=library,
            max_retries=args.max_retries,
        )
        row = {
            "model": args.model,
            "chain_id": chain_id,
            "task_idx": task["idx"],
            "K": int(task["required_steps"]),
            "step_idx": step_idx,
            "target": target_id,
            "chosen": chosen,
            "raw": raw,
            "is_correct": chosen == target_id,
            "library_size": args.library_size,
        }
        rows.append(row)
        previous_summary = (
            f"Step {step_idx + 1} selected tool {chosen}; "
            f"benchmark outcome was {'correct' if row['is_correct'] else 'wrong'}."
        )
    return rows


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.model}_N{args.library_size}_L{args.limit_per_k}"
    raw_path = out_dir / f"transition_cascade_raw_{run_tag}.jsonl"
    summary_path = out_dir / f"transition_cascade_summary_{run_tag}.json"

    all_skills = load_skills()
    skill_by_id = {s.id: s for s in all_skills}
    skills_by_cluster = defaultdict(list)
    for skill in all_skills:
        skills_by_cluster[get_cluster(skill.id)].append(skill)

    tasks = json.loads(finding_data_path("F01", "data", "benchmark_tasks.json").read_text())

    selected = []
    for k in args.k_values:
        k_tasks = [t for t in tasks if int(t["required_steps"]) == k]
        selected.extend(k_tasks[: args.limit_per_k])

    raw_rows = []
    if args.workers <= 1:
        for chain_idx, task in enumerate(selected):
            rows = run_chain(chain_idx, task, args, all_skills, skill_by_id, skills_by_cluster)
            raw_rows.extend(rows)
            print(f"chain={chain_idx + 1}/{len(selected)} rows={len(rows)} done")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_chain, chain_idx, task, args, all_skills, skill_by_id, skills_by_cluster): (
                    chain_idx,
                    task,
                )
                for chain_idx, task in enumerate(selected)
            }
            for future in as_completed(futures):
                chain_idx, task = futures[future]
                rows = future.result()
                raw_rows.extend(rows)
                correct = sum(1 for row in rows if row["is_correct"])
                print(
                    f"chain={chain_idx + 1}/{len(selected)} K={task['required_steps']} "
                    f"rows={len(rows)} correct={correct} done"
                )

    raw_rows = sorted(raw_rows, key=lambda r: (r["chain_id"], r["step_idx"]))
    with raw_path.open("w") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(raw_rows)
    summary.update(
        {
            "model": args.model,
            "library_size": args.library_size,
            "limit_per_k": args.limit_per_k,
            "k_values": args.k_values,
            "workers": args.workers,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
