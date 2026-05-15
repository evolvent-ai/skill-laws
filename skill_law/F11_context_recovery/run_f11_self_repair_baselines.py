

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import openai
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path
from skill_law.runtime import load_env


PAIRS_FILE = finding_data_path("F11", "data", "F11_pairs_v4_rubrics.json")
DATA_DIR = finding_path("F11", "data")
SKILLS_DIR = SKILLS_DIR
JUDGE_MODEL = "gpt-5.4-mini"
CONDITIONS = ["no_upstream_task", "wrong_state_task", "wrong_state_no_task", "perfect_state_task"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit-per-dependency", type=int, default=2)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=120.0,
    )


def stable_seed(*parts: object) -> int:
    payload = "::".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def load_skill_body(skill_id: str) -> str:
    path = SKILLS_DIR / skill_id / "SKILL.md"
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore")
    parts = content.split("---", 2)
    body = parts[2].strip() if len(parts) >= 3 else content
    return body[:1400]


def call_model(api: openai.OpenAI, model: str, prompt: str, temperature: float, seed: int | None, max_retries: int) -> str:
    last_error = ""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if seed is not None:
                kwargs["seed"] = seed
            resp = api.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return f"[EXECUTION FAILED: {last_error}]"


def execute_a(api: openai.OpenAI, pair: dict, model: str, quality: str, seed: int, max_retries: int) -> str:
    if quality == "perfect":
        requirement = "Produce complete, correct, working output that fully solves Step 1."
    else:
        requirement = (
            "Completely miss the point. Produce plausible-looking but fundamentally wrong "
            "output for Step 1: wrong fields, wrong format, or wrong problem."
        )
    prompt = f"""You are executing Step 1 of a software agent task.

SKILL: {pair['a_id']}
DESCRIPTION: {pair.get('a_desc', '')}
SKILL INSTRUCTIONS:
{load_skill_body(pair['a_id'])}

TASK: {pair['task_desc']}

OUTPUT REQUIREMENT: {requirement}

Return only the concrete Step 1 output."""
    return call_model(api, model, prompt, temperature=0.8, seed=seed, max_retries=max_retries)


def execute_b(api: openai.OpenAI, pair: dict, model: str, condition: str, upstream: str, seed: int, max_retries: int) -> str:
    if condition == "wrong_state_no_task":
        task_block = "The original full task is not available. Continue only from the upstream output."
    else:
        task_block = f"TASK: {pair['task_desc']}"
    if condition == "no_upstream_task":
        upstream_block = "No upstream Step 1 output is available."
    else:
        upstream_block = f"UPSTREAM STEP 1 OUTPUT:\n{upstream[:1600]}"
    prompt = f"""You are executing Step 2 of a software agent task.

SKILL: {pair['b_id']}
DESCRIPTION: {pair.get('b_desc', '')}
SKILL INSTRUCTIONS:
{load_skill_body(pair['b_id'])}

{task_block}

{upstream_block}

Execute Step 2. Return only the concrete result."""
    return call_model(api, model, prompt, temperature=0.7, seed=seed + 1000, max_retries=max_retries)


def judge(api: openai.OpenAI, pair: dict, a_output: str, b_output: str, max_retries: int) -> tuple[float, str, list]:
    rubric = pair.get("rubric", [])
    criteria_text = "\n".join(
        [
            f"{i+1}. [{r['weight']:.0%} weight] {r['criterion']}\n"
            f"   PASS if: {r['pass_condition']}\n"
            f"   FAIL if: {r['fail_condition']}"
            for i, r in enumerate(rubric)
        ]
    )
    prompt = f"""You are an expert evaluator for a 2-step AI agent task.

TASK: {pair['task_desc']}

STEP 1 OUTPUT:
{a_output[:1200]}

STEP 2 OUTPUT ({pair['b_id']}):
{b_output[:1200]}

RUBRIC (score each criterion independently: 0.0 = fail, 0.5 = partial, 1.0 = pass):
{criteria_text}

Also assess whether Step 2 meaningfully used Step 1 output:
- "yes": explicitly incorporates specific data from Step 1
- "partial": uses some Step 1 context but ignores key parts
- "ignored": solves independently or does not use Step 1

Respond ONLY with valid JSON:
{{
  "criterion_scores": [
    {{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line reason>"}}
  ],
  "used_upstream": "yes|partial|ignored"
}}"""
    for attempt in range(max_retries):
        try:
            resp = api.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0) if match else raw)
            scores = parsed.get("criterion_scores", [])
            total = 0.0
            for i, criterion in enumerate(rubric):
                if i < len(scores):
                    total += float(criterion["weight"]) * float(scores[i].get("score", 0.0))
            return total, parsed.get("used_upstream", "unknown"), scores
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return 0.0, "unknown", []


def select_pairs(limit_per_dependency: int) -> list[dict]:
    pairs = json.loads(PAIRS_FILE.read_text())
    selected = []
    counts = Counter()
    for pair in pairs:
        dep = pair.get("dependency", "unknown")
        if dep not in {"tight", "loose", "independent"}:
            continue
        if counts[dep] >= limit_per_dependency:
            continue
        selected.append(pair)
        counts[dep] += 1
    return selected


def run_job(job: tuple[dict, int, str], args: argparse.Namespace) -> dict:
    pair, trial, condition = job
    api = client()
    seed = stable_seed(pair["a_id"], pair["b_id"], condition, trial)
    wrong_a = execute_a(api, pair, args.model, "wrong", seed, args.max_retries)
    perfect_a = execute_a(api, pair, args.model, "perfect", seed + 17, args.max_retries) if condition == "perfect_state_task" else ""
    upstream = perfect_a if condition == "perfect_state_task" else wrong_a
    a_for_judge = "" if condition == "no_upstream_task" else upstream
    b_output = execute_b(api, pair, args.model, condition, upstream, seed, args.max_retries)
    score, used, criterion_scores = judge(api, pair, a_for_judge, b_output, args.max_retries)
    return {
        "model": args.model,
        "a_id": pair["a_id"],
        "b_id": pair["b_id"],
        "pair_id": f"{pair['a_id']}__{pair['b_id']}",
        "dependency": pair.get("dependency", "unknown"),
        "condition": condition,
        "trial": trial,
        "score": score,
        "used_upstream": used,
        "criterion_scores": criterion_scores,
        "a_output": a_for_judge[:1200],
        "b_output": b_output[:1200],
        "api_error": "[EXECUTION FAILED:" in wrong_a or "[EXECUTION FAILED:" in b_output,
    }


def main() -> None:
    load_env()
    args = parse_args()
    out_path = DATA_DIR / (
        f"F11_self_repair_baselines_{args.model}_D{args.limit_per_dependency}_T{args.trials}.jsonl"
    )
    pairs = select_pairs(args.limit_per_dependency)
    jobs = [(pair, trial, condition) for pair in pairs for trial in range(args.trials) for condition in args.conditions]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, args): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"job={idx}/{len(jobs)} dep={row['dependency']} condition={row['condition']} score={row['score']:.3f}")
    rows = sorted(rows, key=lambda r: (r["dependency"], r["pair_id"], r["trial"], r["condition"]))
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
