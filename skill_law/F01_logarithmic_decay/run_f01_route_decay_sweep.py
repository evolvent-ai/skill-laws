
import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import sys
from pathlib import Path

load_dotenv()

from skill_law.runtime import load_skills
from skill_law.runtime import LibrarySpec
from skill_law.paths import SKILLS_DIR, finding_data_path, finding_path

def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

def _parse_int_arg(name: str, default: int | None = None) -> int | None:
    args = sys.argv[1:]
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            return int(args[idx + 1])
    return default

def _has_flag(name: str) -> bool:
    return name in sys.argv[1:]

MODEL_NAME = _parse_model()
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key   = os.environ.get("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=_api_key, base_url=_base_url, timeout=120.0)
DATA_DIR = str(finding_path("F01", "data"))
RAW_LOG_FILE = os.path.join(DATA_DIR, f"{MODEL_NAME}_raw_llm_queries.jsonl")
BENCHMARK_FILE = str(finding_data_path("F01", "data", "benchmark_tasks.json"))

log_lock = threading.Lock()

def get_routing_prediction(task_desc, library, history):
    tools_desc = "\n".join([f"- {s.id}: {s.description}" for s in library.skills])

    system_prompt = f"You are an intelligent routing agent. Your job is to select the single most appropriate NEXT tool from the available tools to advance the user's task.\n\nAvailable tools:\n{tools_desc}\n\nRespond ONLY with the exact ID of the chosen tool, and nothing else."

    history_str = " -> ".join(history) if history else "None"
    user_prompt = f"Task: {task_desc}\nPrevious tools executed: {history_str}\n\nWhat is the exact ID of the NEXT tool to use?"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content.strip()
            match = re.search(r'[a-zA-Z0-9_-]+', raw_output)
            chosen = match.group(0) if match else raw_output
            return system_prompt, user_prompt, raw_output, chosen
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return system_prompt, user_prompt, "ERROR", "ERROR"

def load_existing_logs():
    completed = set()
    if os.path.exists(RAW_LOG_FILE):
        with open(RAW_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('chosen_skill') == 'ERROR':
                        continue
                    uid = f"{data['N']}_{data['K']}_{data['trial_idx']}_{data['step_idx']}"
                    completed.add(uid)
                except:
                    pass
    return completed

def append_log(log_entry):
    with log_lock:
        with open(RAW_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def evaluate_single_trial(n, k, trial_idx, task_info, base_library_skills, all_skills, completed_logs):
    target_ids = [s["id"] for s in task_info["required_skills"]]
    task_desc = task_info["task_desc"]

    trial_skills_dict = {s.id: s for s in all_skills if s.id in target_ids}
    needed = n - len(trial_skills_dict)
    if needed > 0:
        for s in base_library_skills:
            if s.id not in trial_skills_dict:
                trial_skills_dict[s.id] = s
                needed -= 1
                if needed == 0:
                    break

    final_library_skills = list(trial_skills_dict.values())
    np.random.default_rng(trial_idx).shuffle(final_library_skills)
    library = LibrarySpec(id=f"lib_{n}", skills=final_library_skills)

    history = []
    chain_failed = False

    for step_idx in range(k):
        uid = f"{n}_{k}_{trial_idx}_{step_idx}"
        expected_target = target_ids[step_idx]

        if uid in completed_logs:
            is_correct = False
            chosen = ""
            with log_lock:
                with open(RAW_LOG_FILE, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if f"{data['N']}_{data['K']}_{data['trial_idx']}_{data['step_idx']}" == uid:
                            is_correct = data['is_correct']
                            chosen = data['chosen_skill']
                            break
        else:
            sys_prompt, user_prompt, raw_out, chosen = get_routing_prediction(task_desc, library, history)
            is_correct = (chosen == expected_target)

            log_entry = {
                "timestamp": time.time(),
                "model": MODEL_NAME,
                "N": n,
                "K": k,
                "trial_idx": trial_idx,
                "step_idx": step_idx,
                "task_desc": task_desc,
                "history": history.copy(),
                "expected_skill": expected_target,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "raw_model_output": raw_out,
                "chosen_skill": chosen,
                "is_correct": is_correct
            }
            append_log(log_entry)

        if not is_correct:
            chain_failed = True
            break

        history.append(chosen)

    return not chain_failed

def run_combination_law_experiment():
    print(f"Running Authentic Combination Law Verification with {MODEL_NAME}")
    demo_mode = _has_flag("--demo")
    limit_cases = _parse_int_arg("--limit-cases")
    all_skills = load_skills()
    rng = np.random.default_rng(42)

    if not os.path.exists(BENCHMARK_FILE):
        print(f"Error: Benchmark file not found at {BENCHMARK_FILE}")
        print("Please run `generate_benchmark_tasks.py` first.")
        return

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        benchmark_tasks = json.load(f)
    if limit_cases is not None:
        benchmark_tasks = benchmark_tasks[:limit_cases]

    tasks_by_k = {}
    unverified_count = 0
    for t in benchmark_tasks:
        k = t["required_steps"]
        if not t.get("human_verified", False):
            unverified_count += 1
        if k not in tasks_by_k:
            tasks_by_k[k] = []
        tasks_by_k[k].append(t)

    if unverified_count > 0:
        print(f"Warning: {unverified_count} tasks in benchmark are marked as NOT human_verified.")
        print("We will still run them, but please ensure data quality!")

    if demo_mode:
        max_k = max(tasks_by_k) if tasks_by_k else 1
        N_values = [max(10, max_k)]
    else:
        N_values = [10, 20, 50, 100, 200, 500]
    K_values = [1, 2, 3, 5, 10]
    completed_logs = load_existing_logs()

    results_summary = {"N": N_values, "K_results": {k: [] for k in K_values}}

    for n in N_values:
        print(f"\n--- Testing N={n} ---")
        base_library_skills = list(rng.choice(all_skills, n, replace=False))

        for k in K_values:
            tasks = tasks_by_k.get(k, [])
            if not tasks:
                continue

            trials = len(tasks)
            success_count = 0

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                for trial_idx, task_info in enumerate(tasks):
                    futures.append(
                        executor.submit(
                            evaluate_single_trial,
                            n, k, trial_idx, task_info, base_library_skills, all_skills, completed_logs
                        )
                    )

                for future in tqdm(as_completed(futures), total=trials, desc=f"N={n}, K={k}"):
                    if future.result():
                        success_count += 1

            success_rate = success_count / trials
            results_summary["K_results"][k].append(success_rate)
            print(f"Results N={n}, K={k} -> Accuracy: {success_count}/{trials} ({success_rate:.2%})")

    summary_file = os.path.join(DATA_DIR, f"{MODEL_NAME}_combination_law_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll tests complete. Summary saved to {summary_file}")
    print(f"Raw query logs saved to {RAW_LOG_FILE}")

if __name__ == "__main__":
    run_combination_law_experiment()
