import argparse
import json, os, re, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
from skill_law.runtime import load_skills
from skill_law.runtime import LibrarySpec
from skill_law.paths import SKILLS_DIR, finding_data_path, finding_path

client = openai.OpenAI(timeout=600.0)
DATA_DIR = str(finding_path("F03", "data", "archive"))
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "step_position_queries.jsonl")
log_lock = threading.Lock()

def get_cluster(skill_id):
    m = re.search(r'(?:auto-|pro-|fast-|advanced-|batch-|bulk-|quick-|secure-|smart-|deep-)?([a-zA-Z-]+?)(?:-\d+)?$', skill_id)
    return m.group(1) if m else "unknown"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-5.4-mini"))
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--limit-per-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=20)
    return parser.parse_args()


def route(task_desc, library_skills, model_name):
    tool_ids = [s.id for s in library_skills]
    tools_desc = "\n".join([f"- {s.id}: {s.description}" for s in library_skills])
    system_prompt = (
        f"You are a tool routing agent. Select the single best tool for the task.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"Valid IDs: {', '.join(tool_ids)}\n"
        f"Respond with ONLY the exact tool ID, nothing else."
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task_desc}\n\nTool ID:"}
                ],
                temperature=0.0
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'[a-zA-Z0-9_.-]+', raw)
            return m.group(0) if m else raw
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "ERROR"

def run():
    args = parse_args()
    all_skills = load_skills()
    skill_dict = {s.id: s for s in all_skills}
    skills_by_cluster = defaultdict(list)
    for s in all_skills:
        skills_by_cluster[get_cluster(s.id)].append(s)

    bench_path = str(finding_data_path("F01", "data", "benchmark_tasks.json"))
    with open(bench_path) as f:
        tasks = json.load(f)

    K_values = args.k_values
    N = 40
    n_per_K = args.limit_per_k
    rng = np.random.default_rng(42)

    step_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    K_step_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    open(LOG_FILE, 'w').close()
    results = {}

    for K in K_values:
        k_tasks = [t for t in tasks if t['required_steps'] == K][:n_per_K]
        print(f"\nK={K}: {len(k_tasks)} tasks")

        def trial(task_info):
            skill_ids = [s['id'] for s in task_info['required_skills']]
            step_results = []
            for step_idx, target_id in enumerate(skill_ids):
                target_skill = skill_dict.get(target_id)
                if not target_skill:
                    continue
                cluster = get_cluster(target_id)
                same = [s for s in skills_by_cluster.get(cluster, []) if s.id != target_id]
                others = [s for s in all_skills if s.id != target_id and get_cluster(s.id) != cluster]
                n_same = min(N-1, len(same))
                library = [target_skill] + list(rng.choice(same, n_same, replace=False))
                if len(library) < N:
                    library += list(rng.choice(others, N - len(library), replace=False))
                rng.shuffle(library)

                chosen = route(task_info['task_desc'], library, args.model)
                is_correct = (chosen == target_id)
                step_results.append((step_idx, target_id, chosen, is_correct))
                with log_lock:
                    with open(LOG_FILE, 'a') as f:
                        f.write(json.dumps({
                            "K": K, "step_idx": step_idx,
                            "target": target_id, "chosen": chosen,
                            "is_correct": is_correct,
                            "task_desc": task_info['task_desc'][:100]
                        }) + "\n")
            return step_results

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(trial, t) for t in k_tasks]
            for fut in as_completed(futures):
                for step_idx, target_id, chosen, is_correct in fut.result():
                    step_stats[step_idx]['total'] += 1
                    K_step_stats[K][step_idx]['total'] += 1
                    if is_correct:
                        step_stats[step_idx]['correct'] += 1
                        K_step_stats[K][step_idx]['correct'] += 1

        k_results = {}
        for step_idx in range(K):
            s = K_step_stats[K][step_idx]
            if s['total'] > 0:
                acc = s['correct'] / s['total']
                k_results[step_idx] = acc
                print(f"  Step {step_idx+1}/{K}: acc={acc:.1%} ({s['total']} trials)")
        results[K] = k_results

    out_path = os.path.join(DATA_DIR, "step_position_decay.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: {out_path}")

if __name__ == "__main__":
    run()
