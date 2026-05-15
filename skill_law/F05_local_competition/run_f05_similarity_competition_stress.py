import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


from skill_law.runtime import LLMRouter
from skill_law.runtime import TaskSpec
from skill_law.runtime import LibrarySpec, SkillSpec

from dotenv import load_dotenv
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

load_dotenv()

def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

TASKS_FILE    = str(finding_data_path("F01", "data", "benchmark_tasks.json"))
SKILLS_DIR    = str(SKILLS_DIR)
OUTPUT_DIR    = str(finding_path("F05", "data"))
MODEL         = _parse_model()
LLM_QUERIES_FILE = f"{OUTPUT_DIR}/raw_llm_queries_{MODEL}.jsonl"
RESULTS_FILE     = f"{OUTPUT_DIR}/raw_results_{MODEL}.json"
ERROR_FILE       = f"{OUTPUT_DIR}/raw_errors_{MODEL}.jsonl"
TOTAL_N = 30
MAX_WORKERS   = int(os.environ.get("MAX_WORKERS", "25"))
NUM_TEST_TASKS = int(os.environ.get("NUM_TEST_TASKS", "100"))
SCENARIOS     = ["Low", "Mid", "High"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_LEXICAL_EMBEDDINGS = os.environ.get("SKILL_LAW_LEXICAL_EMBEDDINGS", "0") == "1"
try:
    if USE_LEXICAL_EMBEDDINGS:
        raise ImportError("lexical embedding fallback requested")
    from sentence_transformers import SentenceTransformer
    print("Loading BGE model (BAAI/bge-small-en-v1.5)...")
    _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("BGE model loaded.")

    def rank_by_similarity(target_emb, other_embs):
        t = target_emb / np.linalg.norm(target_emb)
        o = other_embs / np.linalg.norm(other_embs, axis=1, keepdims=True)
        sims = o @ t
        return np.argsort(-sims)

except ImportError:
    _embedder = None
    print("Using lexical TF-IDF similarity fallback.")

    def rank_by_similarity(target_emb, other_embs):
        sims = np.array([np.dot(target_emb, e) for e in other_embs])
        return np.argsort(-sims)


def embed_skills(skills):
    if _embedder is not None:
        descs = [s.description for s in skills]
        embs  = _embedder.encode(descs, batch_size=64, show_progress_bar=True)
        return embs
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        descs = [s.description for s in skills]
        vec = TfidfVectorizer(max_features=4096).fit_transform(descs).toarray()
        return vec


def load_all_skills():
    skills, seen = [], set()
    for name in os.listdir(SKILLS_DIR):
        if name in seen:
            continue
        path = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                desc = f.read()
            skills.append(SkillSpec(
                id=name, name=name,
                description=desc[:800],
                input_schema={}, output_schema={}
            ))
            seen.add(name)
    return skills


def llm_query(job):
    router = LLMRouter(model_name=MODEL)
    try:
        chosen_id, _ = router.route(job["task_spec"], job["library"])
    except Exception as e:
        print(f"  [ERROR] task={job['task_idx']} scenario={job['scenario']}: {e}")
        chosen_id = "ERROR"

    return {
        "timestamp":    time.time(),
        "task_idx":     job["task_idx"],
        "scenario":     job["scenario"],
        "target_skill": job["target_skill_id"],
        "task_desc":    job["task_spec"].instruction,
        "system_prompt": job["system_prompt"],
        "chosen_skill": chosen_id,
        "is_correct":   chosen_id == job["target_skill_id"],
        "library_size": TOTAL_N,
    }


def main():
    print("Loading skills and computing embeddings (this may take a moment)...")
    all_skills = load_all_skills()
    skills_map = {s.id: s for s in all_skills}

    EMB_CACHE_FILE = str(data_path("processed", "bge_skill_embeddings.npz"))
    os.makedirs(os.path.dirname(EMB_CACHE_FILE), exist_ok=True)
    if os.path.exists(EMB_CACHE_FILE):
        print("Loading cached embeddings...")
        all_embs = np.load(EMB_CACHE_FILE)["embs"]
    else:
        print("Computing embeddings and saving to cache...")
        all_embs = embed_skills(all_skills)
        np.savez(EMB_CACHE_FILE, embs=all_embs)

    skill_idx  = {s.id: i for i, s in enumerate(all_skills)}
    print(f"  {len(all_skills)} skills embedded.")
    with open(TASKS_FILE) as f:
        tasks = json.load(f)
    test_tasks = []
    for t in tasks:
        req = t.get("required_skills", [])
        if len(req) > 0:
            test_tasks.append(t)
        if len(test_tasks) >= NUM_TEST_TASKS:
            break
    print(f"  {len(test_tasks)} tasks selected from benchmark.")

    print("Pre-computing similarity-ranked libraries...")
    rng = np.random.default_rng(42)
    jobs = []

    for t_idx, task_dict in enumerate(test_tasks):
        target_id = task_dict["required_skills"][0]["name"]
        if target_id not in skill_idx:
            continue
        k = task_dict.get("required_steps", 1)
        if k > 1:
            task_dict["task_desc"] = f"Task: {task_dict['task_desc']}\n\nFor STEP 1 of this task, which single tool from the available tools is required?"


        ti = skill_idx[target_id]
        target_emb  = all_embs[ti]
        other_mask  = np.arange(len(all_skills)) != ti
        other_skills = [s for s in all_skills if s.id != target_id]
        other_embs   = all_embs[other_mask]

        order = rank_by_similarity(target_emb, other_embs)
        n_other = len(other_skills)

        most_similar  = [other_skills[order[j]] for j in range(min(TOTAL_N-1, n_other))]
        least_similar = [other_skills[order[-(j+1)]] for j in range(min(TOTAL_N-1, n_other))]
        half_n        = (TOTAL_N - 1) // 2
        mid_start     = n_other // 2 - half_n
        mid_similar   = [other_skills[order[mid_start + j]] for j in range(min(half_n, n_other))]

        configs = {
            "Low":  [skills_map[target_id]] + least_similar[:TOTAL_N-1],
            "Mid":  [skills_map[target_id]] + least_similar[:TOTAL_N-1-half_n] + mid_similar[:half_n],
            "High": [skills_map[target_id]] + most_similar[:TOTAL_N-1],
        }

        task_spec = TaskSpec(
            id=task_dict.get("idx", f"task_{t_idx}"),
            instruction=task_dict["task_desc"],
            required_skills=task_dict.get("required_skills", []),
            gold_trace=task_dict.get("gold_trace", []),
        )

        router_ref = LLMRouter(model_name=MODEL)
        for scenario in SCENARIOS:
            lib_skills = configs[scenario].copy()
            rng.shuffle(lib_skills)
            lib = LibrarySpec(id=f"lib_{t_idx}_{scenario}", skills=lib_skills)
            sys_prompt, _ = router_ref._build_prompt(task_spec, lib)
            jobs.append({
                "task_idx":       t_idx,
                "scenario":       scenario,
                "task_spec":      task_spec,
                "library":        lib,
                "target_skill_id": target_id,
                "system_prompt":  sys_prompt,
            })

    print(f"  {len(jobs)} LLM jobs prepared ({len(jobs)//len(SCENARIOS)} tasks × {len(SCENARIOS)} scenarios).")

    completed_keys = set()
    if os.path.exists(LLM_QUERIES_FILE):
        with open(LLM_QUERIES_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    completed_keys.add((row["task_idx"], row["scenario"]))
                except Exception:
                    continue
        if completed_keys:
            before = len(jobs)
            jobs = [
                job for job in jobs
                if (job["task_idx"], job["scenario"]) not in completed_keys
            ]
            print(f"Resuming from {len(completed_keys)} completed jobs; {len(jobs)}/{before} remain.")

    if not jobs:
        print("No remaining jobs to run; recomputing aggregate stats from existing raw log.")

    print(f"Submitting {len(jobs)} queries to LLM with {MAX_WORKERS} threads...")
    if jobs:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(LLM_QUERIES_FILE, "a", encoding="utf-8") as f:
            future_to_key = {
                executor.submit(llm_query, job): (job["task_idx"], job["scenario"])
                for job in jobs
            }
            done = 0
            with open(ERROR_FILE, "a", encoding="utf-8") as ef:
                for future in as_completed(future_to_key):
                    try:
                        res = future.result()
                        if res.get("chosen_skill") in (None, "", "ERROR", "None"):
                            ef.write(json.dumps(res, ensure_ascii=False) + "\n")
                            ef.flush()
                        else:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                            f.flush()
                    except Exception as exc:
                        print(f"  [FATAL] job {future_to_key[future]}: {exc}")
                    done += 1
                    if done % 25 == 0 or done == len(jobs):
                        print(f"  {done}/{len(jobs)} completed")

    print("Aggregating results from raw log...")
    scenario_stats = {s: {"correct": 0, "total": 0} for s in SCENARIOS}
    with open(LLM_QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                res = json.loads(line)
            except Exception:
                continue
            s = res["scenario"]
            scenario_stats[s]["total"] += 1
            scenario_stats[s]["correct"] += int(res["is_correct"])

    print("\n--- FINAL RESULTS ---")
    final_res = []
    for s in SCENARIOS:
        total = max(1, scenario_stats[s]["total"])
        acc   = scenario_stats[s]["correct"] / total
        print(f"  {s:4s}: {acc:.2%}  ({scenario_stats[s]['correct']}/{total})")
        final_res.append({
            "scenario": s,
            "accuracy": acc,
            "correct":  scenario_stats[s]["correct"],
            "total":    scenario_stats[s]["total"],
        })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_res, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {LLM_QUERIES_FILE}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
