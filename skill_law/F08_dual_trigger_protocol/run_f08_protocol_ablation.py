
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import numpy as np

dotenv.load_dotenv()

from skill_law.runtime import LLMRouter
from skill_law.runtime import StrictLLMRouter
from skill_law.runtime import NameOnlyRouter
from skill_law.runtime import DescriptionOnlyRouter
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.runtime import TaskSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path, finding_path


def _parse_arg(flag: str, default: str) -> str:
    args = sys.argv[1:]
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


MODEL = _parse_arg("--model", "gpt-5.4-mini")
PROTOCOL = _parse_arg("--protocol", "strict")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
TRIALS = int(os.environ.get("ABLATION_TRIALS", "4"))
LIBRARY_SIZES = [int(x) for x in os.environ.get("LIBRARY_SIZES", "150,200,250").split(",") if x]
HARD_NEG_MAP = {150: 30, 200: 40, 250: 55}
SEED = 20260415
SKILLS_DIR = str(SKILLS_DIR)
EMB_CACHE = str(data_path("processed", "bge_skill_embeddings.npz"))

OUT_DIR = str(finding_path("F08", "data", "final"))
TASKS_FILE = f"{OUT_DIR}/hijack_validation_tasks_{MODEL}.json"
RAW_FILE = f"{OUT_DIR}/protocol_ablation_{PROTOCOL}_{MODEL}.jsonl"
SUMMARY_FILE = f"{OUT_DIR}/protocol_ablation_{PROTOCOL}_{MODEL}.json"


def load_clean_skills():
    all_names = sorted(
        n for n in os.listdir(SKILLS_DIR)
        if os.path.exists(os.path.join(SKILLS_DIR, n, "SKILL.md"))
    )
    clean_names, clean_descs = [], []
    for name in all_names:
        with open(os.path.join(SKILLS_DIR, name, "SKILL.md"), encoding="utf-8") as f:
            raw = f.read()
        if "synthetic noise skill" in raw.lower():
            continue
        clean_names.append(name)
        clean_descs.append(raw)
    skill_specs_clean = [
        SkillSpec(id=n, name=n, description=d[:800], input_schema={}, output_schema={})
        for n, d in zip(clean_names, clean_descs)
    ]
    skills_map = {s.id: s for s in skill_specs_clean}
    skill_idx = {name: i for i, name in enumerate(clean_names)}

    all_embs = np.load(EMB_CACHE)["embs"]
    full_idx = {name: i for i, name in enumerate(all_names)}
    clean_embs = np.array([all_embs[full_idx[n]] for n in clean_names])
    sim_matrix = clean_embs @ clean_embs.T
    return clean_names, skills_map, skill_idx, sim_matrix


CLEAN_NAMES, SKILLS_MAP, SKILL_IDX, SIM_MATRIX = load_clean_skills()


def make_router():
    if PROTOCOL == "strict":
        return StrictLLMRouter(model_name=MODEL)
    if PROTOCOL == "text":
        return LLMRouter(model_name=MODEL)
    if PROTOCOL == "name_only":
        return NameOnlyRouter(model_name=MODEL)
    if PROTOCOL == "desc_only":
        return DescriptionOnlyRouter(model_name=MODEL)
    raise ValueError(f"Unsupported protocol: {PROTOCOL}")


def load_tasks():
    rows = json.loads(Path(TASKS_FILE).read_text(encoding="utf-8"))
    return rows


def build_library(target_name: str, n_lib: int, seed: int) -> list:
    trial_rng = np.random.default_rng(seed)
    ti = SKILL_IDX[target_name]
    n_hard = HARD_NEG_MAP[n_lib]
    sims = SIM_MATRIX[ti].copy()
    sims[ti] = -1
    hard_idx = np.argsort(sims)[::-1][:n_hard]
    pool_idx = [i for i in range(len(CLEAN_NAMES)) if i != ti and i not in hard_idx]
    rand_idx = trial_rng.choice(pool_idx, size=(n_lib - 1 - n_hard), replace=False)
    lib = [SKILLS_MAP[target_name]]
    for i in hard_idx:
        lib.append(SKILLS_MAP[CLEAN_NAMES[i]])
    for i in rand_idx:
        lib.append(SKILLS_MAP[CLEAN_NAMES[i]])
    perm = trial_rng.permutation(len(lib))
    return [lib[p] for p in perm]


def build_jobs(task_rows):
    jobs = []
    for entry in task_rows:
        for n_lib in LIBRARY_SIZES:
            for trial in range(TRIALS):
                seed = abs(hash((entry["skill_name"], n_lib, trial, PROTOCOL, SEED))) % (2**32)
                lib = build_library(entry["skill_name"], n_lib=n_lib, seed=seed)
                lib_spec = LibrarySpec(id=f"abl_{entry['skill_name']}_{n_lib}_t{trial}", skills=lib)
                task_spec = TaskSpec(
                    id=f"abl_{entry['skill_name']}_{n_lib}_t{trial}",
                    instruction=entry["user_paraphrase"],
                    required_skills=[{"name": entry["skill_name"]}],
                    gold_trace=[],
                )
                jobs.append({
                    "skill_name": entry["skill_name"],
                    "clarity_bin": entry["clarity_bin"],
                    "library_size": n_lib,
                    "trial": trial,
                    "task_spec": task_spec,
                    "library": lib_spec,
                    "query_text": entry["user_paraphrase"],
                })
    return jobs


def run_one(job):
    router = make_router()
    try:
        chosen_id, route_meta = router.route(job["task_spec"], job["library"])
    except Exception as exc:
        chosen_id = "ERROR"
        route_meta = {"query": "exception", "error": str(exc)}
    target = job["skill_name"]
    is_correct = chosen_id == target
    is_null = chosen_id in (None, "", "None", "ERROR")
    is_hijack = (not is_correct) and (not is_null)
    return {
        "timestamp": time.time(),
        "protocol": PROTOCOL,
        "model": MODEL,
        "skill_name": target,
        "clarity_bin": job["clarity_bin"],
        "library_size": job["library_size"],
        "trial": job["trial"],
        "query_text": job["query_text"],
        "chosen_skill": chosen_id,
        "is_correct": is_correct,
        "is_null": is_null,
        "is_hijack": is_hijack,
        "route_meta": route_meta,
    }


def summarize(rows):
    summary_by_n = []
    for n_lib in LIBRARY_SIZES:
        subset = [r for r in rows if r["library_size"] == n_lib]
        if not subset:
            continue
        summary_by_n.append({
            "library_size": n_lib,
            "n": len(subset),
            "accuracy": round(float(np.mean([r["is_correct"] for r in subset])), 6),
            "hijack_rate": round(float(np.mean([r["is_hijack"] for r in subset])), 6),
            "null_rate": round(float(np.mean([r["is_null"] for r in subset])), 6),
            "route_meta_counts": dict(Counter((r.get("route_meta") or {}).get("query") for r in subset)),
        })
    return {
        "model": MODEL,
        "protocol": PROTOCOL,
        "trials_per_condition": TRIALS,
        "summary_by_n": summary_by_n,
        "overall_route_meta_counts": dict(Counter((r.get("route_meta") or {}).get("query") for r in rows)),
    }


def main():
    task_rows = load_tasks()
    jobs = build_jobs(task_rows)
    results = []

    print(f"Running protocol ablation: model={MODEL}, protocol={PROTOCOL}, jobs={len(jobs)}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(RAW_FILE, "w", encoding="utf-8") as f:
        futures = [executor.submit(run_one, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if idx % 50 == 0 or idx == len(futures):
                print(f"  {idx}/{len(futures)} done")

    summary = summarize(results)
    Path(SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
