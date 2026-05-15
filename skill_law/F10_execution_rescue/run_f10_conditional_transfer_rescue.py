import json, os, sys, time, random
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


from skill_law.runtime import StrictLLMRouter as LLMRouter
from skill_law.runtime import TaskSpec
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

TASKS_FILE   = str(finding_data_path("F01", "data", "benchmark_tasks.json"))
SKILLS_DIR   = str(SKILLS_DIR)
EMB_CACHE    = str(data_path("processed", "bge_skill_embeddings.npz"))
OUT_DIR      = str(finding_path("F10", "data"))
MODEL        = _parse_model()
QUERIES_FILE = f"{OUT_DIR}/raw_llm_queries_{MODEL}.jsonl"
RESULTS_FILE = f"{OUT_DIR}/results_{MODEL}.json"
N_LIB        = 30
TRIALS       = int(os.environ.get("TRIALS", "15"))
MAX_WORKERS  = int(os.environ.get("MAX_WORKERS", "25"))
TASK_LIMIT   = int(os.environ.get("TASK_LIMIT", "0"))

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading skills...")
skills_names = sorted(
    n for n in os.listdir(SKILLS_DIR)
    if os.path.exists(os.path.join(SKILLS_DIR, n, "SKILL.md"))
)
skill_specs = []
for name in skills_names:
    with open(os.path.join(SKILLS_DIR, name, "SKILL.md"), encoding="utf-8") as f:
        desc = f.read()
    skill_specs.append(SkillSpec(
        id=name, name=name,
        description=desc[:800],
        input_schema={}, output_schema={}
    ))
skills_map = {s.id: s for s in skill_specs}
print(f"  {len(skill_specs)} skills loaded.")

if os.path.exists(EMB_CACHE):
    print("Loading cached BGE embeddings...")
    all_embs = np.load(EMB_CACHE)["embs"]
else:
    print("Computing BGE embeddings (BAAI/bge-small-en-v1.5)...")
    from sentence_transformers import SentenceTransformer
    model_bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    descs = [s.description for s in skill_specs]
    all_embs = model_bge.encode(descs, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=True)
    os.makedirs("data/processed", exist_ok=True)
    np.savez(EMB_CACHE, embs=all_embs)
    print("  Embeddings saved to cache.")

skill_idx = {name: i for i, name in enumerate(skills_names)}
print(f"  Embeddings shape: {all_embs.shape}")

print("Loading K=2 tasks from benchmark...")
with open(TASKS_FILE) as f:
    all_tasks = json.load(f)

k2_tasks = [t for t in all_tasks if len(t.get("required_skills", [])) == 2]
if TASK_LIMIT > 0:
    k2_tasks = k2_tasks[:TASK_LIMIT]
print(f"  {len(k2_tasks)} K=2 tasks found.")


def build_library(target_ids: list[str], n: int, rng: np.random.Generator) -> list[SkillSpec]:
    n_targets = len(target_ids)
    n_fill = n - n_targets

    target_embs = np.stack([all_embs[skill_idx[tid]] for tid in target_ids
                            if tid in skill_idx])
    all_sims = (all_embs @ target_embs.T).max(axis=1)

    target_set = set(target_ids)
    candidates = [(i, float(all_sims[i])) for i, name in enumerate(skills_names)
                  if name not in target_set]

    candidates.sort(key=lambda x: -x[1])
    top_fill = [skill_specs[i] for i, _ in candidates[:n_fill]]

    lib = [skills_map[tid] for tid in target_ids if tid in skills_map] + top_fill
    rng.shuffle(lib)
    return lib


print("Building LLM jobs...")
rng = np.random.default_rng(42)
router_ref = LLMRouter(model_name=MODEL)

jobs = []

for t_idx, task_dict in enumerate(k2_tasks):
    skill_a_id = task_dict["required_skills"][0]["name"]
    skill_b_id = task_dict["required_skills"][1]["name"]

    if skill_a_id not in skill_idx or skill_b_id not in skill_idx:
        print(f"  Skipping task {t_idx}: skills not found")
        continue

    task_desc = task_dict["task_desc"]

    sentences = [s.strip() for s in task_desc.replace(";", ".").split(".") if s.strip()]
    mid = max(1, len(sentences) // 2)
    task_a_desc = ". ".join(sentences[:mid]) + "."
    task_b_desc = ". ".join(sentences[mid:]) + "." if sentences[mid:] else task_desc

    emb_a = all_embs[skill_idx[skill_a_id]]
    emb_b = all_embs[skill_idx[skill_b_id]]
    sim_ab = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))

    for trial in range(TRIALS):
        trial_rng = np.random.default_rng(42 + t_idx * 1000 + trial)

        lib_a = build_library([skill_a_id], N_LIB, trial_rng)
        task_spec_a = TaskSpec(
            id=f"t{t_idx}_a_{trial}",
            instruction=task_a_desc,
            required_skills=[{"name": skill_a_id}],
            gold_trace=[],
        )
        lib_spec_a = LibrarySpec(id=f"lib_a_{t_idx}_{trial}", skills=lib_a)
        sys_p_a, _ = router_ref._build_prompt(task_spec_a, lib_spec_a)
        jobs.append({
            "job_type": "single_a",
            "task_idx": t_idx,
            "trial": trial,
            "target_skill": skill_a_id,
            "skill_a": skill_a_id,
            "skill_b": skill_b_id,
            "sim_ab": sim_ab,
            "task_spec": task_spec_a,
            "library": lib_spec_a,
            "system_prompt": sys_p_a,
        })

        lib_b = build_library([skill_b_id], N_LIB, trial_rng)
        task_spec_b = TaskSpec(
            id=f"t{t_idx}_b_{trial}",
            instruction=task_b_desc,
            required_skills=[{"name": skill_b_id}],
            gold_trace=[],
        )
        lib_spec_b = LibrarySpec(id=f"lib_b_{t_idx}_{trial}", skills=lib_b)
        sys_p_b, _ = router_ref._build_prompt(task_spec_b, lib_spec_b)
        jobs.append({
            "job_type": "single_b",
            "task_idx": t_idx,
            "trial": trial,
            "target_skill": skill_b_id,
            "skill_a": skill_a_id,
            "skill_b": skill_b_id,
            "sim_ab": sim_ab,
            "task_spec": task_spec_b,
            "library": lib_spec_b,
            "system_prompt": sys_p_b,
        })

        lib_combo = build_library([skill_a_id, skill_b_id], N_LIB, trial_rng)
        lib_spec_combo = LibrarySpec(id=f"lib_combo_{t_idx}_{trial}", skills=lib_combo)

        task_spec_ca = TaskSpec(
            id=f"t{t_idx}_ca_{trial}",
            instruction=f"[Multi-step task] {task_desc}\n\nFor STEP 1 only, which single tool is needed?",
            required_skills=[{"name": skill_a_id}],
            gold_trace=[],
        )
        sys_p_ca, _ = router_ref._build_prompt(task_spec_ca, lib_spec_combo)
        jobs.append({
            "job_type": "combo_a",
            "task_idx": t_idx,
            "trial": trial,
            "target_skill": skill_a_id,
            "skill_a": skill_a_id,
            "skill_b": skill_b_id,
            "sim_ab": sim_ab,
            "task_spec": task_spec_ca,
            "library": lib_spec_combo,
            "system_prompt": sys_p_ca,
        })

        task_spec_cb = TaskSpec(
            id=f"t{t_idx}_cb_{trial}",
            instruction=f"[Multi-step task] {task_desc}\n\nFor STEP 2 only, which single tool is needed?",
            required_skills=[{"name": skill_b_id}],
            gold_trace=[],
        )
        sys_p_cb, _ = router_ref._build_prompt(task_spec_cb, lib_spec_combo)
        jobs.append({
            "job_type": "combo_b",
            "task_idx": t_idx,
            "trial": trial,
            "target_skill": skill_b_id,
            "skill_a": skill_a_id,
            "skill_b": skill_b_id,
            "sim_ab": sim_ab,
            "task_spec": task_spec_cb,
            "library": lib_spec_combo,
            "system_prompt": sys_p_cb,
        })

        state_text = (
            f"Upstream execution succeeded with tool {skill_a_id}. "
            f"Artifact for the downstream step: {task_a_desc}"
        )
        task_spec_state_b = TaskSpec(
            id=f"t{t_idx}_state_b_{trial}",
            instruction=(
                f"[Correct upstream state]\n{state_text}\n\n"
                f"Original multi-step task: {task_desc}\n\n"
                f"Given this realized upstream artifact, which single tool is needed for the downstream step?"
            ),
            required_skills=[{"name": skill_b_id}],
            gold_trace=[],
        )
        sys_p_state_b, _ = router_ref._build_prompt(task_spec_state_b, lib_spec_combo)
        jobs.append({
            "job_type": "state_b",
            "task_idx": t_idx,
            "trial": trial,
            "target_skill": skill_b_id,
            "skill_a": skill_a_id,
            "skill_b": skill_b_id,
            "sim_ab": sim_ab,
            "task_spec": task_spec_state_b,
            "library": lib_spec_combo,
            "system_prompt": sys_p_state_b,
        })

print(f"  {len(jobs)} LLM jobs prepared ({len(k2_tasks)} tasks × {TRIALS} trials × 5 job types).")


def llm_query(job: dict) -> dict:
    router = LLMRouter(model_name=MODEL)
    try:
        chosen_id, _ = router.route(job["task_spec"], job["library"])
    except Exception as e:
        print(f"  [ERROR] {job['job_type']} task={job['task_idx']} trial={job['trial']}: {e}")
        chosen_id = "ERROR"

    target = job["target_skill"]
    is_correct = (chosen_id == target)

    return {
        "timestamp":    time.time(),
        "job_type":     job["job_type"],
        "task_idx":     job["task_idx"],
        "trial":        job["trial"],
        "skill_a":      job["skill_a"],
        "skill_b":      job["skill_b"],
        "sim_ab":       job["sim_ab"],
        "target_skill": target,
        "chosen_skill": chosen_id,
        "is_correct":   is_correct,
        "system_prompt": job["system_prompt"],
    }


print(f"\nSubmitting {len(jobs)} queries with {MAX_WORKERS} threads...")
all_results = [None] * len(jobs)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_i = {executor.submit(llm_query, job): i for i, job in enumerate(jobs)}
    done = 0
    for future in as_completed(future_to_i):
        i = future_to_i[future]
        try:
            all_results[i] = future.result()
        except Exception as exc:
            print(f"  [FATAL] job {i}: {exc}")
        done += 1
        if done % 50 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} completed")


print("Writing raw query logs...")
with open(QUERIES_FILE, "w", encoding="utf-8") as f:
    for res in all_results:
        if res is not None:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
print(f"  Saved {sum(1 for r in all_results if r)} records → {QUERIES_FILE}")


print("Aggregating results...")
from collections import defaultdict
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path, finding_data_path, finding_path

agg = defaultdict(lambda: {
    "single_a": {}, "single_b": {}, "combo_a": {}, "combo_b": {}, "state_b": {},
    "skill_a": None, "skill_b": None, "sim_ab": None
})

for res in all_results:
    if res is None:
        continue
    ti = res["task_idx"]
    agg[ti]["skill_a"] = res["skill_a"]
    agg[ti]["skill_b"] = res["skill_b"]
    agg[ti]["sim_ab"]  = res["sim_ab"]
    agg[ti][res["job_type"]][res["trial"]] = int(res["is_correct"])

pair_results = []
for ti in sorted(agg.keys()):
    g = agg[ti]
    if not g["single_a"] or not g["single_b"] or not g["combo_a"] or not g["combo_b"]:
        continue
    p_a     = float(np.mean(list(g["single_a"].values())))
    p_b     = float(np.mean(list(g["single_b"].values())))
    p_state_b = float(np.mean(list(g["state_b"].values()))) if g["state_b"] else float("nan")

    trials_both = set(g["combo_a"].keys()) & set(g["combo_b"].keys())
    p_combo = float(np.mean([
        g["combo_a"][t] and g["combo_b"][t] for t in trials_both
    ])) if trials_both else 0.0

    a_correct_trials = [t for t in trials_both if g["combo_a"][t]]
    p_b_given_a = float(np.mean([g["combo_b"][t] for t in a_correct_trials])) if a_correct_trials else float("nan")
    p_ind   = p_a * p_b
    ratio   = p_combo / p_ind if p_ind > 1e-6 else float("nan")
    boost   = p_combo - p_ind
    cond_boost = (p_b_given_a - p_b) if not np.isnan(p_b_given_a) else float("nan")
    state_boost = (p_state_b - p_b) if not np.isnan(p_state_b) else float("nan")
    rescue_potential = (1.0 - p_b) * p_a

    pair_results.append({
        "task_idx":     ti,
        "skill_a":      g["skill_a"],
        "skill_b":      g["skill_b"],
        "sim":          round(g["sim_ab"], 4),
        "p_a":          round(p_a, 4),
        "p_b":          round(p_b, 4),
        "p_b_with_correct_state": round(p_state_b, 4) if not np.isnan(p_state_b) else None,
        "p_combo":      round(p_combo, 4),
        "p_independent": round(p_ind, 4),
        "ratio":        round(ratio, 4) if not np.isnan(ratio) else None,
        "boost":        round(boost, 4),
        "p_b_given_a":  round(p_b_given_a, 4) if not np.isnan(p_b_given_a) else None,
        "cond_boost":   round(cond_boost, 4) if not np.isnan(cond_boost) else None,
        "state_boost":  round(state_boost, 4) if not np.isnan(state_boost) else None,
        "rescue_potential": round(rescue_potential, 4),
        "n_trials":     len(g["single_a"]),
    })


print("\n─── RESULTS ───────────────────────────────────────────────")
print(f"{'#':>3}  {'skill_a':<25} {'skill_b':<25} {'sim':>5} "
      f"{'P(A)':>6} {'P(B)':>6} {'P(B|state)':>10} {'P(A,B)':>7} {'P*P':>6} {'ratio':>6} {'boost':>7} {'state':>7}")
for r in pair_results:
    ratio_s  = f"{r['ratio']:.2f}" if r["ratio"] is not None else "  nan"
    state_s = f"{r['p_b_with_correct_state']:.2f}" if r["p_b_with_correct_state"] is not None else "  nan"
    sbst_s = f"{r['state_boost']:+.3f}" if r["state_boost"] is not None else "   nan"
    print(f"{r['task_idx']:>3}  {r['skill_a']:<25} {r['skill_b']:<25} "
          f"{r['sim']:>5.3f} {r['p_a']:>6.2f} {r['p_b']:>6.2f} "
          f"{state_s:>10} {r['p_combo']:>7.2f} {r['p_independent']:>6.3f} {ratio_s:>6} {r['boost']:>+7.3f} "
          f"{sbst_s:>7}")

ratios = [r["ratio"] for r in pair_results if r["ratio"] is not None]
boosts = [r["boost"] for r in pair_results]
cboosts = [r["cond_boost"] for r in pair_results if r["cond_boost"] is not None]
state_boosts = [r["state_boost"] for r in pair_results if r["state_boost"] is not None]
print(f"\n  Mean ratio  P(A,B)/P(A)P(B)  = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
print(f"  Mean boost  P(A,B)-P(A)P(B)  = {np.mean(boosts):+.3f} ± {np.std(boosts):.3f}")
print(f"  Cond boost  P(B|A)-P(B)       = {np.mean(cboosts):+.3f} ± {np.std(cboosts):.3f}")
if state_boosts:
    print(f"  State boost P(B|state)-P(B)   = {np.mean(state_boosts):+.3f} ± {np.std(state_boosts):.3f}")
neg = sum(1 for b in boosts if b < 0)
pos = sum(1 for b in boosts if b > 0)
print(f"  Negative transfer: {neg}/{len(boosts)}  Positive: {pos}/{len(boosts)}")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(pair_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_FILE}")
print(f"Saved: {QUERIES_FILE}")
