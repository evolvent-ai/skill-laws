
import json, os, sys, time, re
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


from skill_law.runtime import StrictLLMRouter as LLMRouter
from skill_law.runtime import TaskSpec
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_path

def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

SKILLS_DIR    = str(SKILLS_DIR)
EMB_CACHE     = str(data_path("processed", "bge_skill_embeddings.npz"))
OUT_DIR       = str(finding_path("F07", "data", "final"))
MODEL         = _parse_model()
QUERIES_FILE  = f"{OUT_DIR}/raw_llm_queries_real_n100_{MODEL}.jsonl"
RESULTS_FILE  = f"{OUT_DIR}/results_real_n100_{MODEL}.json"
N_LIB         = 100
N_HARD_NEGS   = 20
TRIALS        = int(os.environ.get("TRIALS", "15"))
N_PER_BIN     = int(os.environ.get("N_PER_BIN", "10"))
N_BINS        = int(os.environ.get("N_BINS", "5"))
MAX_WORKERS   = int(os.environ.get("MAX_WORKERS", "25"))
TOP_K_CLARITY = 10

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading real skills (filtering noise)...")
all_names = sorted(
    n for n in os.listdir(SKILLS_DIR)
    if os.path.exists(os.path.join(SKILLS_DIR, n, "SKILL.md"))
)

clean_names = []
clean_descs = []
for name in all_names:
    with open(os.path.join(SKILLS_DIR, name, "SKILL.md"), encoding="utf-8") as f:
        raw = f.read()
    if "synthetic noise skill" in raw.lower():
        continue
    clean_names.append(name)
    clean_descs.append(raw[:800])

print(f"  Clean real skills: {len(clean_names)} (filtered {len(all_names)-len(clean_names)} noise)")

skill_specs_clean = [
    SkillSpec(id=n, name=n, description=d[:800], input_schema={}, output_schema={})
    for n, d in zip(clean_names, clean_descs)
]
skills_map = {s.id: s for s in skill_specs_clean}
skill_idx  = {name: i for i, name in enumerate(clean_names)}

print("Loading BGE embeddings and computing features...")
all_embs  = np.load(EMB_CACHE)["embs"]
all_names_full = sorted(n for n in os.listdir(SKILLS_DIR)
                        if os.path.exists(os.path.join(SKILLS_DIR, n, "SKILL.md")))
full_idx = {name: i for i, name in enumerate(all_names_full)}

clean_embs = np.array([all_embs[full_idx[n]] for n in clean_names])
sim_matrix = clean_embs @ clean_embs.T

clarities = np.zeros(len(clean_names))
radii     = np.zeros(len(clean_names))

for i in range(len(clean_names)):
    row = sim_matrix[i].copy(); row[i] = -1
    top_k = np.sort(row)[::-1][:TOP_K_CLARITY]
    clarities[i] = 1.0 - top_k.mean()
    radii[i]     = (sim_matrix[i].sum() - 1.0) / (len(clean_names) - 1)

print(f"  Clarity: mean={clarities.mean():.4f} std={clarities.std():.4f}")
print(f"  Radius:  mean={radii.mean():.4f} std={radii.std():.4f}")

def extract_task_from_desc(name: str, desc_raw: str) -> str:
    text = re.sub(r'^---.*?---\s*', '', desc_raw, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith('#')]
    if not lines:
        return f"Use the {name} tool."

    body = " ".join(lines)
    sentences = re.split(r'(?<=[.!?])\s+', body)
    if not sentences:
        return body[:250]

    task = sentences[0]
    if len(task) < 50 and len(sentences) > 1:
        task += " " + sentences[1]

    return task[:300].strip()

print(f"\nSelecting {N_PER_BIN} skills per clarity bin ({N_BINS} bins)...")
rng = np.random.default_rng(101)

bin_edges = np.percentile(clarities, np.linspace(0, 100, N_BINS + 1))
bin_edges[0] -= 1e-9; bin_edges[-1] += 1e-9

selected = []
for b in range(N_BINS):
    lo, hi = bin_edges[b], bin_edges[b + 1]
    mask = (clarities > lo) & (clarities <= hi)
    idxs = np.where(mask)[0]
    chosen = rng.choice(idxs, size=min(N_PER_BIN, len(idxs)), replace=False)
    for i in chosen:
        selected.append({
            "skill_name":  clean_names[i],
            "clarity":     float(clarities[i]),
            "radius":      float(radii[i]),
            "clarity_bin": b,
            "task_desc":   extract_task_from_desc(clean_names[i], clean_descs[i]),
        })
    print(f"  Bin {b} [{lo:.4f},{hi:.4f}]: {len(chosen)} skills")

print("\nSample real tasks extracted:")
for s in selected[:3]:
    print(f"  [{s['clarity_bin']}] {s['skill_name']}: {s['task_desc']}")

def build_library_n100(target_name: str, seed: int) -> list:
    trial_rng = np.random.default_rng(seed)
    ti = skill_idx[target_name]

    sims = sim_matrix[ti].copy(); sims[ti] = -1
    hard_idx = np.argsort(sims)[::-1][:N_HARD_NEGS]

    pool_idx = [i for i in range(len(clean_names)) if i != ti and i not in hard_idx]
    rand_idx = trial_rng.choice(pool_idx, size=(N_LIB - 1 - N_HARD_NEGS), replace=False)

    lib = [skills_map[target_name]]
    for i in hard_idx: lib.append(skill_specs_clean[i])
    for i in rand_idx: lib.append(skill_specs_clean[i])

    perm = trial_rng.permutation(len(lib))
    return [lib[p] for p in perm]

print(f"\nBuilding LLM jobs ({len(selected)} skills × {TRIALS} trials, N={N_LIB})...")
router_ref = LLMRouter(model_name=MODEL)
jobs = []

for entry in selected:
    skill_name = entry["skill_name"]
    for trial in range(TRIALS):
        lib = build_library_n100(skill_name, seed=abs(hash(skill_name)) + trial)
        lib_spec = LibrarySpec(id=f"lib_{skill_name}_t{trial}", skills=lib)
        task_spec = TaskSpec(
            id=f"{skill_name}_t{trial}",
            instruction=entry["task_desc"],
            required_skills=[{"name": skill_name}],
            gold_trace=[],
        )
        sys_p, _ = router_ref._build_prompt(task_spec, lib_spec)
        jobs.append({
            "skill_name":    skill_name,
            "clarity":       entry["clarity"],
            "radius":        entry["radius"],
            "clarity_bin":   entry["clarity_bin"],
            "trial":         trial,
            "task_spec":     task_spec,
            "library":       lib_spec,
            "system_prompt": sys_p,
        })

print(f"  {len(jobs)} total jobs prepared")

def llm_query(job: dict) -> dict:
    router = LLMRouter(model_name=MODEL)
    try:
        chosen_id, _ = router.route(job["task_spec"], job["library"])
    except Exception as e:
        chosen_id = "ERROR"

    target     = job["skill_name"]
    is_correct = (chosen_id == target)
    return {
        "timestamp":     time.time(),
        "skill_name":    target,
        "clarity":       job["clarity"],
        "radius":        job["radius"],
        "clarity_bin":   job["clarity_bin"],
        "trial":         job["trial"],
        "chosen_skill":  chosen_id,
        "is_correct":    is_correct,
    }

print(f"\nSubmitting {len(jobs)} queries ({MAX_WORKERS} threads)...")
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
        if done % 100 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} done")

with open(QUERIES_FILE, "w", encoding="utf-8") as f:
    for res in all_results:
        if res is not None:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

agg = defaultdict(lambda: {"trials": [], "clarity": None, "radius": None, "clarity_bin": None})
for res in all_results:
    if res is None: continue
    sk = res["skill_name"]
    agg[sk]["trials"].append(int(res["is_correct"]))
    agg[sk]["clarity"]     = res["clarity"]
    agg[sk]["radius"]      = res["radius"]
    agg[sk]["clarity_bin"] = res["clarity_bin"]

pair_results = []
for sk, g in sorted(agg.items()):
    acc = float(np.mean(g["trials"])) if g["trials"] else float("nan")
    pair_results.append({
        "skill_name":  sk,
        "clarity":     round(g["clarity"], 6),
        "radius":      round(g["radius"],  6),
        "clarity_bin": g["clarity_bin"],
        "accuracy":    round(acc, 4),
        "n_trials":    len(g["trials"]),
    })

from scipy.stats import pearsonr, spearmanr
from numpy.linalg import lstsq

cls_arr = np.array([r["clarity"]  for r in pair_results])
rad_arr = np.array([r["radius"]   for r in pair_results])
acc_arr = np.array([r["accuracy"] for r in pair_results])

print("\n─── PER-SKILL RESULTS (N=100) ──────────────────────────────────────────")
print(f"{'Skill':<35} {'Clarity':>8} {'Radius':>7} {'Bin':>4} {'Acc':>7}")
for r in sorted(pair_results, key=lambda x: x["clarity"]):
    print(f"{r['skill_name']:<35} {r['clarity']:>8.4f} {r['radius']:>7.4f} "
          f"{r['clarity_bin']:>4}  {r['accuracy']:>6.1%}")

print("\n─── BIN SUMMARY ─────────────────────────────────────────────────────────")
for b in range(N_BINS):
    bin_r = [r for r in pair_results if r["clarity_bin"] == b]
    if not bin_r: continue
    mc = np.mean([r["clarity"]  for r in bin_r])
    ma = np.mean([r["accuracy"] for r in bin_r])
    print(f"  Bin {b}: mean_clarity={mc:.4f}  mean_acc={ma:.1%}  n={len(bin_r)}")

if len(cls_arr) >= 2:
    r_p, p_p = pearsonr(cls_arr, acc_arr)
    r_s, p_s = spearmanr(cls_arr, acc_arr)
    r_rp, p_rp = pearsonr(rad_arr, acc_arr)
    print(f"\n  Pearson  r(clarity, acc) = {r_p:+.4f}  p={p_p:.4f}")
    print(f"  Spearman r(clarity, acc) = {r_s:+.4f}  p={p_s:.4f}")
    print(f"  Pearson  r(radius,  acc) = {r_rp:+.4f}  p={p_rp:.4f}")

    X = np.column_stack([cls_arr, rad_arr, np.ones(len(cls_arr))])
    coef, _, _, _ = lstsq(X, acc_arr, rcond=None)
    pred = X @ coef
    denom = np.sum((acc_arr - acc_arr.mean())**2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((acc_arr - pred)**2) / denom
    print(f"  OLS: acc = {coef[0]:+.3f}*clarity {coef[1]:+.3f}*radius {coef[2]:+.3f}  R²={r2:.3f}")
else:
    print("\n  Correlation skipped: at least two sampled skills are required.")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(pair_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_FILE}")
