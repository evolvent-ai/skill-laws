
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import numpy as np
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr

dotenv.load_dotenv()

from skill_law.runtime import StrictLLMRouter
from skill_law.runtime import TaskSpec
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

SKILLS_DIR = str(SKILLS_DIR)
EMB_CACHE = str(data_path("processed", "bge_skill_embeddings.npz"))
STRUCT_FILE = str(finding_data_path("F08", "data", "final", "structural_scores_all.json"))
def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

MODEL = _parse_model()
OUT_DIR = str(finding_path("F08", "data", "final"))
TASKS_FILE = f"{OUT_DIR}/hijack_validation_tasks_{MODEL}.json"
RAW_FILE = f"{OUT_DIR}/raw_hijack_validation_{MODEL}.jsonl"
ERROR_FILE = f"{OUT_DIR}/raw_hijack_validation_errors_{MODEL}.jsonl"
RESULTS_FILE = f"{OUT_DIR}/hijack_validation_results_{MODEL}.json"
LIBRARY_SIZES = [int(x) for x in os.environ.get("LIBRARY_SIZES", "150,200,250").split(",") if x]
HARD_NEG_MAP = {150: 30, 200: 40, 250: 55}
TRIALS = int(os.environ.get("TRIALS", "12"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "25"))
TOP_K_CLARITY = 10
TARGETS_PER_BIN = int(os.environ.get("TARGETS_PER_BIN", "6"))
N_BINS = int(os.environ.get("N_BINS", "5"))
SEED = 20260404

os.makedirs(OUT_DIR, exist_ok=True)
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key   = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=_api_key, base_url=_base_url, timeout=120.0)


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


def safe_corr(fn, x, y, k1, k2):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    stat, p = fn(x, y)
    return {k1: round(float(stat), 6), k2: round(float(p), 6)}


def gen_user_paraphrase(skill_name: str, desc_snippet: str) -> str:
    prompt = (
        "You are helping design a strict routing benchmark. "
        "Given a tool description, write a single natural user request (15-30 words) that asks for the capability. "
        "Do NOT mention the tool name, repository name, exact command names, or implementation details. "
        "Do NOT copy phrases verbatim from the description. "
        "Write only one sentence from the user's perspective.\n\n"
        f"Tool: {skill_name}\n"
        f"Description: {desc_snippet[:350]}\n\n"
        "User request:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        return extract_task_from_desc(skill_name, desc_snippet)


print("Loading clean real skills...")
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
clean_desc_map = {n: d for n, d in zip(clean_names, clean_descs)}
print(f"  Clean skills: {len(clean_names)}")

print("Loading embeddings and structural scores...")
all_embs = np.load(EMB_CACHE)["embs"]
full_idx = {name: i for i, name in enumerate(all_names)}
clean_embs = np.array([all_embs[full_idx[n]] for n in clean_names])
sim_matrix = clean_embs @ clean_embs.T

clarities = np.zeros(len(clean_names))
radii = np.zeros(len(clean_names))
for i in range(len(clean_names)):
    row = sim_matrix[i].copy()
    row[i] = -1
    top_k = np.sort(row)[::-1][:TOP_K_CLARITY]
    clarities[i] = 1.0 - top_k.mean()
    radii[i] = (sim_matrix[i].sum() - 1.0) / (len(clean_names) - 1)

if os.path.exists(STRUCT_FILE):
    struct_rows = json.load(open(STRUCT_FILE, encoding="utf-8"))
else:
    centrality = (sim_matrix.sum(axis=1) - 1.0) / max(1, len(clean_names) - 1)
    nearest = []
    for i in range(len(clean_names)):
        row = sim_matrix[i].copy()
        row[i] = -1
        nearest.append(float(np.sort(row)[::-1][:TOP_K_CLARITY].mean()))
    struct_rows = [
        {
            "skill_name": clean_names[i],
            "sink_score": float(centrality[i]),
            "victim_score": float(nearest[i]),
            "dominance": float(centrality[i] - (1.0 - nearest[i])),
        }
        for i in range(len(clean_names))
    ]
struct_map = {r["skill_name"]: r for r in struct_rows}

print("Selecting hijack-validation targets across clarity bins...")
rng = np.random.default_rng(SEED)
bin_edges = np.percentile(clarities, np.linspace(0, 100, N_BINS + 1))
bin_edges[0] -= 1e-9
bin_edges[-1] += 1e-9

selected = []
for b in range(N_BINS):
    lo, hi = bin_edges[b], bin_edges[b + 1]
    idxs = np.where((clarities > lo) & (clarities <= hi))[0]
    idxs = sorted(
        idxs,
        key=lambda i: (
            -float(struct_map.get(clean_names[i], {}).get("victim_score", 0.0)),
            -float(radii[i]),
            float(clarities[i]),
        ),
    )
    if len(idxs) <= TARGETS_PER_BIN:
        chosen = idxs
    else:
        anchors = np.linspace(0, len(idxs) - 1, TARGETS_PER_BIN, dtype=int)
        chosen = [idxs[j] for j in anchors]
    for i in chosen:
        struct = struct_map.get(clean_names[i], {})
        selected.append({
            "skill_name": clean_names[i],
            "clarity": float(clarities[i]),
            "radius": float(radii[i]),
            "clarity_bin": b,
            "sink_score": float(struct.get("sink_score", 0.0)),
            "victim_score": float(struct.get("victim_score", 0.0)),
            "dominance": float(struct.get("dominance", 0.0)),
            "desc_snippet": clean_descs[i][:400],
        })
    print(f"  Bin {b}: selected {len(chosen)} targets")

print(f"  Total selected targets: {len(selected)}")

if os.path.exists(TASKS_FILE):
    cached = {r["skill_name"]: r for r in json.load(open(TASKS_FILE, encoding="utf-8"))}
else:
    cached = {}

print("Generating/loading user paraphrases...")
for entry in selected:
    cached_row = cached.get(entry["skill_name"], {})
    para = cached_row.get("user_paraphrase")
    if not para:
        para = gen_user_paraphrase(entry["skill_name"], entry["desc_snippet"])
    entry["user_paraphrase"] = para

json.dump([
    {
        "skill_name": e["skill_name"],
        "clarity": round(e["clarity"], 6),
        "radius": round(e["radius"], 6),
        "clarity_bin": e["clarity_bin"],
        "sink_score": round(e["sink_score"], 6),
        "victim_score": round(e["victim_score"], 6),
        "dominance": round(e["dominance"], 6),
        "user_paraphrase": e["user_paraphrase"],
    }
    for e in selected
], open(TASKS_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"  Saved task cache -> {TASKS_FILE}")

def build_library(target_name: str, n_lib: int, seed: int) -> list:
    trial_rng = np.random.default_rng(seed)
    ti = skill_idx[target_name]
    n_hard = HARD_NEG_MAP[n_lib]
    sims = sim_matrix[ti].copy()
    sims[ti] = -1
    hard_idx = np.argsort(sims)[::-1][:n_hard]
    pool_idx = [i for i in range(len(clean_names)) if i != ti and i not in hard_idx]
    rand_idx = trial_rng.choice(pool_idx, size=(n_lib - 1 - n_hard), replace=False)
    lib = [skills_map[target_name]]
    for i in hard_idx:
        lib.append(skill_specs_clean[i])
    for i in rand_idx:
        lib.append(skill_specs_clean[i])
    perm = trial_rng.permutation(len(lib))
    return [lib[p] for p in perm]


print("Building jobs...")
router_ref = StrictLLMRouter(model_name=MODEL)
jobs = []
for entry in selected:
    for n_lib in LIBRARY_SIZES:
        for trial in range(TRIALS):
            seed = abs(hash((entry["skill_name"], n_lib, trial, SEED))) % (2**32)
            lib = build_library(entry["skill_name"], n_lib=n_lib, seed=seed)
            lib_spec = LibrarySpec(id=f"lib_{entry['skill_name']}_{n_lib}_t{trial}", skills=lib)
            task_spec = TaskSpec(
                id=f"{entry['skill_name']}_{n_lib}_t{trial}",
                instruction=entry["user_paraphrase"],
                required_skills=[{"name": entry["skill_name"]}],
                gold_trace=[],
            )
            sys_p, _ = router_ref._build_prompt(task_spec, lib_spec)
            jobs.append({
                "skill_name": entry["skill_name"],
                "clarity": entry["clarity"],
                "radius": entry["radius"],
                "clarity_bin": entry["clarity_bin"],
                "sink_score": entry["sink_score"],
                "victim_score": entry["victim_score"],
                "dominance": entry["dominance"],
                "library_size": n_lib,
                "trial": trial,
                "query_text": entry["user_paraphrase"],
                "task_spec": task_spec,
                "library": lib_spec,
                "system_prompt": sys_p,
            })
print(f"  Total jobs: {len(jobs)}")

completed_keys = set()
if os.path.exists(RAW_FILE):
    with open(RAW_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed_keys.add((row["skill_name"], row["library_size"], row["trial"]))
            except Exception:
                continue
    if completed_keys:
        before = len(jobs)
        jobs = [
            job for job in jobs
            if (job["skill_name"], job["library_size"], job["trial"]) not in completed_keys
        ]
        print(f"Resuming from {len(completed_keys)} completed jobs; {len(jobs)}/{before} remain.")


def llm_query(job: dict) -> dict:
    router = StrictLLMRouter(model_name=MODEL)
    try:
        chosen_id, route_meta = router.route(job["task_spec"], job["library"])
    except Exception as exc:
        chosen_id = "ERROR"
        route_meta = {"query": "exception", "error": str(exc)}
    target = job["skill_name"]
    is_correct = (chosen_id == target)
    is_null = chosen_id in (None, "None", "ERROR", "")
    is_hijack = (not is_correct) and (not is_null)
    return {
        "timestamp": time.time(),
        "skill_name": target,
        "clarity": job["clarity"],
        "radius": job["radius"],
        "clarity_bin": job["clarity_bin"],
        "sink_score": job["sink_score"],
        "victim_score": job["victim_score"],
        "dominance": job["dominance"],
        "library_size": job["library_size"],
        "trial": job["trial"],
        "query_text": job["query_text"],
        "chosen_skill": chosen_id,
        "is_correct": is_correct,
        "is_null": is_null,
        "is_hijack": is_hijack,
        "route_meta": route_meta,
    }


print(f"Submitting {len(jobs)} queries ({MAX_WORKERS} threads)...")
if jobs:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(RAW_FILE, "a", encoding="utf-8") as f:
        future_to_key = {
            executor.submit(llm_query, job): (job["skill_name"], job["library_size"], job["trial"])
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
                if done % 100 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} done")

print(f"Saved raw -> {RAW_FILE}")

all_results = []
with open(RAW_FILE, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            all_results.append(json.loads(line))
        except Exception:
            continue

by_condition = defaultdict(list)
absorber_count = Counter()
absorbed_from = defaultdict(Counter)
for res in all_results:
    if res is None:
        continue
    key = (res["skill_name"], res["library_size"])
    by_condition[key].append(res)
    if res["is_hijack"]:
        absorber_count[res["chosen_skill"]] += 1
        absorbed_from[res["chosen_skill"]][res["skill_name"]] += 1

condition_rows = []
for (skill_name, library_size), rows in sorted(by_condition.items()):
    n = len(rows)
    acc = np.mean([r["is_correct"] for r in rows])
    hij = np.mean([r["is_hijack"] for r in rows])
    nul = np.mean([r["is_null"] for r in rows])
    absorbed_by = Counter(r["chosen_skill"] for r in rows if r["is_hijack"])
    condition_rows.append({
        "skill_name": skill_name,
        "library_size": library_size,
        "clarity": round(float(rows[0]["clarity"]), 6),
        "radius": round(float(rows[0]["radius"]), 6),
        "clarity_bin": rows[0]["clarity_bin"],
        "sink_score": round(float(rows[0]["sink_score"]), 6),
        "victim_score": round(float(rows[0]["victim_score"]), 6),
        "dominance": round(float(rows[0]["dominance"]), 6),
        "accuracy": round(float(acc), 6),
        "error_rate": round(float(1 - acc), 6),
        "hijack_rate": round(float(hij), 6),
        "null_rate": round(float(nul), 6),
        "behavioral_victim_count": int(sum(absorbed_by.values())),
        "behavioral_absorber_count": int(absorber_count.get(skill_name, 0)),
        "absorber_diversity": len(absorbed_by),
        "absorbed_by": dict(absorbed_by),
        "n_trials": n,
    })

summary = []
for n_lib in LIBRARY_SIZES:
    subset = [r for r in condition_rows if r["library_size"] == n_lib]
    acc = np.array([r["accuracy"] for r in subset])
    hij = np.array([r["hijack_rate"] for r in subset])
    nul = np.array([r["null_rate"] for r in subset])
    rad = np.array([r["radius"] for r in subset])
    clr = np.array([r["clarity"] for r in subset])
    sink = np.array([r["sink_score"] for r in subset])
    victim = np.array([r["victim_score"] for r in subset])
    dom = np.array([r["dominance"] for r in subset])
    beh_victim = np.array([r["behavioral_victim_count"] for r in subset])
    beh_absorb = np.array([r["behavioral_absorber_count"] for r in subset])
    summary.append({
        "library_size": n_lib,
        "mean_accuracy": round(float(acc.mean()), 6),
        "mean_hijack_rate": round(float(hij.mean()), 6),
        "mean_null_rate": round(float(nul.mean()), 6),
        "radius_vs_accuracy": {
            "pearson": safe_corr(pearsonr, rad, acc, "r", "p"),
            "spearman": safe_corr(spearmanr, rad, acc, "rho", "p"),
        },
        "clarity_vs_accuracy": {
            "pearson": safe_corr(pearsonr, clr, acc, "r", "p"),
            "spearman": safe_corr(spearmanr, clr, acc, "rho", "p"),
        },
        "sink_vs_behavioral_absorber_count": {
            "pearson": safe_corr(pearsonr, sink, beh_absorb, "r", "p"),
            "spearman": safe_corr(spearmanr, sink, beh_absorb, "rho", "p"),
        },
        "victim_vs_behavioral_victim_count": {
            "pearson": safe_corr(pearsonr, victim, beh_victim, "r", "p"),
            "spearman": safe_corr(spearmanr, beh_victim, victim, "rho", "p"),
        },
        "victim_vs_hijack_rate": {
            "pearson": safe_corr(pearsonr, victim, hij, "r", "p"),
            "spearman": safe_corr(spearmanr, victim, hij, "rho", "p"),
        },
        "dominance_vs_accuracy": {
            "pearson": safe_corr(pearsonr, dom, acc, "r", "p"),
            "spearman": safe_corr(spearmanr, dom, acc, "rho", "p"),
        },
    })

behavioral_absorbers = []
for sk, ct in absorber_count.most_common(50):
    struct = struct_map.get(sk, {})
    sink_rank = None
    if sk in struct_map:
        ordered = sorted(struct_rows, key=lambda x: x["sink_score"], reverse=True)
        rank_map = {r["skill_name"]: i + 1 for i, r in enumerate(ordered)}
        sink_rank = rank_map.get(sk)
    behavioral_absorbers.append({
        "skill_name": sk,
        "behavioral_absorber_count": int(ct),
        "absorbed_from": dict(absorbed_from[sk]),
        "sink_score": struct.get("sink_score"),
        "victim_score": struct.get("victim_score"),
        "dominance": struct.get("dominance"),
        "structural_sink_rank": sink_rank,
    })

out = {
    "meta": {
        "model": MODEL,
        "library_sizes": LIBRARY_SIZES,
        "trials_per_condition": TRIALS,
        "selected_targets": len(selected),
        "total_jobs": len(jobs),
        "query_source": "user-paraphrase",
    },
    "selected_targets": [
        {
            "skill_name": e["skill_name"],
            "clarity": round(e["clarity"], 6),
            "radius": round(e["radius"], 6),
            "clarity_bin": e["clarity_bin"],
            "sink_score": round(e["sink_score"], 6),
            "victim_score": round(e["victim_score"], 6),
            "dominance": round(e["dominance"], 6),
            "user_paraphrase": e["user_paraphrase"],
        }
        for e in selected
    ],
    "condition_results": condition_rows,
    "summary": summary,
    "top_behavioral_absorbers": behavioral_absorbers,
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Saved results -> {RESULTS_FILE}")
print("\n=== SUMMARY ===")
for row in summary:
    print(
        f"N={row['library_size']:<3} acc={row['mean_accuracy']:.3f} "
        f"hijack={row['mean_hijack_rate']:.3f} null={row['mean_null_rate']:.3f}"
    )
