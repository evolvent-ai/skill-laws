
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

from skill_law.runtime import LLMRouter
from skill_law.runtime import TaskSpec
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path

SKILLS_DIR = str(SKILLS_DIR)
EMB_CACHE = str(data_path("processed", "bge_skill_embeddings.npz"))
BENCH_FILE = str(finding_data_path("F01", "data", "benchmark_tasks.json"))
STRUCT_FILE = str(finding_data_path("F08", "data", "final", "structural_scores_all.json"))
OUT_DIR = str(finding_path("F07", "data", "final"))
TASKS_FILE = f"{OUT_DIR}/query_source_tasks.json"

def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

MODEL = _parse_model()
RAW_FILE = f"{OUT_DIR}/raw_query_source_ablation_{MODEL}.jsonl"
RESULTS_FILE = f"{OUT_DIR}/query_source_ablation_results_{MODEL}.json"
QUERY_SOURCES = ["skill-desc", "benchmark", "user-paraphrase"]
LIBRARY_SIZES = [int(x) for x in os.environ.get("LIBRARY_SIZES", "100,150,200").split(",") if x]
HARD_NEG_MAP = {100: 20, 150: 30, 200: 40}
TRIALS = int(os.environ.get("TRIALS", "10"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "25"))
TOP_K_CLARITY = 10
TARGET_LIMIT = int(os.environ.get("TARGET_LIMIT", "0"))
SEED = 20260404

os.makedirs(OUT_DIR, exist_ok=True)
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key  = os.environ.get("OPENAI_API_KEY", "")
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


def normalize_skill_text(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def query_mentions_skill(query: str, skill_name: str) -> bool:
    q = normalize_skill_text(query)
    toks = [t for t in re.split(r'[-_:/.]+', skill_name.lower()) if t]
    joined = normalize_skill_text(skill_name)
    if joined and joined in q:
        return True
    hits = 0
    for tok in toks:
        if len(tok) >= 4 and tok in q:
            hits += 1
    return hits >= 2


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
            temperature=0.4,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        return f"Please help me with {skill_name.replace('-', ' ')}."


def safe_corr(fn, x, y, k1, k2):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    stat, p = fn(x, y)
    return {k1: round(float(stat), 6), k2: round(float(p), 6)}


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

print("Loading embeddings and computing clarity/radius...")
all_embs = np.load(EMB_CACHE)["embs"]
full_idx = {name: i for i, name in enumerate(all_names)}
clean_embs = np.array([all_embs[full_idx[n]] for n in clean_names])
sim_matrix = clean_embs @ clean_embs.T

clarities = np.zeros(len(clean_names))
radii = np.zeros(len(clean_names))
for i in range(len(clean_names)):
    row = sim_matrix[i].copy(); row[i] = -1
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

print("Loading strict benchmark queries...")
bench = json.load(open(BENCH_FILE, encoding="utf-8"))
benchmark_map = {}
rejected_named = 0
for item in bench:
    req = item.get("required_skills", [])
    if item.get("required_steps") != 1 or len(req) != 1:
        continue
    skill_name = req[0].get("name") or req[0].get("id")
    if not skill_name:
        continue
    q = item.get("task_desc", "").strip()
    if not q:
        continue
    if query_mentions_skill(q, skill_name):
        rejected_named += 1
        continue
    if skill_name not in benchmark_map:
        benchmark_map[skill_name] = q
print(f"  Strict benchmark queries kept: {len(benchmark_map)}  rejected for name leakage: {rejected_named}")

print("Selecting all benchmark-covered targets for strict comparison...")
rng = np.random.default_rng(SEED)
eligible_idxs = np.where(np.array([name in benchmark_map for name in clean_names]))[0]
selected = []
for i in eligible_idxs:
    rank = int(np.searchsorted(np.percentile(clarities, np.linspace(0, 100, 6)), clarities[i], side="right") - 1)
    rank = max(0, min(4, rank))
    selected.append({
        "skill_name": clean_names[i],
        "clarity": float(clarities[i]),
        "radius": float(radii[i]),
        "clarity_bin": rank,
        "desc_query": extract_task_from_desc(clean_names[i], clean_descs[i]),
        "benchmark_query": benchmark_map[clean_names[i]],
        "desc_snippet": clean_descs[i][:400],
    })

selected.sort(key=lambda x: x["clarity"])
if TARGET_LIMIT > 0:
    selected = selected[:TARGET_LIMIT]
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
        "skill_desc": e["desc_query"],
        "benchmark": e["benchmark_query"],
        "user_paraphrase": e["user_paraphrase"],
    }
    for e in selected
], open(TASKS_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"  Saved query-source tasks -> {TASKS_FILE}")

def build_library(target_name: str, n_lib: int, seed: int) -> list:
    trial_rng = np.random.default_rng(seed)
    ti = skill_idx[target_name]
    n_hard = HARD_NEG_MAP[n_lib]
    sims = sim_matrix[ti].copy(); sims[ti] = -1
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
router_ref = LLMRouter(model_name=MODEL)
jobs = []
for entry in selected:
    for source in QUERY_SOURCES:
        query_text = {
            "skill-desc": entry["desc_query"],
            "benchmark": entry["benchmark_query"],
            "user-paraphrase": entry["user_paraphrase"],
        }[source]
        for n_lib in LIBRARY_SIZES:
            for trial in range(TRIALS):
                seed = abs(hash((entry["skill_name"], source, n_lib, trial))) % (2**32)
                lib = build_library(entry["skill_name"], n_lib=n_lib, seed=seed)
                lib_spec = LibrarySpec(id=f"lib_{entry['skill_name']}_{source}_{n_lib}_t{trial}", skills=lib)
                task_spec = TaskSpec(
                    id=f"{entry['skill_name']}_{source}_{n_lib}_t{trial}",
                    instruction=query_text,
                    required_skills=[{"name": entry["skill_name"]}],
                    gold_trace=[],
                )
                sys_p, _ = router_ref._build_prompt(task_spec, lib_spec)
                jobs.append({
                    "skill_name": entry["skill_name"],
                    "clarity": entry["clarity"],
                    "radius": entry["radius"],
                    "clarity_bin": entry["clarity_bin"],
                    "query_source": source,
                    "library_size": n_lib,
                    "trial": trial,
                    "query_text": query_text,
                    "task_spec": task_spec,
                    "library": lib_spec,
                    "system_prompt": sys_p,
                })
print(f"  Total jobs: {len(jobs)}")


def llm_query(job: dict) -> dict:
    router = LLMRouter(model_name=MODEL)
    try:
        chosen_id, _ = router.route(job["task_spec"], job["library"])
    except Exception:
        chosen_id = "ERROR"
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
        "query_source": job["query_source"],
        "library_size": job["library_size"],
        "trial": job["trial"],
        "query_text": job["query_text"],
        "chosen_skill": chosen_id,
        "is_correct": is_correct,
        "is_null": is_null,
        "is_hijack": is_hijack,
    }


print(f"Submitting {len(jobs)} queries ({MAX_WORKERS} threads)...")
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

with open(RAW_FILE, "w", encoding="utf-8") as f:
    for res in all_results:
        if res is not None:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
print(f"Saved raw -> {RAW_FILE}")

by_condition = defaultdict(list)
absorber_count = Counter()
for res in all_results:
    if res is None:
        continue
    key = (res["skill_name"], res["query_source"], res["library_size"])
    by_condition[key].append(res)
    if res["is_hijack"]:
        absorber_count[res["chosen_skill"]] += 1

condition_rows = []
for (skill_name, query_source, library_size), rows in sorted(by_condition.items()):
    n = len(rows)
    acc = np.mean([r["is_correct"] for r in rows])
    hij = np.mean([r["is_hijack"] for r in rows])
    nul = np.mean([r["is_null"] for r in rows])
    absorbed_by = Counter(r["chosen_skill"] for r in rows if r["is_hijack"])
    struct = struct_map.get(skill_name, {})
    condition_rows.append({
        "skill_name": skill_name,
        "query_source": query_source,
        "library_size": library_size,
        "clarity": round(float(rows[0]["clarity"]), 6),
        "radius": round(float(rows[0]["radius"]), 6),
        "clarity_bin": rows[0]["clarity_bin"],
        "sink_score": struct.get("sink_score"),
        "victim_score": struct.get("victim_score"),
        "dominance": struct.get("dominance"),
        "accuracy": round(float(acc), 6),
        "error_rate": round(float(1 - acc), 6),
        "hijack_rate": round(float(hij), 6),
        "null_rate": round(float(nul), 6),
        "behavioral_victim_count": int(sum(absorbed_by.values())),
        "absorber_diversity": len(absorbed_by),
        "absorbed_by": dict(absorbed_by),
        "n_trials": n,
    })

summary = []
for source in QUERY_SOURCES:
    for n_lib in LIBRARY_SIZES:
        subset = [r for r in condition_rows if r["query_source"] == source and r["library_size"] == n_lib]
        acc = np.array([r["accuracy"] for r in subset])
        hij = np.array([r["hijack_rate"] for r in subset])
        nul = np.array([r["null_rate"] for r in subset])
        rad = np.array([r["radius"] for r in subset])
        clr = np.array([r["clarity"] for r in subset])
        sink = np.array([r["sink_score"] for r in subset])
        victim = np.array([r["victim_score"] for r in subset])
        beh_victim = np.array([r["behavioral_victim_count"] for r in subset])
        summary.append({
            "query_source": source,
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
            "victim_vs_behavioral_victim": {
                "pearson": safe_corr(pearsonr, victim, beh_victim, "r", "p"),
                "spearman": safe_corr(spearmanr, victim, beh_victim, "rho", "p"),
            },
            "sink_vs_hijack_rate": {
                "pearson": safe_corr(pearsonr, sink, hij, "r", "p"),
                "spearman": safe_corr(spearmanr, sink, hij, "rho", "p"),
            },
        })

out = {
    "meta": {
        "model": MODEL,
        "query_sources": QUERY_SOURCES,
        "library_sizes": LIBRARY_SIZES,
        "trials_per_condition": TRIALS,
        "selected_targets": len(selected),
        "total_jobs": len(jobs),
        "strict_benchmark_queries_available": len(benchmark_map),
    },
    "selected_targets": [
        {
            "skill_name": e["skill_name"],
            "clarity": round(e["clarity"], 6),
            "radius": round(e["radius"], 6),
            "clarity_bin": e["clarity_bin"],
        }
        for e in selected
    ],
    "condition_results": condition_rows,
    "summary": summary,
    "top_behavioral_absorbers": [
        {"skill_name": sk, "count": int(ct), "structural_sink_score": struct_map.get(sk, {}).get("sink_score")}
        for sk, ct in absorber_count.most_common(30)
    ],
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Saved results -> {RESULTS_FILE}")
print("\n=== SUMMARY ===")
for row in summary:
    print(f"{row['query_source']:<15} N={row['library_size']:<3} acc={row['mean_accuracy']:.3f} hijack={row['mean_hijack_rate']:.3f} null={row['mean_null_rate']:.3f}")
