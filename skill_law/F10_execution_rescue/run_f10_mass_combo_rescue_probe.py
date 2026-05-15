import json, os, random, time
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
from skill_law.paths import SKILLS_DIR, data_path, finding_path
from skill_law.runtime import load_env

load_env()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.bltcy.ai/v1")
)

skill_dir = str(SKILLS_DIR)
all_skills = []
for d in sorted(os.listdir(skill_dir)):
    p = os.path.join(skill_dir, d, 'SKILL.md')
    if os.path.exists(p):
        all_skills.append({'name': d, 'content': open(p).read()})

synth = json.load(open(data_path("processed", "synthetic_skill_library.json")))
synth_names = [s['name'] for s in synth]
synth_descs = [s['description'] for s in synth]

if os.environ.get("SKILL_LAW_LEXICAL_EMBEDDINGS", "0") == "1":
    from sklearn.feature_extraction.text import TfidfVectorizer
    embs = TfidfVectorizer(max_features=4096).fit_transform(synth_descs).toarray()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms == 0, 1.0, norms)
else:
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embs = bge.encode(synth_descs, normalize_embeddings=True, show_progress_bar=False)
sim_matrix = embs @ embs.T

all_skill_names = {s['name'] for s in all_skills}
valid_idx = [i for i, n in enumerate(synth_names) if n in all_skill_names]

bins = [
    (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
    (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
]
PAIRS_PER_BIN = int(os.environ.get("PAIRS_PER_BIN", "5"))
selected_pairs = []

random.seed(42)
for lo, hi in bins:
    candidates = []
    for i in valid_idx:
        for j in valid_idx:
            if i >= j: continue
            if lo <= sim_matrix[i, j] < hi:
                candidates.append((i, j, float(sim_matrix[i, j])))

    if len(candidates) > PAIRS_PER_BIN:
        chosen = random.sample(candidates, PAIRS_PER_BIN)
    else:
        chosen = candidates

    for i, j, sim in chosen:
        selected_pairs.append({
            'skill_a': synth_names[i], 'skill_b': synth_names[j],
            'desc_a': synth_descs[i], 'desc_b': synth_descs[j],
            'sim': sim, 'bin': f"{lo:.1f}-{hi:.1f}"
        })

print(f"Generated {len(selected_pairs)} pairs across {len(bins)} bins.")

def call_llm(sys_p, task, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role":"system","content":sys_p}, {"role":"user","content":task}],
                temperature=0.0, max_tokens=60, timeout=10
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2 ** attempt)
    return ""

def run_routing(task, targets, N=80, trials=10):
    success = 0

    def run_single(trial):
        bg = [s for s in all_skills if s['name'] not in targets]
        rng = random.Random(42 + trial)
        bg_sample = rng.sample(bg, N - len(targets))
        lib = bg_sample + [s for s in all_skills if s['name'] in targets]
        rng.shuffle(lib)

        sys_p = "You are a tool router. Tools:\n\n"
        for sk in lib:
            sys_p += f"Name: {sk['name']}\n{sk['content'][:250]}...\n\n"
        if len(targets) == 1:
            sys_p += "Output ONLY the exact tool name. Nothing else."
        else:
            sys_p += "Output ONLY the exact tool names in order, comma-separated. Nothing else."

        pred = call_llm(sys_p, task)
        pred_list = [x.strip() for x in pred.split(',')]
        return 1 if pred_list == targets else 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_single, t) for t in range(trials)]
        for f in as_completed(futures):
            success += f.result()

    return success / trials

N_SIZE = int(os.environ.get("N_SIZE", "80"))
TRIALS = int(os.environ.get("TRIALS", "12"))
OUTPUT_FILE = finding_path("F10", "data", "F10_real_combo_mass_v4.jsonl")

print(f"Starting mass-scale test (N={N_SIZE}, Trials={TRIALS})")
open(OUTPUT_FILE, 'w').close()

for idx, pair in enumerate(selected_pairs):
    a, b = pair['skill_a'], pair['skill_b']
    da = pair['desc_a'][:80].lower().rstrip('.')
    db = pair['desc_b'][:80].lower().rstrip('.')
    task_a = f"Use the tool that {da}."
    task_b = f"Use the tool that {db}."
    task_combo = f"First: {task_a}  Then: {task_b}"

    print(f"[{idx+1}/{len(selected_pairs)}] sim={pair['sim']:.3f} | {a[:20]} + {b[:20]}", end=" ", flush=True)

    p_a = run_routing(task_a, [a], N_SIZE, TRIALS)
    p_b = run_routing(task_b, [b], N_SIZE, TRIALS)
    p_combo = run_routing(task_combo, [a, b], N_SIZE, TRIALS)

    p_ind = p_a * p_b
    ratio = p_combo / p_ind if p_ind > 0 else float('nan')
    diff = p_combo - p_ind

    print(f"| P(A)={p_a:.2f} P(B)={p_b:.2f} P(A,B)={p_combo:.2f} | ratio={ratio:.2f}")

    pair.update({
        'p_a': p_a, 'p_b': p_b, 'p_combo': p_combo,
        'p_independent': p_ind, 'ratio': ratio, 'diff': diff
    })

    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(pair) + '\n')

print("All done!")
