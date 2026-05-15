
import json, os, re, time, threading, sys
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from skill_law.paths import SKILLS_DIR, data_path, finding_data_path, finding_path
from skill_law.runtime import load_env

load_env()
def _parse_model(default="gpt-5.4-mini"):
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default

MODEL = _parse_model()
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key   = os.environ.get("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=_api_key, base_url=_base_url, timeout=120.0)
JUDGE_MODEL = "gpt-5.4-mini"
N_TRIALS = int(os.environ.get("N_TRIALS", "10"))
PAIR_LIMIT = int(os.environ.get("PAIR_LIMIT", "0"))

PAIRS_FILE = str(finding_data_path("F12", "data", "F12_pairs.json"))
SKILLS_DIR = str(SKILLS_DIR)
DATA_DIR   = str(finding_path("F12", "data"))
RAW_OUT    = os.path.join(DATA_DIR, f"F12_raw_{MODEL}.jsonl")
log_lock   = threading.Lock()

def load_skill(skill_id):
    path = os.path.join(SKILLS_DIR, skill_id, 'SKILL.md')
    if not os.path.exists(path): return None, None
    content = open(path).read()
    parts = content.split('---', 2)
    body = parts[2].strip() if len(parts) >= 3 else ''
    m = re.search(r'description:\s*(.*?)(?=\nmodel:|\nallowed|\nversion|\nmetadata|\n---|\Z)', content, re.DOTALL)
    desc = m.group(1).strip().strip('|').strip() if m else ''
    return desc, body if len(body) > 100 else None


def exec_A_solo(pair, seed):
    a_desc, a_body = load_skill(pair['a_id'])
    prompt = f"""You are executing a single software skill/tool.

SKILL: {pair['a_id']}
DESCRIPTION: {a_desc}

SKILL INSTRUCTIONS:
{a_body[:1500]}

TASK (complete only the part this skill is responsible for):
{pair['task_desc']}

Produce complete, correct output. Do NOT explain — just output the concrete result."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, seed=seed)
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "[FAILED]"

def exec_B_solo(pair, seed):
    a_desc, a_body = load_skill(pair['a_id'])
    b_desc, b_body = load_skill(pair['b_id'])

    prompt_gen_a = f"""Generate a complete, realistic, perfect output for the following software skill:

SKILL: {pair['a_id']}
DESCRIPTION: {a_desc}
SKILL INSTRUCTIONS:
{a_body[:1200]}
TASK: {pair['task_desc']}

Produce a realistic, detailed, complete output as if this skill ran successfully.
Do NOT explain — output only the concrete result (code, data, config, etc.)."""
    
    perfect_a_output = "[FAILED]"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_gen_a}],
                temperature=0.3, seed=seed + 5000)
            perfect_a_output = resp.choices[0].message.content.strip()
            break
        except Exception:
            time.sleep(2 * (attempt + 1))

    prompt_b = f"""You are executing a single software skill/tool.

SKILL: {pair['b_id']}
DESCRIPTION: {b_desc}

SKILL INSTRUCTIONS:
{b_body[:1500]}

TASK: {pair['task_desc']}

UPSTREAM INPUT (perfect output from the previous step — {pair['a_id']}):
{perfect_a_output[:1500]}

Use the upstream input above. Produce complete, correct output for YOUR skill.
Do NOT explain — just output the concrete result."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_b}],
                temperature=0.7, seed=seed + 6000)
            return resp.choices[0].message.content.strip(), perfect_a_output
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "[FAILED]", perfect_a_output

def exec_joint_natural(pair, seed):
    a_desc, a_body = load_skill(pair['a_id'])
    b_desc, b_body = load_skill(pair['b_id'])

    system_prompt = f"""You are an autonomous AI software engineer completing a multi-step task.
You have access to two specific tools/skills. Execute them in the correct order to complete the task.

═══ TOOL 1: {pair['a_id']} ═══
Description: {a_desc}
Instructions:
{a_body[:1200]}

═══ TOOL 2: {pair['b_id']} ═══
Description: {b_desc}
Instructions:
{b_body[:1200]}

Important: Execute Tool 1 first, then pass its output to Tool 2. 
Produce the FINAL combined output that completes the task end-to-end.
Do NOT explain — just output all concrete results (code, data, config)."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"TASK: {pair['task_desc']}"}
                ],
                temperature=0.7, seed=seed, max_tokens=2500)
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "[FAILED]"


def judge(rubric, task_desc, output, skill_context=""):
    if not rubric:
        return 0.0, []
    criteria_text = "\n".join([
        f"{i+1}. [{r['weight']:.0%} weight] {r['criterion']}\n"
        f"   PASS if: {r['pass_condition']}\n"
        f"   FAIL if: {r['fail_condition']}"
        for i, r in enumerate(rubric)])
    prompt = f"""You are an expert evaluator.
{skill_context}
TASK: {task_desc}

OUTPUT TO EVALUATE:
{output[:2000]}

RUBRIC (score each criterion independently: 0.0=fail, 0.5=partial, 1.0=pass):
{criteria_text}

Respond ONLY with valid JSON:
{{"criterion_scores": [{{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line>"}}]}}"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"})
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                r = json.loads(m.group(0))
                cs = r.get('criterion_scores', [])
                total = sum(rubric[i]['weight'] * float(cs[i].get('score', 0.0))
                           for i in range(min(len(rubric), len(cs))))
                return total, cs
        except Exception:
            time.sleep(2 * (attempt + 1))
    return 0.0, []


def evaluate_trial(pair, trial_idx):
    seed = abs(hash((pair['a_id'], pair['b_id'], trial_idx, 'F12'))) % (2**31)

    out_a = exec_A_solo(pair, seed)
    s_a, cs_a = judge(pair.get('rubric_a', []), pair['task_desc'], out_a,
                      f"Evaluating ONLY Step 1 ({pair['a_id']}) output.")

    out_b, perfect_a = exec_B_solo(pair, seed)
    s_b, cs_b = judge(pair.get('rubric_b', []), pair['task_desc'], out_b,
                      f"Evaluating ONLY Step 2 ({pair['b_id']}) given perfect upstream input.")

    out_joint = exec_joint_natural(pair, seed)
    s_joint, cs_joint = judge(pair.get('rubric', []), pair['task_desc'], out_joint,
                              f"Evaluating combined output of {pair['a_id']} → {pair['b_id']}.")

    s_independent = s_a * s_b
    s_harmonic    = (2 * s_a * s_b / (s_a + s_b)) if (s_a + s_b) > 0 else 0.0
    s_min         = min(s_a, s_b)
    synergy       = s_joint - s_independent

    result = {
        'a_id': pair['a_id'], 'b_id': pair['b_id'],
        'dependency': pair.get('dependency'),
        'trial': trial_idx,
        's_a': s_a, 's_b': s_b,
        's_joint': s_joint,
        's_independent': s_independent,
        's_harmonic': s_harmonic,
        's_min': s_min,
        'synergy': synergy,
        'out_a_len': len(out_a), 'out_b_len': len(out_b), 'out_joint_len': len(out_joint),
    }

    with log_lock:
        with open(RAW_OUT, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    return result

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    pairs = json.load(open(PAIRS_FILE))
    if PAIR_LIMIT > 0:
        pairs = pairs[:PAIR_LIMIT]
    print(f"Loaded {len(pairs)} pairs")
    from collections import Counter
    print(f"Distribution: {dict(Counter(p['dependency'] for p in pairs))}")
    total = len(pairs) * N_TRIALS
    print(f"Total evaluations: {len(pairs)} × {N_TRIALS} = {total}")
    print(f"Model: {MODEL}\n")

    done = set()
    if os.path.exists(RAW_OUT):
        with open(RAW_OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['a_id'], r['b_id'], r['trial']))
                except Exception:
                    pass
    if done:
        print(f"Resuming: {len(done)} trials already done, skipping them.")
    else:
        open(RAW_OUT, 'w').close()

    jobs = [(p, t) for p in pairs for t in range(N_TRIALS)
            if (p['a_id'], p['b_id'], t) not in done]
    print(f"Remaining jobs: {len(jobs)}")

    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(evaluate_trial, p, t): (p['a_id'], t) for p, t in jobs}
        for future in tqdm(as_completed(futures), total=len(jobs), desc="F12"):
            res = future.result()
            if res: results.append(res)

    print("\n=== Results by Dependency Type ===")
    import statistics
    for dep in ['tight', 'loose', 'independent']:
        rows = [r for r in results if r['dependency'] == dep]
        if not rows: continue
        sa   = statistics.mean(r['s_a'] for r in rows)
        sb   = statistics.mean(r['s_b'] for r in rows)
        sj   = statistics.mean(r['s_joint'] for r in rows)
        si   = statistics.mean(r['s_independent'] for r in rows)
        sh   = statistics.mean(r['s_harmonic'] for r in rows)
        syn  = statistics.mean(r['synergy'] for r in rows)
        print(f"\n  {dep.upper()} (n={len(rows)}):")
        print(f"    S(A)         = {sa:.3f}")
        print(f"    S(B)         = {sb:.3f}")
        print(f"    S_joint      = {sj:.3f}  ← actual")
        print(f"    S_A×S_B      = {si:.3f}  ← independence baseline")
        print(f"    H(S_A,S_B)   = {sh:.3f}  ← harmonic mean")
        print(f"    Synergy      = {syn:+.3f}  (joint − independent)")

    run_routing_crowding_experiment(pairs)


def run_routing_crowding_experiment(pairs):
    from skill_law.runtime import LLMRouter, LibrarySpec, TaskSpec, SkillSpec, load_skills
    from skill_law.paths import SKILLS_DIR as SKILLS_PATH

    router = LLMRouter(model_name=MODEL)
    all_skills = load_skills(SKILLS_PATH)
    skills_map = {s.id: s for s in all_skills}

    routing_out = os.path.join(DATA_DIR, f"F12_routing_crowding_{MODEL}.jsonl")
    routing_results = []
    n_routing_trials = min(N_TRIALS, 5)

    same_family_pairs = [p for p in pairs if p.get('dependency') == 'tight']
    if not same_family_pairs:
        same_family_pairs = pairs[:10]

    print(f"\n=== Routing Crowding Experiment ({len(same_family_pairs)} pairs × {n_routing_trials} trials) ===")

    def identify_strong_weak(pair):
        a_desc, _ = load_skill(pair['a_id'])
        b_desc, _ = load_skill(pair['b_id'])
        a_len = len(a_desc or '')
        b_len = len(b_desc or '')
        if a_len >= b_len:
            return pair['a_id'], pair['b_id'], a_desc, b_desc
        return pair['b_id'], pair['a_id'], b_desc, a_desc

    def route_one_trial(pair, trial_idx):
        strong_id, weak_id, strong_desc, weak_desc = identify_strong_weak(pair)
        seed = abs(hash((strong_id, weak_id, trial_idx, 'F12_routing'))) % (2**31)
        rng = __import__('random').Random(seed)

        weak_spec = skills_map.get(weak_id)
        if weak_spec is None:
            return None
        query = f"Use the {weak_id.replace('-', ' ').replace('_', ' ')} tool to complete: {pair['task_desc']}"

        strong_spec = skills_map.get(strong_id)
        if strong_spec is None:
            return None

        distractor_pool = [s for s in all_skills if s.id not in (strong_id, weak_id)]
        n_distractors = min(18, len(distractor_pool))
        distractors = rng.sample(distractor_pool, n_distractors)

        lib_skills = [strong_spec, weak_spec] + distractors
        rng.shuffle(lib_skills)
        lib_spec = LibrarySpec(id=f"crowding_{weak_id}_{trial_idx}", skills=lib_skills)
        task_spec = TaskSpec(id=f"crowding_{weak_id}_{trial_idx}", instruction=query, required_skills=[{"name": weak_id}])

        try:
            chosen_id, meta = router.route(task_spec, lib_spec)
        except Exception:
            chosen_id = "ERROR"

        is_correct = chosen_id == weak_id
        is_hijack_by_strong = chosen_id == strong_id

        return {
            'strong_id': strong_id,
            'weak_id': weak_id,
            'trial': trial_idx,
            'chosen_id': chosen_id,
            'is_correct': is_correct,
            'is_hijack_by_strong': is_hijack_by_strong,
            'library_size': len(lib_skills),
            'dependency': pair.get('dependency'),
        }

    jobs = [(p, t) for p in same_family_pairs for t in range(n_routing_trials)]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(route_one_trial, p, t) for p, t in jobs]
        for future in tqdm(as_completed(futures), total=len(jobs), desc="F12-routing"):
            res = future.result()
            if res:
                routing_results.append(res)

    with open(routing_out, 'w') as f:
        for row in routing_results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    if routing_results:
        import statistics
        acc = statistics.mean(r['is_correct'] for r in routing_results)
        hijack = statistics.mean(r['is_hijack_by_strong'] for r in routing_results)
        print(f"\n  Routing crowding results (n={len(routing_results)}):")
        print(f"    Weak-skill accuracy   = {acc:.3f}")
        print(f"    Strong-skill hijack   = {hijack:.3f}")
        print(f"    Crowding effect       = {hijack:.3f} (fraction of weak queries captured by strong)")
    print(f"  Results saved to: {routing_out}")


if __name__ == '__main__':
    main()
