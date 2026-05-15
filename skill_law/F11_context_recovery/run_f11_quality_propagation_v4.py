
import json, os, re, time, threading, sys
import numpy as np
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
N_TRIALS = int(os.environ.get("N_TRIALS", "8"))
PAIR_LIMIT = int(os.environ.get("PAIR_LIMIT", "0"))
PAIRS_FILE = str(finding_data_path("F11", "data", "F11_pairs_v4_rubrics.json"))
DATA_DIR = str(finding_path("F11", "data"))
RAW_OUT = os.path.join(DATA_DIR, f"F11_raw_v4_{MODEL}.jsonl")
SKILLS_DIR = str(SKILLS_DIR)

log_lock = threading.Lock()
QUALITY_LEVELS = ["perfect", "degraded", "noisy", "wrong"]

QUALITY_PROMPTS = {
    "perfect":  "OUTPUT REQUIREMENT: Perfect execution. Produce complete, correct, working output that fully solves the skill's goal.",
    "degraded": "OUTPUT REQUIREMENT: Intentionally degraded execution. Your output MUST contain 1-2 concrete flaws: e.g. a broken path, missing variable, or hardcoded value that should be parameterized. The structure is correct but it will fail in practice.",
    "noisy":    "OUTPUT REQUIREMENT: Noisy execution. Mix correct parts with clearly wrong parts: use placeholder values like <YOUR_VALUE>, leave TODO comments, mix in unrelated content, or use wrong parameters. About half the output is useful, half is garbage.",
    "wrong":    "OUTPUT REQUIREMENT: Wrong execution. COMPLETELY miss the point. Output something that looks like an attempt but fundamentally fails — wrong tool, wrong format, wrong problem solved entirely. Do NOT produce a correct or near-correct output.",
}

def load_skill_body(skill_id):
    path = os.path.join(SKILLS_DIR, skill_id, 'SKILL.md')
    if not os.path.exists(path):
        return None
    content = open(path).read()
    parts = content.split('---', 2)
    body = parts[2].strip() if len(parts) >= 3 else ''
    return body if len(body) > 100 else None

def execute_skill_a(pair, quality_level, seed):
    qp = QUALITY_PROMPTS[quality_level]
    a_body = load_skill_body(pair['a_id']) or ''
    prompt = f"""You are executing a software skill/tool.

SKILL: {pair['a_id']}
DESCRIPTION: {pair.get('a_desc', '')}

SKILL INSTRUCTIONS:
{a_body[:1200]}

TASK: {pair['task_desc']}

{qp}

Produce the output now. Do NOT explain — just output the concrete result (code, config, commands, data, etc.)."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                seed=seed,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "[EXECUTION FAILED]"

def execute_skill_b(pair, a_output, seed):
    b_body = load_skill_body(pair['b_id']) or ''
    prompt = f"""You are executing a software skill/tool.

SKILL: {pair['b_id']}
DESCRIPTION: {pair.get('b_desc', '')}

SKILL INSTRUCTIONS:
{b_body[:1200]}

TASK: {pair['task_desc']}

UPSTREAM SKILL OUTPUT (from Step 1, use this as your input/context):
{a_output[:1500]}

Execute this skill. Do NOT explain — just produce the concrete result."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                seed=seed + 1000,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "[EXECUTION FAILED]"

def judge_with_rubric(pair, a_output, b_output):
    rubric = pair.get('rubric', [])
    dep = pair.get('dependency', 'independent')

    criteria_text = "\n".join([
        f"{i+1}. [{r['weight']:.0%} weight] {r['criterion']}\n"
        f"   PASS if: {r['pass_condition']}\n"
        f"   FAIL if: {r['fail_condition']}"
        for i, r in enumerate(rubric)
    ])

    dep_note = ""
    if dep == 'tight':
        dep_note = "\nIMPORTANT: Criteria 1 and 2 specifically test whether Step 2 used Step 1's ACTUAL output data. If Step 2 invented its own data or used generic placeholders instead of Step 1's specific output, Criterion 1 MUST be scored 0.0."
    elif dep == 'loose':
        dep_note = "\nIMPORTANT: Criterion 1 tests whether Step 2 shows evidence of using Step 1's output as context. If Step 2 is completely generic with no reference to Step 1's output structure or values, Criterion 1 MUST be scored 0.0."

    prompt = f"""You are an expert evaluator for a 2-step AI agent task.

TASK: {pair['task_desc']}

STEP 1 OUTPUT ({pair['a_id']}):
{a_output[:1200]}

STEP 2 OUTPUT ({pair['b_id']}):
{b_output[:1200]}

RUBRIC (score each criterion independently: 0.0 = fail, 0.5 = partial, 1.0 = pass):
{criteria_text}
{dep_note}

Also assess: did Step 2 meaningfully USE Step 1's output?
- "yes": Step 2 explicitly references/incorporates specific data from Step 1
- "partial": Step 2 uses some context from Step 1 but ignores key parts
- "ignored": Step 2 appears to have solved the problem independently without using Step 1's output

Respond ONLY with valid JSON:
{{
  "criterion_scores": [
    {{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line reason>"}},
    {{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line reason>"}},
    {{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line reason>"}},
    {{"criterion": "<name>", "score": <0.0|0.5|1.0>, "note": "<one line reason>"}}
  ],
  "used_upstream": "yes|partial|ignored"
}}"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                r = json.loads(m.group(0))
                scores_list = r.get('criterion_scores', [])
                total = 0.0
                for i, rb in enumerate(rubric):
                    if i < len(scores_list):
                        total += rb['weight'] * float(scores_list[i].get('score', 0.0))
                return total, r.get('used_upstream', 'unknown'), scores_list
        except Exception:
            time.sleep(2 * (attempt + 1))
    return 0.0, 'unknown', []

def evaluate_trial(pair, quality_level, trial_idx):
    seed = abs(hash((pair['a_id'], pair['b_id'], quality_level, trial_idx))) % (2**31)
    a_output = execute_skill_a(pair, quality_level, seed)
    b_output = execute_skill_b(pair, a_output, seed)
    score, used, criterion_scores = judge_with_rubric(pair, a_output, b_output)

    result = {
        'a_id': pair['a_id'], 'b_id': pair['b_id'],
        'dependency': pair.get('dependency', 'unknown'),
        'task_desc': pair['task_desc'][:100],
        'quality_level': quality_level,
        'trial': trial_idx,
        'score': score,
        'used_upstream': used,
        'criterion_scores': criterion_scores,
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
    dep_counts = Counter(p.get('dependency') for p in pairs)
    print(f"Dependency distribution: {dict(dep_counts)}")

    total = len(pairs) * len(QUALITY_LEVELS) * N_TRIALS
    print(f"Total evaluations: {len(pairs)} × {len(QUALITY_LEVELS)} × {N_TRIALS} = {total}")
    print(f"Model: {MODEL}\n")

    done = set()
    if os.path.exists(RAW_OUT):
        with open(RAW_OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['a_id'], r['b_id'], r['quality_level'], r['trial']))
                except Exception:
                    pass
    if done:
        print(f"Resuming: {len(done)} trials already done, skipping them.")
    else:
        open(RAW_OUT, 'w').close()

    jobs = [(pair, ql, t)
            for pair in pairs
            for ql in QUALITY_LEVELS
            for t in range(N_TRIALS)
            if (pair['a_id'], pair['b_id'], ql, t) not in done]

    print(f"Remaining jobs: {len(jobs)}")

    all_results = []
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = {executor.submit(evaluate_trial, p, q, t): (p['a_id'], q, t)
                   for p, q, t in jobs}
        for future in tqdm(as_completed(futures), total=len(jobs), desc="Evaluating"):
            res = future.result()
            if res:
                all_results.append(res)

    print("\n=== Results by Dependency × Quality ===")
    import statistics
    dep_types = ['tight', 'loose', 'independent']
    ql_order = ['perfect', 'degraded', 'noisy', 'wrong']
    for dep in dep_types:
        scores_by_ql = {}
        for ql in ql_order:
            rows = [r['score'] for r in all_results if r['dependency'] == dep and r['quality_level'] == ql]
            scores_by_ql[ql] = statistics.mean(rows) if rows else 0
        drop = scores_by_ql['perfect'] - scores_by_ql['wrong']
        print(f"  {dep:12s}: perfect={scores_by_ql['perfect']:.3f}  degraded={scores_by_ql['degraded']:.3f}  noisy={scores_by_ql['noisy']:.3f}  wrong={scores_by_ql['wrong']:.3f}  drop={drop:+.3f}")

if __name__ == '__main__':
    main()
