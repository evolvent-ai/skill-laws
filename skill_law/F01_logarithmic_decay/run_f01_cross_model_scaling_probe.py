import os
import json
import numpy as np
from tqdm import tqdm
import sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from skill_law.runtime import TaskSpec
from skill_law.runtime import LLMRouter
from skill_law.runtime import LibrarySpec, SkillSpec
from skill_law.paths import SKILLS_DIR, data_path, finding_path

print("Scaling law of tool capacity")
print("="*60)


def _parse_int_arg(name: str, default: int) -> int:
    args = sys.argv[1:]
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            return int(args[idx + 1])
    return default


def _parse_model(default: str | None = None) -> str | None:
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def _has_flag(name: str) -> bool:
    return name in sys.argv[1:]

with (data_path("processed", "high_similarity_skills.json")).open() as f:
    skills_data = json.load(f)

all_skills = [
    SkillSpec(id=s["name"], name=s["name"], description=s["description"],
              input_schema={}, output_schema={})
    for s in skills_data
]
base_skills_data = [s for s in skills_data if "variant_of" not in s]
base_name_map = {s["name"]: s["base_name"] for s in skills_data}

N_values = [6, 12, 24, 48, 72, 96]
n_samples = 30
if _has_flag("--demo"):
    N_values = [6]
    n_samples = _parse_int_arg("--limit-cases", 5)
rng = np.random.default_rng(42)

MODELS = ["gpt-3.5-turbo", "gpt-5.4-mini", "gpt-4o"]
selected_model = _parse_model()
if selected_model:
    MODELS = [selected_model]

def fuzzy_instruction(skill_id):
    base = base_name_map.get(skill_id, skill_id)
    feature = base.split(":")[-1] 
    return f"I need to use the tool for {feature}"

def run_single_trial(router, N, trial_id):
    base_idx = rng.integers(0, len(base_skills_data))
    base_skill_data = base_skills_data[base_idx]
    base_name = base_skill_data["name"]
    
    skill_group = [s for s in all_skills if s.id == base_name or
                   any(sd["name"] == s.id and sd.get("base_name") == base_skill_data["base_name"]
                       for sd in skills_data)]
    
    if len(skill_group) < 1:
        return 0

    target = rng.choice(skill_group)
    target_id = target.id
    library_skills = [target]
    
    others = [s for s in all_skills if s.id != target_id]
    n_other = min(N - 1, len(others))
    if n_other > 0:
        library_skills += list(rng.choice(others, n_other, replace=False))

    library = LibrarySpec(id=f"lib_{N}_{trial_id}", skills=library_skills)
    instruction = fuzzy_instruction(target_id)
    task = TaskSpec(id=f"task_{trial_id}", instruction=instruction,
                   required_skills=[target_id], gold_trace=[])
                   
    chosen, _ = router.route(task, library)
    return 1 if chosen == target_id else 0

all_results = {"N": N_values, "models": {}}

for model_name in MODELS:
    print(f"\nTesting model: {model_name}")
    router = LLMRouter(model_name=model_name)
    acc_list = []
    
    for N in N_values:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_single_trial, router, N, i) for i in range(n_samples)]
            outcomes = []
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except Exception as e:
                    outcomes.append(0)
                    
        acc = np.mean(outcomes) if outcomes else 0.0
        acc_list.append(acc)
        print(f"  N={N:2d} -> accuracy: {acc:.1%}")
        
    all_results["models"][model_name] = acc_list

out_path = finding_path("F01", "data", "archive", "scaling_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nCross-model test complete. Results saved to: {out_path}")
