# Skill-Law Experiments

This directory contains the F01-F12 experiment entry scripts used by the Auto Skill Manager paper artifact.

The scripts are intentionally organized as small, inspectable Python programs rather than a monolithic experiment framework. Each finding has a dedicated directory with one or two public entry points:

- `run_*.py`: generate raw or intermediate experiment outputs.
- `analyze_*.py`: aggregate generated outputs into tables, JSON summaries, or figures.

All generated outputs go under `skill_law/data/`, which is ignored by git.

## Findings

| Finding | Entry Directory | Entry Scripts |
| --- | --- | --- |
| F01 | `F01_logarithmic_decay` | `run_f01_route_decay_sweep.py`, `run_f01_cross_model_scaling_probe.py` |
| F02 | `F02_pipeline_compounding` | `run_f02_transition_cascade_audit.py`, `analyze_f02_cascade_penalty_decomposition.py` |
| F03 | `F03_rebound_effect` | `run_f03_step_position_rebound_probe.py`, `analyze_f03_position_profile_rebound.py` |
| F04 | `F04_description_quality` | `run_f04_controlled_description_quality.py`, `analyze_f04_controlled_description_quality.py` |
| F05 | `F05_local_competition` | `run_f05_similarity_competition_stress.py`, `analyze_f05_local_competition_index.py` |
| F06 | `F06_failure_geometry` | `run_f06_structured_boundary_rewrite.py`, `analyze_f06_route_asymmetry_cross_model.py` |
| F07 | `F07_anchor_removal_black_hole` | `run_f07_query_anchor_ablation.py`, `run_f07_real_skills_anchor_stress_n100.py` |
| F08 | `F08_dual_trigger_protocol` | `run_f08_dual_trigger_hijack_validation.py`, `run_f08_protocol_ablation.py` |
| F09 | `F09_routing_independence` | `analyze_f09_clustered_routing_independence.py`, `analyze_f09_mixed_effects_independence.py` |
| F10 | `F10_execution_rescue` | `run_f10_conditional_transfer_rescue.py`, `run_f10_mass_combo_rescue_probe.py` |
| F11 | `F11_context_recovery` | `run_f11_quality_propagation_v4.py`, `run_f11_self_repair_baselines.py` |
| F12 | `F12_strong_tow_crowding` | `run_f12_strong_tow_product_baseline.py`, `analyze_f12_strong_tow_crowding.py` |

## Setup

From `open_source/`:

```bash
python3 -m pip install -e skill_law
```

For no-install usage:

```bash
export PYTHONPATH="$PWD/skill_law/src"
```

Several scripts call LLM APIs. Configure credentials through environment variables or a local `.env` file:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

The `.env` file is ignored by git.

## Data Root

By default, generated outputs are written to:

```text
skill_law/data/
```

To keep data outside the repository:

```bash
export SKILL_LAW_DATA_ROOT=/path/to/local/skill_law_data
```

The runtime helper package resolves paths through `skill_law.paths`, so scripts should not need absolute local paths.

## Smoke Test

Run all public entry scripts once with small settings:

```bash
PYTHONPATH=skill_law/src PYTHONDONTWRITEBYTECODE=1 \
python3 skill_law/run_all_skill_law_scripts.py --timeout 240
```

Expected result:

```text
SUMMARY passed=24 failed=0 total=24
```

This runner is a code-path and artifact-layout check. It uses small sample sizes and may produce diagnostic summaries marked `valid_for_analysis: false` when the sample is too small for the full statistical analysis.

## Running Individual Findings

Example F01 smoke run:

```bash
PYTHONPATH=skill_law/src \
python3 skill_law/F01_logarithmic_decay/run_f01_route_decay_sweep.py \
  --model gpt-5.4-mini --demo --limit-cases 5
```

Example F02 run plus analysis:

```bash
PYTHONPATH=skill_law/src \
python3 skill_law/F02_pipeline_compounding/run_f02_transition_cascade_audit.py \
  --model gpt-5.4-mini --limit-per-k 1 --k-values 4 --workers 1

PYTHONPATH=skill_law/src \
python3 skill_law/F02_pipeline_compounding/analyze_f02_cascade_penalty_decomposition.py
```

Use larger sample sizes and the paper's model set for full reproduction.

## Output Convention

New outputs use the F01-F12 naming scheme:

```text
skill_law/data/F01/...
skill_law/data/F02/...
...
skill_law/data/F12/...
```

Figures shared across analyses are written under:

```text
skill_law/data/figures/exports/
```

## Notes for Reproduction

- Some analyses require generated raw outputs from the corresponding `run_*.py` script.
- Some scripts support small smoke-test settings through environment variables used by `run_all_skill_law_scripts.py`.
- Full statistical reproduction should use the complete benchmark inputs and model set described in the paper.
- The open-source package does not ship private data, local caches, or generated results.
