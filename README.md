# Auto Skill Manager

Reference implementation and experiment code for studying automatic management of large skill libraries.

This repository contains two connected components:

- `auto_skill_manager`: an installable Python package and CLI for analyzing, optimizing, and runtime-gating skill libraries.
- `skill_law`: reproduction scripts for the F01-F12 empirical findings used to validate the routing and skill-library management principles.

The code is organized as a paper artifact: core implementation first, experiment entry points second, and generated data kept outside version control.

## Repository Layout

```text
.
├── auto_skill_manager/      # installable library, CLI, tests, examples, static viewer
├── skill_law/               # F01-F12 experiment scripts and shared runtime helpers
├── LICENSE
└── README.md
```

## Main Components

### Auto Skill Manager

`auto_skill_manager` provides the implementation-facing part of the project:

- library-level diagnostics for overlap, ambiguity, and routing risk
- single-skill inspection
- candidate-skill comparison and add-simulation
- optimization-plan generation and application
- runtime context selection with domain-aware gating
- artifact-closure checks for agent workflows
- a static viewer for JSON reports

See [`auto_skill_manager/README.md`](auto_skill_manager/README.md) for the full CLI guide.

### Skill-Law Experiments

`skill_law` contains one or two Python entry scripts for each empirical finding. The scripts share a small runtime package under `skill_law/src/skill_law` and write all generated artifacts under `skill_law/data/`.

| Finding | Directory | Purpose |
| --- | --- | --- |
| F01 | `F01_logarithmic_decay` | library-size scaling and single-step routing decay |
| F02 | `F02_pipeline_compounding` | transition-level compounding in multi-step routing |
| F03 | `F03_rebound_effect` | step-position rebound and profile-level penalties |
| F04 | `F04_description_quality` | controlled description-quality interventions |
| F05 | `F05_local_competition` | local competitor effects and similarity bands |
| F06 | `F06_failure_geometry` | descriptor-boundary failure geometry |
| F07 | `F07_anchor_removal_black_hole` | anchor-removal and real-skill stress tests |
| F08 | `F08_dual_trigger_protocol` | dual-trigger hijack and protocol ablations |
| F09 | `F09_routing_independence` | routing independence and mixed-effects diagnostics |
| F10 | `F10_execution_rescue` | conditional transfer and execution-rescue probes |
| F11 | `F11_context_recovery` | context recovery and self-repair baselines |
| F12 | `F12_strong_tow_crowding` | strong-tow crowding under a product baseline |

See [`skill_law/README.md`](skill_law/README.md) for reproduction commands.

## Installation

Python 3.11 or newer is recommended.

```bash
cd open_source
python3 -m pip install -e auto_skill_manager
python3 -m pip install -e skill_law
```

For development without installation, set `PYTHONPATH` explicitly:

```bash
export PYTHONPATH="$PWD/auto_skill_manager/src:$PWD/skill_law/src"
```

## Quick Verification

Run the package tests:

```bash
PYTHONPATH=auto_skill_manager/src python3 -m unittest discover -s auto_skill_manager/tests
```

Run every F01-F12 entry script once in small demo settings:

```bash
PYTHONPATH=skill_law/src PYTHONDONTWRITEBYTECODE=1 \
python3 skill_law/run_all_skill_law_scripts.py --timeout 240
```

The demo runner is designed as a smoke test for code paths and output structure. It uses very small sample sizes and should not be interpreted as the reported experimental result.

## Data and Generated Artifacts

This repository intentionally does not include private, local, or large experiment data.

Generated artifacts should be placed under:

```text
skill_law/data/
```

That directory is ignored by git. The scripts can also read from `SKILL_LAW_DATA_ROOT` if you want to keep generated data outside the repository:

```bash
export SKILL_LAW_DATA_ROOT=/path/to/local/skill_law_data
```

API keys should be provided through environment variables or a local `.env` file. `.env` is ignored by git.

## Reproducing Experiments

Each finding directory contains explicit `run_*.py` and, where needed, `analyze_*.py` entry scripts. A typical pattern is:

```bash
cd open_source
PYTHONPATH=skill_law/src python3 skill_law/F01_logarithmic_decay/run_f01_route_decay_sweep.py --model gpt-5.4-mini --demo --limit-cases 5
```

Then run the corresponding analysis script if one exists:

```bash
PYTHONPATH=skill_law/src python3 skill_law/F02_pipeline_compounding/analyze_f02_cascade_penalty_decomposition.py
```

For a full smoke-test pass:

```bash
PYTHONPATH=skill_law/src python3 skill_law/run_all_skill_law_scripts.py --timeout 240
```


## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Citation

If you use this repository in academic work, cite the paper artifact:

```bibtex
@misc{evolventai2026skilllaw,
  title        = {The Scaling Laws of Skills in LLM Agent Systems},
  author       = {{Evolvent AI Team} and Charles Chen and Qiming Yu and Yuhang Gu and Zhuoye Huang and Hanjing Li and Hongyu Liu and Jinhao Liu and Simin Liu and Dengyun Peng and Jiangyi Wang and Zheng Yan and Fanqing Meng and Ethan Qin and Carl Che and Mengkang Hu},
  year         = {2026},
  howpublished = {\url{https://evolvent.co/en/research}},
  note         = {Code: \url{https://github.com/evolvent-ai/auto-skill-manager}}
}
```
