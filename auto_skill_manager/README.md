# auto-skill-manager

Theory-driven toolkit for diagnosing, comparing, optimizing, and runtime-gating large skill libraries.

- package: `auto_skill_manager`
- CLI: `auto-skill-manager`
- distribution: `auto-skill-manager`

## First run in 60 seconds

From `open_source/auto_skill_manager/`:

```bash
python3 -m pip install -e .
auto-skill-manager report library examples/library.sample.yaml --format json > report.json
open viewer/index.html
```

Then select `report.json` in the viewer.

If you want the shortest no-install path instead:

```bash
cd open_source/auto_skill_manager
python3 -m auto_skill_manager.cli.main report library examples/library.sample.yaml --format json > report.json
open viewer/index.html
```

## What it does

`auto-skill-manager` helps you:
- analyze a skill library,
- inspect one risky skill,
- compare a candidate skill against an existing library,
- simulate adding a candidate,
- generate an optimization plan,
- apply an edited optimization plan and review the before/after diff in the static viewer,
- build runtime skill context with domain-gated selection,
- check whether a task has actually produced its required output artifacts before the agent finishes.

## Requirements

- Python 3.11+
- `pip`

## Install

From the repository root:

```bash
python3 -m pip install -e .
```

Or from `open_source/auto_skill_manager/`:

```bash
python3 -m pip install -e .
```

That installs the `auto-skill-manager` CLI from `pyproject.toml`.

If you do not want to install it globally, you can also run commands with Python directly:

```bash
cd open_source/auto_skill_manager
python3 -m auto_skill_manager.cli.main --help
```

## Quick start

From `open_source/auto_skill_manager/`:

```bash
auto-skill-manager analyze library examples/library.sample.yaml
auto-skill-manager inspect skill examples/library.sample.yaml --id generic_data_handler
auto-skill-manager report library examples/library.sample.yaml --format json
```

## Common workflows

### Quick diagnostics

```bash
auto-skill-manager analyze library examples/library.sample.yaml
auto-skill-manager inspect skill examples/library.sample.yaml --id generic_data_handler
```

### Generate a viewer report

```bash
auto-skill-manager report library examples/library.sample.yaml --format json > report.json
open viewer/index.html
```

### Run the optimization loop

```bash
auto-skill-manager optimize plan examples/library.sample.yaml --format json > plan.json
# edit plan.json
auto-skill-manager optimize apply examples/library.sample.yaml --plan plan.json --write-library optimized-library.yaml --format json > optimization-report.json
open viewer/index.html
```

### Filter optimization plan generation

```bash
auto-skill-manager optimize plan examples/library.sample.yaml --only rewrite --format json
auto-skill-manager optimize plan examples/library.sample.yaml --only boundary_rewrite --format json
auto-skill-manager optimize plan examples/library.sample.yaml --skill generic_data_handler --format json
```

Use this to scaffold a narrower plan before editing it.

### Diff two libraries

```bash
auto-skill-manager diff library examples/library.sample.yaml examples/library.sample.yaml
```

Use this to compare two library files and review metric / skill deltas in markdown.

### Build runtime context for an agent

```bash
auto-skill-manager runtime context examples/library.sample.yaml \
  --task-id doc-002 \
  --instruction "Convert document.md to output.html" \
  --mode domain-gated \
  --limit 250
```

`domain-profile` is the recommended v2 runtime mode. It keeps broad tool context
for routing-heavy tasks, uses smaller positive-match context for local artifact
tasks, and adds lightweight domain priors to down-rank unrelated tool/vendor
matches.

Use `--mode domain-gated` to reproduce the v1 runtime policy, or `--mode full`
when you want the older top-k behavior for comparison.

JSON selection output:

```bash
auto-skill-manager runtime context examples/library.sample.yaml \
  --task-id doc-002 \
  --instruction "Convert document.md to output.html" \
  --mode domain-profile \
  --format json
```

### Check artifact closure before finishing

```bash
auto-skill-manager runtime check \
  --workspace /path/to/workspace \
  --task-id doc-002 \
  --instruction "Convert document.md to output.html"
```

The closure checker infers common required outputs from task text and includes
explicit guards for known structured tasks such as `doc-002`, `db-004`,
`debug-001`, and `xdom-012`. It exits successfully as a CLI command and reports
`ok: false` in JSON/markdown when files are missing or empty; host agents should
use that signal to continue rather than declare completion.

## Command reference

### Analyze a library

```bash
auto-skill-manager analyze library examples/library.sample.yaml
```

Outputs a markdown diagnostic report for the full library.

### Inspect one skill

Markdown:

```bash
auto-skill-manager inspect skill examples/library.sample.yaml --id generic_data_handler
```

JSON:

```bash
auto-skill-manager inspect skill examples/library.sample.yaml --id generic_data_handler --format json
```

Use this when you want a focused view of one skill, its risk signals, and related conflict pairs.

### Compare a candidate skill

```bash
auto-skill-manager compare candidate examples/library.sample.yaml --file examples/candidate.sample.yaml
```

Use this to see whether a new skill overlaps too much with the existing library.

### Simulate adding a candidate

```bash
auto-skill-manager simulate add examples/library.sample.yaml --file examples/candidate.sample.yaml
```

Use this to preview the predicted library impact before actually adding the candidate.

### Export a full report

Markdown:

```bash
auto-skill-manager report library examples/library.sample.yaml --format markdown
```

JSON:

```bash
auto-skill-manager report library examples/library.sample.yaml --format json
```

### Build runtime context

```bash
auto-skill-manager runtime context examples/library.sample.yaml \
  --task-id custom-001 \
  --instruction "Create a BibTeX entry from citation metadata." \
  --mode domain-gated
```

### Check runtime closure

Markdown:

```bash
auto-skill-manager runtime check \
  --workspace /path/to/workspace \
  --task-id custom-001 \
  --instruction "Save the answer as `results.json`."
```

JSON:

```bash
auto-skill-manager runtime check \
  --workspace /path/to/workspace \
  --task-id custom-001 \
  --instruction "Save the answer as `results.json`." \
  --format json
```

Use `--format json` when loading the result into the static viewer.

## Optimization workflow

### 1. Generate an optimization plan

JSON plan:

```bash
auto-skill-manager optimize plan examples/library.sample.yaml --format json > plan.json
```

Filtered JSON plan:

```bash
auto-skill-manager optimize plan examples/library.sample.yaml --only rewrite --format json
auto-skill-manager optimize plan examples/library.sample.yaml --only boundary_rewrite --format json
auto-skill-manager optimize plan examples/library.sample.yaml --skill generic_data_handler --format json
```

Markdown summary:

```bash
auto-skill-manager optimize plan examples/library.sample.yaml --format markdown
```

The generated plan is review-first. Actions start as `pending`.

Use `--only rewrite|narrow|boundary_rewrite|merge|remove` to scaffold a narrower plan, and `--skill <id>` to keep only actions tied to one skill. `action_type` is the executable operation; `action_subtype` preserves the manager intent for cases such as `narrow` and `boundary_rewrite`.

### 2. Edit the plan

Open `plan.json` and decide which actions should be applied.

Typical edits:
- change an action `status` from `pending` to `applied`,
- refine `proposed_changes.description`,
- refine `proposed_changes.anchors.verbs`,
- refine `proposed_changes.anchors.objects`,
- refine `proposed_changes.anchors.constraints`.

Example applied rewrite action:

```json
{
  "action_id": "rewrite-generic",
  "action_type": "rewrite",
  "action_subtype": "",
  "target_skill_ids": ["generic_data_handler"],
  "status": "applied",
  "proposed_changes": {
    "description": "Transform structured tabular input into normalized records for analytics workflows.",
    "anchors": {
      "verbs": ["transform", "normalize"],
      "objects": ["tabular input", "records", "analytics workflows"],
      "constraints": ["structured data only"]
    }
  }
}
```

### 3. Apply the plan

JSON optimization report:

```bash
auto-skill-manager optimize apply examples/library.sample.yaml --plan plan.json --format json > optimization-report.json
```

Write the optimized library YAML while also emitting the report:

```bash
auto-skill-manager optimize apply examples/library.sample.yaml --plan plan.json --write-library optimized-library.yaml --format json > optimization-report.json
```

Markdown optimization report:

```bash
auto-skill-manager optimize apply examples/library.sample.yaml --plan plan.json --format markdown
```

The result includes:
- before/after library metrics,
- per-skill deltas,
- applied/skipped actions,
- action impact cards for viewer drilldown.

### Diff two libraries

```bash
auto-skill-manager diff library examples/library.sample.yaml examples/library.sample.yaml
```

Use this to compare two library files and review added / removed skills plus library-level deltas.

## Open the static viewer

The viewer is a plain HTML file at:

```text
viewer/index.html
```

Open it in a browser directly, for example on macOS:

```bash
open viewer/index.html
```

Then load one of these JSON files from the file picker:
- `auto-skill-manager report library ... --format json`
- `auto-skill-manager inspect skill ... --format json`
- `auto-skill-manager optimize apply ... --format json`

## Viewer workflow

### Library / inspect reports

The viewer supports:
- risk table filtering,
- risk table sorting,
- skill drilldown,
- recommendation drilldown,
- pair drilldown,
- recommendation decisions (`accepted`, `dismissed`, `needs review`),
- decision badges and decision summary counts,
- filtered export,
- share mode for cleaner screenshots and async review,
- hash-based state persistence for current focus and recommendation decisions.

### Optimization reports

The viewer supports:
- clickable metric delta cards,
- clickable action impact cards,
- clickable applied/skipped action lists,
- clickable skill delta rows,
- focus summary for current metric/action/skill state,
- inline explanations for metric and skill changes,
- optimization snapshot export,
- share mode,
- hash-based state persistence.

Suggested workflow:
1. generate `optimization-report.json`,
2. open `viewer/index.html`,
3. load the JSON report,
4. click a metric card to narrow the action chain,
5. click an action to see linked skills,
6. click a skill row to inspect its inline explanation.

For library / inspect reports, you can also mark each recommendation as accepted, dismissed, or needs review. The viewer keeps those decisions in the current browser state, includes them in exported viewer snapshots, and restores them when a snapshot with decision state is reloaded.

## Run tests

From `open_source/auto_skill_manager/`:

```bash
python3 -m unittest discover tests
```

## Project layout

```text
open_source/auto_skill_manager/
├── examples/         sample library and candidate YAML files
├── src/              package source
├── tests/            unit tests
├── viewer/           static HTML viewer
├── PROJECT.md        design notes and product scope
└── README.md         usage guide
```

## Notes

- The viewer is static and reads JSON locally in the browser.
- The optimization flow is human-in-the-loop: generate plan, edit plan, then apply it.
- The recommended runtime mode is `domain-profile` plus closure checking.
- No benchmark result is bundled with this package. Run the target benchmark from a clean, documented checkout before reporting external numbers.
- `PROJECT.md` contains broader product/design context; this README is the practical usage guide.
