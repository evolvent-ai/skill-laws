from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_skill_manager.analyze.library import AnalysisResult, AnalysisSummary, LibraryAnalyzer
from auto_skill_manager.ingest.loaders import load_library, load_skill, write_library
from auto_skill_manager.optimize import apply_plan, diff_libraries, plan_from_analysis
from auto_skill_manager.reporting.json_report import render_diff_json_report, render_json_report, render_optimization_json_report
from auto_skill_manager.reporting.markdown import render_library_diff_markdown_report, render_markdown_report, render_optimization_markdown_report
from auto_skill_manager.runtime import build_selection_payload, build_skill_context, check_required_outputs, infer_required_outputs
from auto_skill_manager.schema.optimization import OptimizationAction, OptimizationPlan
from auto_skill_manager.schema.scorecard import LibraryScoreCard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-skill-manager")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze a skill library")
    analyze_sub = analyze.add_subparsers(dest="target")
    analyze_library = analyze_sub.add_parser("library", help="Analyze one library file")
    analyze_library.add_argument("library_path", type=Path)

    inspect_cmd = subparsers.add_parser("inspect", help="Inspect one skill")
    inspect_sub = inspect_cmd.add_subparsers(dest="target")
    inspect_skill = inspect_sub.add_parser("skill", help="Inspect one skill in a library")
    inspect_skill.add_argument("library_path", type=Path)
    inspect_skill.add_argument("--id", dest="skill_id", required=True)
    inspect_skill.add_argument("--format", choices=["markdown", "json"], default="markdown")

    compare = subparsers.add_parser("compare", help="Compare a candidate skill")
    compare_sub = compare.add_subparsers(dest="target")
    compare_candidate = compare_sub.add_parser("candidate", help="Compare one candidate against a library")
    compare_candidate.add_argument("library_path", type=Path)
    compare_candidate.add_argument("--file", dest="candidate_path", type=Path, required=True)

    simulate = subparsers.add_parser("simulate", help="Simulate a library change")
    simulate_sub = simulate.add_subparsers(dest="target")
    simulate_add = simulate_sub.add_parser("add", help="Simulate adding one candidate skill")
    simulate_add.add_argument("library_path", type=Path)
    simulate_add.add_argument("--file", dest="candidate_path", type=Path, required=True)

    report = subparsers.add_parser("report", help="Export a library report")
    report_sub = report.add_subparsers(dest="target")
    report_library = report_sub.add_parser("library", help="Export one library report")
    report_library.add_argument("library_path", type=Path)
    report_library.add_argument("--format", choices=["markdown", "json"], default="markdown")

    diff = subparsers.add_parser("diff", help="Diff two skill libraries")
    diff_sub = diff.add_subparsers(dest="target")
    diff_library = diff_sub.add_parser("library", help="Diff two library files")
    diff_library.add_argument("base_library_path", type=Path)
    diff_library.add_argument("candidate_library_path", type=Path)
    diff_library.add_argument("--format", choices=["markdown", "json"], default="markdown")

    optimize = subparsers.add_parser("optimize", help="Plan or apply skill library optimization")
    optimize_sub = optimize.add_subparsers(dest="target")
    optimize_plan = optimize_sub.add_parser("plan", help="Generate an optimization plan")
    optimize_plan.add_argument("library_path", type=Path)
    optimize_plan.add_argument("--format", choices=["json", "markdown"], default="json")
    optimize_plan.add_argument("--only", choices=["rewrite", "narrow", "boundary_rewrite", "merge", "remove"], action="append")
    optimize_plan.add_argument("--skill", dest="skill_id")

    optimize_apply = optimize_sub.add_parser("apply", help="Apply an optimization plan")
    optimize_apply.add_argument("library_path", type=Path)
    optimize_apply.add_argument("--plan", dest="plan_path", type=Path, required=True)
    optimize_apply.add_argument("--write-library", dest="write_library_path", type=Path)
    optimize_apply.add_argument("--format", choices=["json", "markdown"], default="json")

    runtime = subparsers.add_parser("runtime", help="Build runtime skill context or check task closure")
    runtime_sub = runtime.add_subparsers(dest="target")
    runtime_context = runtime_sub.add_parser("context", help="Select skills and render runtime context")
    runtime_context.add_argument("library_path", type=Path)
    runtime_context.add_argument("--task-id", required=True)
    runtime_context.add_argument("--instruction", required=True)
    runtime_context.add_argument("--limit", type=int, default=250)
    runtime_context.add_argument("--mode", choices=["full", "domain-gated", "domain-profile"], default="domain-profile")
    runtime_context.add_argument("--format", choices=["text", "json"], default="text")

    runtime_check = runtime_sub.add_parser("check", help="Check whether required output artifacts exist")
    runtime_check.add_argument("--workspace", type=Path, required=True)
    runtime_check.add_argument("--task-id", required=True)
    runtime_check.add_argument("--instruction", default="")
    runtime_check.add_argument("--required-output", action="append", default=[])
    runtime_check.add_argument("--format", choices=["json", "markdown"], default="markdown")

    return parser


def build_inspect_result(result: AnalysisResult, skill_id: str) -> AnalysisResult:
    filtered_skill_cards = [card for card in result.skill_scorecards if card.skill_id == skill_id]
    if not filtered_skill_cards:
        raise ValueError(f"Skill not found: {skill_id}")

    filtered_pair_cards = [
        card for card in result.pair_scorecards
        if card.left_skill_id == skill_id or card.right_skill_id == skill_id
    ]
    filtered_recommendations = [
        rec for rec in result.recommendations
        if skill_id in rec.entities.get("skill_ids", [])
    ]
    summary = AnalysisSummary(
        top_risky_skill_ids=[skill_id],
        top_conflict_pairs=[
            {"left": card.left_skill_id, "right": card.right_skill_id}
            for card in filtered_pair_cards[:5]
        ],
        recommendation_counts={
            rec.type: sum(1 for item in filtered_recommendations if item.type == rec.type)
            for rec in filtered_recommendations
        },
        health_status=result.summary.health_status,
    )
    return AnalysisResult(
        library_scorecard=LibraryScoreCard(
            library_id=result.library_scorecard.library_id,
            size=result.library_scorecard.size,
            competition_density=result.library_scorecard.competition_density,
            danger_zone_mass=result.library_scorecard.danger_zone_mass,
            anchor_weakness_mass=result.library_scorecard.anchor_weakness_mass,
            predicted_routing_stability=result.library_scorecard.predicted_routing_stability,
            predicted_black_hole_exposure=result.library_scorecard.predicted_black_hole_exposure,
            pipeline_fragility_index=result.library_scorecard.pipeline_fragility_index,
            family_interference_index=result.library_scorecard.family_interference_index,
            details={**result.library_scorecard.details, "view": "inspect_skill", "focus_skill_id": skill_id},
        ),
        skill_scorecards=filtered_skill_cards,
        pair_scorecards=filtered_pair_cards[:10],
        recommendations=filtered_recommendations,
        change_impact=None,
        summary=summary,
    )


def render_optimization_plan(plan: OptimizationPlan, output_format: str) -> str:
    if output_format == "json":
        return json.dumps({
            "plan_id": plan.plan_id,
            "library_id": plan.library_id,
            "created_at": plan.created_at,
            "metadata": plan.metadata,
            "actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "action_subtype": action.action_subtype,
                    "target_skill_ids": action.target_skill_ids,
                    "source_recommendation_id": action.source_recommendation_id,
                    "proposed_changes": action.proposed_changes,
                    "status": action.status,
                    "notes": action.notes,
                }
                for action in plan.actions
            ],
        }, indent=2, ensure_ascii=False, sort_keys=True)
    lines = [
        f"# Optimization Plan: {plan.plan_id}",
        "",
        f"- Library: {plan.library_id}",
        f"- Created at: {plan.created_at}",
        f"- Recommendation count: {plan.metadata.get('recommendation_count', 0)}",
        f"- Action count: {plan.metadata.get('action_count', len(plan.actions))}",
        "",
        "## Actions",
    ]
    for action in plan.actions:
        lines.append(
            f"- `{action.action_id}`: type={action.action_type} subtype={action.action_subtype or '-'} "
            f"status={action.status} targets={action.target_skill_ids}"
        )
        if action.notes:
            lines.append(f"  - notes: {action.notes}")
    return "\n".join(lines)


def load_plan(path: Path) -> OptimizationPlan:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return OptimizationPlan(
        plan_id=str(raw.get("plan_id") or "manual-plan"),
        library_id=str(raw.get("library_id") or ""),
        created_at=str(raw.get("created_at") or ""),
        metadata=dict(raw.get("metadata") or {}),
        actions=[
            OptimizationAction(
                action_id=str(item.get("action_id") or ""),
                action_type=str(item.get("action_type") or "rewrite"),
                action_subtype=str(item.get("action_subtype") or ""),
                target_skill_ids=list(item.get("target_skill_ids") or []),
                source_recommendation_id=str(item.get("source_recommendation_id") or ""),
                proposed_changes=dict(item.get("proposed_changes") or {}),
                status=str(item.get("status") or "pending"),
                notes=str(item.get("notes") or ""),
            )
            for item in raw.get("actions", [])
        ],
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    analyzer = LibraryAnalyzer()

    if args.command == "analyze" and args.target == "library":
        result = analyzer.analyze_library(load_library(args.library_path))
        print(render_markdown_report(result))
        return 0

    if args.command == "inspect" and args.target == "skill":
        result = analyzer.analyze_library(load_library(args.library_path))
        try:
            inspect_result = build_inspect_result(result, args.skill_id)
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(render_json_report(inspect_result))
        else:
            print(render_markdown_report(inspect_result))
        return 0

    if args.command == "compare" and args.target == "candidate":
        library = load_library(args.library_path)
        candidate = load_skill(args.candidate_path)
        result = analyzer.analyze_library(library)
        result.change_impact = analyzer.compare_candidate(library, candidate)
        print(render_markdown_report(result))
        return 0

    if args.command == "simulate" and args.target == "add":
        library = load_library(args.library_path)
        candidate = load_skill(args.candidate_path)
        result = analyzer.simulate_add(library, candidate)
        print(render_markdown_report(result))
        return 0

    if args.command == "report" and args.target == "library":
        result = analyzer.analyze_library(load_library(args.library_path))
        if args.format == "json":
            print(render_json_report(result))
        else:
            print(render_markdown_report(result))
        return 0

    if args.command == "diff" and args.target == "library":
        base_result = analyzer.analyze_library(load_library(args.base_library_path))
        candidate_result = analyzer.analyze_library(load_library(args.candidate_library_path))
        diff_result = diff_libraries(base_result, candidate_result)
        if args.format == "json":
            print(render_diff_json_report(diff_result))
        else:
            print(render_library_diff_markdown_report(diff_result))
        return 0

    if args.command == "optimize" and args.target == "plan":
        result = analyzer.analyze_library(load_library(args.library_path))
        print(render_optimization_plan(
            plan_from_analysis(
                result,
                only_action_types=args.only,
                skill_id=args.skill_id,
            ),
            args.format,
        ))
        return 0

    if args.command == "optimize" and args.target == "apply":
        library = load_library(args.library_path)
        plan = load_plan(args.plan_path)
        if args.write_library_path:
            optimized_library = apply_plan(library, plan)
            write_library(optimized_library, args.write_library_path)
        result = analyzer.optimize(library, plan)
        if args.format == "json":
            print(render_optimization_json_report(result))
        else:
            print(render_optimization_markdown_report(result))
        return 0

    if args.command == "runtime" and args.target == "context":
        library = load_library(args.library_path)
        if args.format == "json":
            print(json.dumps(
                build_selection_payload(
                    library,
                    instruction=args.instruction,
                    task_id=args.task_id,
                    limit=args.limit,
                    mode=args.mode,
                ),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ))
        else:
            print(build_skill_context(
                library,
                instruction=args.instruction,
                task_id=args.task_id,
                limit=args.limit,
                mode=args.mode,
            ))
        return 0

    if args.command == "runtime" and args.target == "check":
        required = args.required_output or infer_required_outputs(args.task_id, args.instruction)
        check = check_required_outputs(args.workspace, required)
        payload = {
            "ok": check.ok,
            "workspace": str(args.workspace),
            "task_id": args.task_id,
            "expected": check.expected,
            "feedback": check.feedback(),
            "missing": check.missing,
            "empty": check.empty,
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            status = "ok" if check.ok else "failed"
            print(f"# Runtime Closure Check: {status}")
            print("")
            print(f"- task_id: `{args.task_id}`")
            print(f"- workspace: `{args.workspace}`")
            print(f"- expected: {', '.join(check.expected) or '(none inferred)'}")
            print(f"- missing: {', '.join(check.missing) or '(none)'}")
            print(f"- empty: {', '.join(check.empty) or '(none)'}")
            print(f"- feedback: {check.feedback()}")
        return 0

    parser.error("Unsupported command combination")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
