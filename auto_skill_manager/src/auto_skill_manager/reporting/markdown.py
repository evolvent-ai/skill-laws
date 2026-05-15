from __future__ import annotations

from auto_skill_manager.analyze.library import AnalysisResult
from auto_skill_manager.schema.optimization import LibraryDiffResult, OptimizationResult


def render_markdown_report(result: AnalysisResult) -> str:
    lines: list[str] = []
    library = result.library_scorecard
    summary = result.summary
    is_skill_inspection = library.details.get("view") == "inspect_skill"
    focus_skill_id = library.details.get("focus_skill_id") if is_skill_inspection else None

    lines.append(f"# Skill Manager Report: {library.library_id}")
    lines.append("")
    if is_skill_inspection and focus_skill_id:
        lines.append(f"## Skill Focus")
        lines.append(f"- Skill: {focus_skill_id}")
        lines.append(f"- Related conflict pairs: {len(result.pair_scorecards)}")
        lines.append(f"- Related recommendations: {len(result.recommendations)}")
        lines.append("")
    lines.append("## Summary")
    lines.append(f"- Health status: {summary.health_status}")
    lines.append(f"- Top risky skills: {', '.join(summary.top_risky_skill_ids) if summary.top_risky_skill_ids else 'none'}")
    lines.append(f"- Recommendation counts: {summary.recommendation_counts or {}}")
    lines.append("")
    lines.append("## Library Scorecard")
    lines.append(f"- Size: {library.size}")
    lines.append(f"- Competition density: {library.competition_density}")
    lines.append(f"- Danger-zone mass: {library.danger_zone_mass}")
    lines.append(f"- Anchor-weakness mass: {library.anchor_weakness_mass}")
    lines.append(f"- Predicted routing stability: {library.predicted_routing_stability}")
    lines.append(f"- Predicted black-hole exposure: {library.predicted_black_hole_exposure}")
    lines.append(f"- Pipeline fragility index: {library.pipeline_fragility_index}")
    lines.append("")

    lines.append("## Top Risky Skills" if not is_skill_inspection else "## Skill Risk Card")
    for card in sorted(result.skill_scorecards, key=lambda item: item.rewrite_priority, reverse=True)[:5]:
        lines.append(
            f"- `{card.skill_id}`: rewrite_priority={card.rewrite_priority}, "
            f"black_hole_risk={card.black_hole_risk}, anchor_strength={card.anchor_strength_score}, "
            f"competition_score={card.competition_score}"
        )
    lines.append("")

    lines.append("## Top Conflict Pairs" if not is_skill_inspection else "## Related Conflict Pairs")
    for card in result.pair_scorecards[:5]:
        lines.append(
            f"- `{card.left_skill_id}` ↔ `{card.right_skill_id}`: "
            f"competition_risk={card.competition_risk}, overlap={card.overlap_score}, "
            f"direction={card.interference_direction}"
        )
    if not result.pair_scorecards:
        lines.append("- No related conflict pairs.")
    lines.append("")

    lines.append("## Recommendations")
    if not result.recommendations:
        lines.append("- No recommendations generated.")
    else:
        for rec in result.recommendations:
            lines.append(f"- [{rec.priority}] {rec.summary}")
            action = rec.suggested_action.get("kind") if rec.suggested_action else "review"
            lines.append(f"  - action: {action}")
            if rec.rationale:
                triggers = rec.rationale.get("score_triggers", {})
                if triggers:
                    lines.append(f"  - triggers: {triggers}")
    lines.append("")

    if result.change_impact is not None:
        impact = result.change_impact
        lines.append("## Candidate Change Impact")
        lines.append(f"- Candidate: {impact.candidate_skill_id}")
        lines.append(f"- Recommended action: {impact.recommended_action}")
        lines.append(f"- Δ competition density: {impact.delta_competition_density}")
        lines.append(f"- Δ danger-zone mass: {impact.delta_danger_zone_mass}")
        lines.append(f"- Δ anchor conflict: {impact.delta_anchor_conflict}")
        lines.append(f"- Δ routing stability: {impact.delta_routing_stability}")
        if impact.details:
            lines.append(f"- Details: {impact.details}")
        lines.append("")

    return "\n".join(lines)


def render_library_diff_markdown_report(result: LibraryDiffResult) -> str:
    lines: list[str] = []
    lines.append(f"# Skill Manager Diff: {result.base_library_id} → {result.candidate_library_id}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Base library: {result.base_library_id}")
    lines.append(f"- Candidate library: {result.candidate_library_id}")
    lines.append(f"- Added skills: {result.added_skill_ids or []}")
    lines.append(f"- Removed skills: {result.removed_skill_ids or []}")
    lines.append("")
    lines.append("## Metric Deltas")
    for name, item in result.summary_delta.items():
        lines.append(
            f"- {name}: before={item.get('before')} after={item.get('after')} "
            f"delta={item.get('delta')} direction={item.get('direction')}"
        )
    lines.append("")
    lines.append("## Skill Deltas")
    changed = [
        item for item in result.skill_deltas
        if item.changed or item.skill_id in result.added_skill_ids or item.skill_id in result.removed_skill_ids
    ]
    if not changed:
        lines.append("- No skill-level changes detected.")
    else:
        for item in changed:
            status = "added" if item.skill_id in result.added_skill_ids else "removed" if item.skill_id in result.removed_skill_ids else "changed"
            lines.append(
                f"- `{item.skill_id}` [{status}]: competition Δ={round(item.competition_score_after - item.competition_score_before, 3)}, "
                f"anchor Δ={round(item.anchor_strength_after - item.anchor_strength_before, 3)}, "
                f"black-hole Δ={round(item.black_hole_risk_after - item.black_hole_risk_before, 3)}, "
                f"rewrite Δ={round(item.rewrite_priority_after - item.rewrite_priority_before, 3)}"
            )
    lines.append("")
    return "\n".join(lines)


def render_optimization_markdown_report(result: OptimizationResult) -> str:
    lines: list[str] = []
    lines.append(f"# Skill Manager Optimization: {result.library_id_before}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Plan: {result.plan_id}")
    lines.append(f"- Library before: {result.library_id_before}")
    lines.append(f"- Library after: {result.library_id_after}")
    lines.append(f"- Actions applied: {len(result.actions_applied)}")
    lines.append(f"- Actions skipped: {len(result.actions_skipped)}")
    lines.append("")

    lines.append("## Metric Deltas")
    for name, item in result.summary_delta.items():
        lines.append(
            f"- {name}: before={item.get('before')} after={item.get('after')} "
            f"delta={item.get('delta')} direction={item.get('direction')}"
        )
    lines.append("")

    lines.append("## Skill Deltas")
    changed = [item for item in result.skill_deltas if item.changed]
    if not changed:
        lines.append("- No skill-level metric changes detected.")
    else:
        for item in changed:
            lines.append(
                f"- `{item.skill_id}`: competition Δ={round(item.competition_score_after - item.competition_score_before, 3)}, "
                f"anchor Δ={round(item.anchor_strength_after - item.anchor_strength_before, 3)}, "
                f"black-hole Δ={round(item.black_hole_risk_after - item.black_hole_risk_before, 3)}, "
                f"rewrite Δ={round(item.rewrite_priority_after - item.rewrite_priority_before, 3)}"
            )
    lines.append("")

    lines.append("## Action Status")
    lines.append(f"- Applied: {result.actions_applied or []}")
    lines.append(f"- Skipped: {result.actions_skipped or []}")
    lines.append("")
    return "\n".join(lines)
