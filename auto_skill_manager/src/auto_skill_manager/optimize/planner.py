from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Iterable

from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.optimization import (
    ActionImpact,
    LibraryDiffResult,
    OptimizationAction,
    OptimizationPlan,
    OptimizationResult,
    SkillDelta,
)
from auto_skill_manager.schema.scorecard import LibraryScoreCard, SkillScoreCard
from auto_skill_manager.schema.skill import SkillRecord

if TYPE_CHECKING:
    from auto_skill_manager.analyze.library import AnalysisResult


def plan_from_analysis(
    result: AnalysisResult,
    *,
    only_action_types: Iterable[str] | None = None,
    skill_id: str | None = None,
) -> OptimizationPlan:
    allowed_types = {item for item in (only_action_types or []) if item}
    actions: list[OptimizationAction] = []
    for recommendation in result.recommendations:
        action_subtype = recommendation.suggested_action.get("kind", recommendation.type) if recommendation.suggested_action else recommendation.type
        normalized_action_type = "rewrite" if action_subtype in {"rewrite", "narrow", "boundary_rewrite"} else recommendation.type
        target_skill_ids = list(recommendation.entities.get("skill_ids", []))
        if allowed_types and normalized_action_type not in allowed_types and action_subtype not in allowed_types:
            continue
        if skill_id and skill_id not in target_skill_ids:
            continue
        actions.append(
            OptimizationAction(
                action_id=recommendation.id,
                action_type=normalized_action_type,
                action_subtype=action_subtype if action_subtype != normalized_action_type else "",
                target_skill_ids=target_skill_ids,
                source_recommendation_id=recommendation.id,
                proposed_changes=dict(
                    recommendation.suggested_action.get("proposed_changes")
                    or _default_proposed_changes(recommendation.type, target_skill_ids)
                ),
                status="pending",
                notes=recommendation.summary,
            )
        )
    metadata = {
        "recommendation_count": len(result.recommendations),
        "action_count": len(actions),
    }
    if allowed_types:
        metadata["only_action_types"] = sorted(allowed_types)
    if skill_id:
        metadata["skill_id"] = skill_id
    created_at = datetime.now(UTC)
    return OptimizationPlan(
        plan_id=created_at.strftime("opt-%Y%m%d-%H%M%S-%f"),
        library_id=result.library_scorecard.library_id,
        created_at=created_at.isoformat(),
        actions=actions,
        metadata=metadata,
    )


def diff_libraries(before: AnalysisResult, after: AnalysisResult) -> LibraryDiffResult:
    before_cards = {card.skill_id: card for card in before.skill_scorecards}
    after_cards = {card.skill_id: card for card in after.skill_scorecards}
    skill_ids = sorted(set(before_cards) | set(after_cards))
    skill_deltas = [
        _build_skill_delta(skill_id, before_cards.get(skill_id), after_cards.get(skill_id))
        for skill_id in skill_ids
    ]
    before_ids = set(before_cards)
    after_ids = set(after_cards)
    return LibraryDiffResult(
        base_library_id=before.library_scorecard.library_id,
        candidate_library_id=after.library_scorecard.library_id,
        summary_delta=_build_summary_delta(before.library_scorecard, after.library_scorecard),
        skill_deltas=skill_deltas,
        added_skill_ids=sorted(after_ids - before_ids),
        removed_skill_ids=sorted(before_ids - after_ids),
    )
def apply_plan(library: LibraryRecord, plan: OptimizationPlan) -> LibraryRecord:
    skills = [deepcopy(skill) for skill in library.skills]
    for action in plan.actions:
        if action.status != "applied":
            continue
        if action.action_type == "rewrite":
            _apply_rewrite(skills, action)
        elif action.action_type == "merge":
            skills = _apply_merge(skills, action)
        elif action.action_type == "remove":
            skills = _apply_remove(skills, action)
    return LibraryRecord(
        library_id=f"{library.library_id}:optimized:{plan.plan_id}",
        skills=skills,
        pipeline_edges=list(library.pipeline_edges),
        metadata={**library.metadata, "optimization_plan_id": plan.plan_id},
    )


def diff_results(before: AnalysisResult, after: AnalysisResult, plan: OptimizationPlan) -> OptimizationResult:
    before_cards = {card.skill_id: card for card in before.skill_scorecards}
    after_cards = {card.skill_id: card for card in after.skill_scorecards}
    skill_ids = sorted(set(before_cards) | set(after_cards))
    skill_deltas = [
        _build_skill_delta(skill_id, before_cards.get(skill_id), after_cards.get(skill_id))
        for skill_id in skill_ids
    ]
    summary_delta = _build_summary_delta(before.library_scorecard, after.library_scorecard)
    return OptimizationResult(
        plan_id=plan.plan_id,
        library_id_before=before.library_scorecard.library_id,
        library_id_after=after.library_scorecard.library_id,
        actions_applied=[action.action_id for action in plan.actions if action.status == "applied"],
        actions_skipped=[action.action_id for action in plan.actions if action.status != "applied"],
        scorecard_before=before.library_scorecard,
        scorecard_after=after.library_scorecard,
        skill_deltas=skill_deltas,
        summary_delta=summary_delta,
        action_impacts=_build_action_impacts(plan, skill_deltas, summary_delta),
    )


def _default_proposed_changes(action_type: str, skill_ids: list[str]) -> dict:
    if action_type == "rewrite":
        return {"description": "", "anchors": {"verbs": [], "objects": [], "constraints": []}}
    if action_type == "merge":
        return {"merged_into": skill_ids[0] if skill_ids else "", "drop_ids": skill_ids[1:]}
    return {}


def _apply_rewrite(skills: list[SkillRecord], action: OptimizationAction) -> None:
    target_ids = set(action.target_skill_ids)
    per_skill_changes = action.proposed_changes.get("skills") or {}
    for skill in skills:
        if skill.id not in target_ids:
            continue
        proposed_changes = per_skill_changes.get(skill.id) or action.proposed_changes
        if "description" in proposed_changes and proposed_changes["description"]:
            skill.description = str(proposed_changes["description"])
        anchors = proposed_changes.get("anchors") or {}
        if anchors:
            skill.anchors = {
                "verbs": list(anchors.get("verbs", skill.anchors.get("verbs", []))),
                "objects": list(anchors.get("objects", skill.anchors.get("objects", []))),
                "constraints": list(anchors.get("constraints", skill.anchors.get("constraints", []))),
            }


def _apply_merge(skills: list[SkillRecord], action: OptimizationAction) -> list[SkillRecord]:
    target_id = str(action.proposed_changes.get("merged_into") or (action.target_skill_ids[0] if action.target_skill_ids else ""))
    drop_ids = set(action.proposed_changes.get("drop_ids") or action.target_skill_ids[1:])
    if not target_id:
        return skills
    target_skill = next((skill for skill in skills if skill.id == target_id), None)
    if target_skill is None:
        return skills
    merged_skills = [skill for skill in skills if skill.id in drop_ids]
    for skill in merged_skills:
        target_skill.examples = list(dict.fromkeys([*target_skill.examples, *skill.examples]))
        target_skill.tags = list(dict.fromkeys([*target_skill.tags, *skill.tags]))
        target_skill.anchors = {
            "verbs": list(dict.fromkeys([*target_skill.anchors.get("verbs", []), *skill.anchors.get("verbs", [])])),
            "objects": list(dict.fromkeys([*target_skill.anchors.get("objects", []), *skill.anchors.get("objects", [])])),
            "constraints": list(dict.fromkeys([*target_skill.anchors.get("constraints", []), *skill.anchors.get("constraints", [])])),
        }
    return [skill for skill in skills if skill.id not in drop_ids]


def _apply_remove(skills: list[SkillRecord], action: OptimizationAction) -> list[SkillRecord]:
    target_ids = set(action.target_skill_ids)
    return [skill for skill in skills if skill.id not in target_ids]


def _build_skill_delta(skill_id: str, before: SkillScoreCard | None, after: SkillScoreCard | None) -> SkillDelta:
    delta = SkillDelta(
        skill_id=skill_id,
        competition_score_before=before.competition_score if before else 0.0,
        competition_score_after=after.competition_score if after else 0.0,
        anchor_strength_before=before.anchor_strength_score if before else 0.0,
        anchor_strength_after=after.anchor_strength_score if after else 0.0,
        black_hole_risk_before=before.black_hole_risk if before else 0.0,
        black_hole_risk_after=after.black_hole_risk if after else 0.0,
        rewrite_priority_before=before.rewrite_priority if before else 0.0,
        rewrite_priority_after=after.rewrite_priority if after else 0.0,
    )
    delta.changed = any(
        abs(value) > 0.02
        for value in [
            delta.competition_score_after - delta.competition_score_before,
            delta.anchor_strength_after - delta.anchor_strength_before,
            delta.black_hole_risk_after - delta.black_hole_risk_before,
            delta.rewrite_priority_after - delta.rewrite_priority_before,
        ]
    )
    return delta


def _build_summary_delta(before: LibraryScoreCard, after: LibraryScoreCard) -> dict[str, dict[str, float | str]]:
    metrics = {
        "competition_density": (before.competition_density, after.competition_density, False),
        "danger_zone_mass": (before.danger_zone_mass, after.danger_zone_mass, False),
        "anchor_weakness_mass": (before.anchor_weakness_mass, after.anchor_weakness_mass, False),
        "predicted_routing_stability": (before.predicted_routing_stability, after.predicted_routing_stability, True),
        "predicted_black_hole_exposure": (before.predicted_black_hole_exposure, after.predicted_black_hole_exposure, False),
    }
    return {
        name: {
            "before": round(before_value, 3),
            "after": round(after_value, 3),
            "delta": round(after_value - before_value, 3),
            "direction": _direction(after_value - before_value, higher_is_better),
        }
        for name, (before_value, after_value, higher_is_better) in metrics.items()
    }


def _build_action_impacts(
    plan: OptimizationPlan,
    skill_deltas: list[SkillDelta],
    summary_delta: dict[str, dict[str, float | str]],
) -> list[ActionImpact]:
    changed_skill_ids = [item.skill_id for item in skill_deltas if item.changed]
    return [
        ActionImpact(
            action_id=action.action_id,
            action_type=action.action_type,
            action_subtype=action.action_subtype,
            status=action.status,
            target_skill_ids=list(action.target_skill_ids),
            changed_skill_ids=[skill_id for skill_id in changed_skill_ids if skill_id in action.target_skill_ids],
            metric_impacts={
                name: dict(item)
                for name, item in summary_delta.items()
                if action.status == "applied"
            },
            notes=action.notes,
        )
        for action in plan.actions
    ]


def _direction(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 0.001:
        return "neutral"
    improved = delta > 0 if higher_is_better else delta < 0
    return "improved" if improved else "degraded"
