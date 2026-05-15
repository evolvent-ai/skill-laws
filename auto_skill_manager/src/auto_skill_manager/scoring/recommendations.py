from __future__ import annotations

from dataclasses import dataclass, field

from auto_skill_manager.schema.recommendation import RecommendationRecord
from auto_skill_manager.schema.scorecard import LibraryScoreCard, PairScoreCard, SkillScoreCard
from auto_skill_manager.schema.skill import SkillRecord


@dataclass(slots=True)
class AnalysisSummaryRecord:
    top_risky_skill_ids: list[str] = field(default_factory=list)
    top_conflict_pairs: list[dict[str, str]] = field(default_factory=list)
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    health_status: str = "healthy"


def recommendation_priority_rank(priority: str) -> int:
    return {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }.get(priority, 0)


def recommendation_records(
    skill_cards: list[SkillScoreCard],
    pair_cards: list[PairScoreCard],
    skills: list[SkillRecord] | None = None,
) -> list[RecommendationRecord]:
    recommendations: list[RecommendationRecord] = []
    skills_by_id = {skill.id: skill for skill in skills or []}
    for card in skill_cards:
        should_manage = (
            card.rewrite_priority >= 0.32
            and (card.abstraction_score >= 0.45 or card.black_hole_risk >= 0.22 or card.anchor_weakness_score >= 0.18)
        )
        if should_manage:
            skill = skills_by_id.get(card.skill_id)
            action = _skill_management_action(card, skill)
            recommendations.append(
                RecommendationRecord(
                    id=f"{action['kind']}:{card.skill_id}",
                    type=action["kind"],
                    priority="critical" if action["kind"] == "remove" else "high" if card.rewrite_priority >= 0.5 else "medium",
                    confidence=round(max(card.rewrite_priority, card.black_hole_risk), 3),
                    entities={"skill_ids": [card.skill_id]},
                    summary=action["summary"],
                    rationale={
                        "observed_signals": ["high rewrite priority", "anchor weakness or abstraction"],
                        "score_triggers": {
                            "rewrite_priority": card.rewrite_priority,
                            "black_hole_risk": card.black_hole_risk,
                            "competition_score": card.competition_score,
                            "abstraction_score": card.abstraction_score,
                            "anchor_weakness_score": card.anchor_weakness_score,
                        },
                        "mapped_principles": ["anchor strength", "attractor risk"],
                    },
                    suggested_action=action,
                )
            )
    for card in pair_cards[:8]:
        should_review_overlap = (
            card.merge_candidate_score >= 0.22
            and (card.competition_risk >= 0.22 or card.overlap_score >= 0.2 or card.details.get("same_family", False))
        )
        if should_review_overlap:
            pair_action = _pair_management_action(card, skills_by_id)
            recommendations.append(
                RecommendationRecord(
                    id=f"overlap-review:{card.left_skill_id}:{card.right_skill_id}",
                    type=pair_action["kind"],
                    priority="medium",
                    confidence=round(max(card.merge_candidate_score, card.competition_risk), 3),
                    entities={"skill_ids": [card.left_skill_id, card.right_skill_id]},
                    summary=pair_action["summary"],
                    rationale={
                        "observed_signals": ["high pair overlap", "high competition risk"],
                        "score_triggers": {
                            "competition_risk": card.competition_risk,
                            "merge_candidate_score": card.merge_candidate_score,
                            "overlap_score": card.overlap_score,
                        },
                        "mapped_principles": ["local competition", "family interference"],
                    },
                    suggested_action=pair_action,
                )
            )
    recommendations.sort(key=lambda item: (recommendation_priority_rank(item.priority), item.confidence), reverse=True)
    return recommendations


def _skill_management_action(card: SkillScoreCard, skill: SkillRecord | None) -> dict:
    if _should_remove(card, skill):
        return {
            "kind": "remove",
            "notes": "Remove this broad catch-all skill from the flat routing pool unless it has a protected owner or explicit runtime-only role.",
            "summary": f"Remove black-hole skill {card.skill_id} from broad routing",
            "proposed_changes": {
                "reason": "high abstraction and black-hole risk with weak anchors",
                "routing_policy": "remove-from-flat-pool",
            },
        }
    if _should_narrow(card):
        rewrite = _propose_rewrite(card, skill, mode="narrow")
        return {
            "kind": "narrow",
            "notes": "Narrow this skill so it no longer absorbs vague or cross-family tasks.",
            "summary": f"Narrow broad skill {card.skill_id}",
            "proposed_changes": rewrite,
        }
    rewrite = _propose_rewrite(card, skill, mode="rewrite")
    return {
        "kind": "rewrite",
        "notes": "Apply a concrete boundary rewrite with stronger verbs, objects, and scope exclusions.",
        "summary": f"Rewrite boundaries for {card.skill_id}",
        "proposed_changes": rewrite,
    }


def _pair_management_action(card: PairScoreCard, skills_by_id: dict[str, SkillRecord]) -> dict:
    if card.details.get("same_family", False) and card.overlap_score >= 0.32:
        return {
            "kind": "merge",
            "notes": "Merge likely redundant same-family skills, or edit this plan if the pair must stay separate.",
            "summary": f"Merge or consolidate {card.left_skill_id} and {card.right_skill_id}",
            "proposed_changes": {
                "merged_into": card.left_skill_id,
                "drop_ids": [card.right_skill_id],
            },
        }
    left = skills_by_id.get(card.left_skill_id)
    right = skills_by_id.get(card.right_skill_id)
    return {
        "kind": "boundary_rewrite",
        "notes": "Apply reciprocal boundary clauses so each skill states when not to route to the paired neighbor.",
        "summary": f"Rewrite boundary between {card.left_skill_id} and {card.right_skill_id}",
        "proposed_changes": {
            "skills": {
                card.left_skill_id: _boundary_payload(left, right),
                card.right_skill_id: _boundary_payload(right, left),
            },
            "pair": [card.left_skill_id, card.right_skill_id],
        },
    }


def _boundary_payload(skill: SkillRecord | None, neighbor: SkillRecord | None) -> dict:
    if skill is None:
        return {}
    neighbor_name = neighbor.name if neighbor else "the paired neighbor"
    neighbor_objects = ", ".join((neighbor.anchors.get("objects", []) if neighbor else [])[:3]) or "that neighbor's target objects"
    anchors = skill.anchors or {}
    constraints = list(dict.fromkeys([
        *_clean_terms(anchors.get("constraints", [])),
        f"do not use when the request is specifically for {neighbor_name}",
        f"exclude tasks centered on {neighbor_objects}",
    ]))
    rewrite = _propose_rewrite_for_skill(skill, extra_constraints=constraints)
    rewrite["boundary_neighbor_id"] = neighbor.id if neighbor else None
    return rewrite


def _should_remove(card: SkillScoreCard, skill: SkillRecord | None) -> bool:
    if skill and skill.metadata.get("protected"):
        return False
    return card.black_hole_risk >= 0.55 and card.abstraction_score >= 0.5 and card.anchor_strength_score < 0.2


def _should_narrow(card: SkillScoreCard) -> bool:
    return card.black_hole_risk >= 0.45 or (card.abstraction_score >= 0.5 and card.anchor_weakness_score >= 0.3)


def _propose_rewrite(card: SkillScoreCard, skill: SkillRecord | None, *, mode: str) -> dict:
    name = skill.name if skill else card.skill_id.replace("_", " ").title()
    if skill:
        extra_constraints = ["prefer neighboring specialist skills when their anchors match"] if mode == "rewrite" else []
        return _propose_rewrite_for_skill(skill, extra_constraints=extra_constraints)
    return _fallback_rewrite(name)


def _propose_rewrite_for_skill(skill: SkillRecord, *, extra_constraints: list[str] | None = None) -> dict:
    name = skill.name
    anchors = skill.anchors if skill else {}
    verbs = _clean_terms(anchors.get("verbs", []))
    objects = _clean_terms(anchors.get("objects", []))
    constraints = _clean_terms(anchors.get("constraints", []))
    tags = _clean_terms(skill.tags)
    family = skill.family if skill.family else "its explicit domain"

    proposed_verbs = _concrete_terms(verbs, fallback=["validate", "transform"])
    proposed_objects = _concrete_terms(objects or tags, fallback=[name.lower()])
    proposed_constraints = list(dict.fromkeys([
        *constraints,
        f"only for {family} tasks",
        "do not use for vague catch-all requests",
        *(extra_constraints or []),
    ]))

    verb_phrase = " and ".join(proposed_verbs[:2])
    object_phrase = ", ".join(proposed_objects[:3])
    constraint_phrase = "; ".join(proposed_constraints[:2])
    description = f"{name} should {verb_phrase} {object_phrase} for {family} workflows. Scope: {constraint_phrase}."
    return {
        "description": description,
        "anchors": {
            "verbs": proposed_verbs[:4],
            "objects": proposed_objects[:5],
            "constraints": proposed_constraints[:5],
        },
    }


def _fallback_rewrite(name: str) -> dict:
    return {
        "description": f"{name} should validate and transform {name.lower()} for its explicit domain. Scope: do not use for vague catch-all requests.",
        "anchors": {
            "verbs": ["validate", "transform"],
            "objects": [name.lower()],
            "constraints": ["do not use for vague catch-all requests"],
        },
    }


def _clean_terms(values: list[str]) -> list[str]:
    return [item.strip() for item in values if item and item.strip()]


def _concrete_terms(values: list[str], *, fallback: list[str]) -> list[str]:
    banned = {"generic", "general", "handle", "process", "multiple", "workflow", "various", "utility", "input", "data"}
    concrete = [item for item in values if item.lower() not in banned]
    return list(dict.fromkeys(concrete or fallback))


def build_summary(
    skill_cards: list[SkillScoreCard],
    pair_cards: list[PairScoreCard],
    recommendations: list[RecommendationRecord],
    library_card: LibraryScoreCard,
) -> AnalysisSummaryRecord:
    top_risky = [card.skill_id for card in sorted(skill_cards, key=lambda item: item.rewrite_priority, reverse=True)[:5]]
    top_pairs = [
        {"left": card.left_skill_id, "right": card.right_skill_id}
        for card in pair_cards[:5]
    ]
    recommendation_counts: dict[str, int] = {}
    for recommendation in recommendations:
        recommendation_counts[recommendation.type] = recommendation_counts.get(recommendation.type, 0) + 1
    if library_card.predicted_routing_stability >= 0.8:
        health_status = "healthy"
    elif library_card.predicted_routing_stability >= 0.6:
        health_status = "watch"
    else:
        health_status = "risky"
    return AnalysisSummaryRecord(
        top_risky_skill_ids=top_risky,
        top_conflict_pairs=top_pairs,
        recommendation_counts=recommendation_counts,
        health_status=health_status,
    )
