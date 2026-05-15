from __future__ import annotations

import math

from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.scorecard import PairScoreCard, SkillScoreCard
from auto_skill_manager.schema.skill import SkillRecord
from auto_skill_manager.scoring.anchors import abstraction_score, anchor_strength


def mean(values) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0


def skill_scorecard(
    skill: SkillRecord,
    library: LibraryRecord,
    pair_index: dict[str, list[PairScoreCard]],
    *,
    beta: float = 20.0,
) -> SkillScoreCard:
    anchors = skill.anchors or {}
    verbs = anchors.get("verbs", [])
    objects = anchors.get("objects", [])
    constraints = anchors.get("constraints", [])
    score_abstraction = abstraction_score(skill.description)
    score_anchor_strength = anchor_strength(skill)
    neighbor_pairs = pair_index.get(skill.id, [])
    top_overlap = max((pair.semantic_similarity for pair in neighbor_pairs), default=0.0)
    mean_topk_similarity = mean(
        pair.semantic_similarity for pair in sorted(neighbor_pairs, key=lambda item: item.semantic_similarity, reverse=True)[:3]
    )

    ci = sum(
        pair.details.get("boltzmann_raw", math.exp(beta * pair.semantic_similarity))
        for pair in neighbor_pairs
    )
    self_affinity = math.exp(beta * score_anchor_strength)
    competition_score = min(1.0, ci / (self_affinity + ci)) if ci > 0 else 0.0

    family_size = sum(1 for other in library.skills if skill.family is not None and other.family == skill.family and other.id != skill.id)
    family_conflict = min(1.0, family_size / max(1, len(library.skills) - 1))
    anchor_weakness = max(0.0, 1.0 - score_anchor_strength)
    black_hole_risk = min(
        1.0,
        0.35 * competition_score + 0.35 * anchor_weakness + 0.2 * score_abstraction + 0.1 * family_conflict,
    )
    routing_fragility = min(1.0, 0.5 * competition_score + 0.35 * anchor_weakness + 0.15 * score_abstraction)
    rewrite_priority = min(1.0, 0.45 * black_hole_risk + 0.25 * score_abstraction + 0.2 * anchor_weakness + 0.1 * top_overlap)
    return SkillScoreCard(
        skill_id=skill.id,
        competition_score=round(competition_score, 3),
        anchor_strength_score=round(score_anchor_strength, 3),
        anchor_weakness_score=round(anchor_weakness, 3),
        abstraction_score=round(score_abstraction, 3),
        black_hole_risk=round(black_hole_risk, 3),
        family_conflict_score=round(family_conflict, 3),
        routing_fragility_score=round(routing_fragility, 3),
        rewrite_priority=round(rewrite_priority, 3),
        details={
            "family": skill.family,
            "family_size": family_size,
            "verbs": verbs,
            "objects": objects,
            "constraints": constraints,
            "top_overlap": round(top_overlap, 3),
            "mean_topk_similarity": round(mean_topk_similarity, 3),
            "competition_index_raw": round(ci, 3),
            "self_affinity": round(self_affinity, 3),
        },
    )
