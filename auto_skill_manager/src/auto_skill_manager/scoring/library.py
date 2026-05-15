from __future__ import annotations

import math

from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.scorecard import LibraryScoreCard, PairScoreCard, SkillScoreCard


def mean(values) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0


def build_pair_index(pair_cards: list[PairScoreCard]) -> dict[str, list[PairScoreCard]]:
    pair_index: dict[str, list[PairScoreCard]] = {}
    for card in pair_cards:
        pair_index.setdefault(card.left_skill_id, []).append(card)
        pair_index.setdefault(card.right_skill_id, []).append(card)
    return pair_index


def boltzmann_accuracy(self_affinity: float, competition_index: float) -> float:
    denom = self_affinity + competition_index
    return self_affinity / denom if denom > 0 else 1.0


def predicted_accuracy_at_size(n: int, a: float = 0.834, b: float = 0.042) -> float:
    return max(0.0, a - b * math.log(max(1, n)))


def rescue_potential(acc_upstream: float, acc_downstream: float) -> float:
    return (1.0 - acc_downstream) * acc_upstream


def pipeline_fragility_index(
    library: LibraryRecord,
    *,
    skill_accuracies: dict[str, float] | None = None,
) -> float:
    if not library.pipeline_edges:
        return 0.0
    dep_weights = {
        "tight": 1.0,
        "loose": 0.4,
        "independent": 0.1,
    }
    total = 0.0
    for edge in library.pipeline_edges:
        dep_weight = dep_weights.get(edge.dependency_type, 0.3) * edge.weight
        if skill_accuracies:
            acc_up = skill_accuracies.get(edge.upstream_skill, 0.5)
            acc_down = skill_accuracies.get(edge.downstream_skill, 0.5)
            rescue = rescue_potential(acc_up, acc_down)
            dep_weight *= (1.0 - rescue)
        total += dep_weight
    return min(1.0, total / len(library.pipeline_edges))


def library_scorecard(
    library: LibraryRecord,
    skill_cards: list[SkillScoreCard],
    pair_cards: list[PairScoreCard],
    *,
    beta: float = 20.0,
) -> LibraryScoreCard:
    size = len(library.skills)
    competition_density = mean(card.competition_risk for card in pair_cards)
    danger_zone_mass = mean(card.competition_score for card in skill_cards if 0.45 <= card.competition_score <= 0.8)
    anchor_weakness_mass = mean(card.anchor_weakness_score for card in skill_cards)
    black_hole_exposure = mean(card.black_hole_risk for card in skill_cards)
    family_interference = mean(card.family_conflict_score for card in skill_cards)

    pair_idx = build_pair_index(pair_cards)
    skill_accuracies: dict[str, float] = {}
    for card in skill_cards:
        neighbors = pair_idx.get(card.skill_id, [])
        ci = sum(p.details.get("boltzmann_raw", math.exp(beta * p.semantic_similarity)) for p in neighbors)
        self_aff = math.exp(beta * card.anchor_strength_score)
        skill_accuracies[card.skill_id] = boltzmann_accuracy(self_aff, ci)

    routing_stability = mean(skill_accuracies.values()) if skill_accuracies else 1.0
    pipeline_fragility = pipeline_fragility_index(library, skill_accuracies=skill_accuracies)

    return LibraryScoreCard(
        library_id=library.library_id,
        size=size,
        competition_density=round(competition_density, 3),
        family_interference_index=round(family_interference, 3),
        danger_zone_mass=round(danger_zone_mass, 3),
        anchor_weakness_mass=round(anchor_weakness_mass, 3),
        predicted_routing_stability=round(routing_stability, 3),
        predicted_black_hole_exposure=round(black_hole_exposure, 3),
        pipeline_fragility_index=round(pipeline_fragility, 3),
        details={
            "edge_count": len(library.pipeline_edges),
            "log_law_predicted_accuracy": round(predicted_accuracy_at_size(size), 3),
            "skill_accuracies": {k: round(v, 3) for k, v in skill_accuracies.items()},
        },
    )
