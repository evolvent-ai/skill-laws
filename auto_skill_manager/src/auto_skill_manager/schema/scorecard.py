from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillScoreCard:
    skill_id: str
    competition_score: float = 0.0
    anchor_strength_score: float = 0.0
    anchor_weakness_score: float = 0.0
    abstraction_score: float = 0.0
    black_hole_risk: float = 0.0
    family_conflict_score: float = 0.0
    routing_fragility_score: float = 0.0
    rewrite_priority: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairScoreCard:
    left_skill_id: str
    right_skill_id: str
    semantic_similarity: float = 0.0
    overlap_score: float = 0.0
    competition_risk: float = 0.0
    interference_direction: str = "symmetric"
    merge_candidate_score: float = 0.0
    split_signal: float = 0.0
    strong_tow_potential: float = 0.0
    weak_drag_risk: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LibraryScoreCard:
    library_id: str
    size: int = 0
    competition_density: float = 0.0
    family_interference_index: float = 0.0
    danger_zone_mass: float = 0.0
    anchor_weakness_mass: float = 0.0
    predicted_routing_stability: float = 0.0
    predicted_black_hole_exposure: float = 0.0
    pipeline_fragility_index: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChangeImpactReport:
    candidate_skill_id: str | None = None
    delta_competition_density: float = 0.0
    delta_danger_zone_mass: float = 0.0
    delta_anchor_conflict: float = 0.0
    delta_routing_stability: float = 0.0
    recommended_action: str = "review"
    details: dict[str, Any] = field(default_factory=dict)
