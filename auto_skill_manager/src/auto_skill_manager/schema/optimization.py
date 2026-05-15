from dataclasses import dataclass, field
from typing import Any

from .scorecard import LibraryScoreCard


@dataclass(slots=True)
class SkillDelta:
    skill_id: str
    competition_score_before: float = 0.0
    competition_score_after: float = 0.0
    anchor_strength_before: float = 0.0
    anchor_strength_after: float = 0.0
    black_hole_risk_before: float = 0.0
    black_hole_risk_after: float = 0.0
    rewrite_priority_before: float = 0.0
    rewrite_priority_after: float = 0.0
    changed: bool = False


@dataclass(slots=True)
class LibraryDiffResult:
    base_library_id: str
    candidate_library_id: str
    summary_delta: dict[str, dict[str, Any]] = field(default_factory=dict)
    skill_deltas: list[SkillDelta] = field(default_factory=list)
    added_skill_ids: list[str] = field(default_factory=list)
    removed_skill_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OptimizationAction:
    action_id: str
    action_type: str
    action_subtype: str = ""
    target_skill_ids: list[str] = field(default_factory=list)
    source_recommendation_id: str = ""
    proposed_changes: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    notes: str = ""


@dataclass(slots=True)
class ActionImpact:
    action_id: str
    action_type: str
    action_subtype: str = ""
    status: str = "pending"
    target_skill_ids: list[str] = field(default_factory=list)
    changed_skill_ids: list[str] = field(default_factory=list)
    metric_impacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class OptimizationPlan:
    plan_id: str
    library_id: str
    created_at: str
    actions: list[OptimizationAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationResult:
    plan_id: str
    library_id_before: str
    library_id_after: str
    actions_applied: list[str] = field(default_factory=list)
    actions_skipped: list[str] = field(default_factory=list)
    scorecard_before: LibraryScoreCard | None = None
    scorecard_after: LibraryScoreCard | None = None
    skill_deltas: list[SkillDelta] = field(default_factory=list)
    summary_delta: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_impacts: list[ActionImpact] = field(default_factory=list)
