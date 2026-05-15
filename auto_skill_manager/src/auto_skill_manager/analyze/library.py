from __future__ import annotations

from dataclasses import dataclass, field

from auto_skill_manager.optimize import apply_plan, diff_results
from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.optimization import OptimizationPlan, OptimizationResult
from auto_skill_manager.schema.recommendation import RecommendationRecord
from auto_skill_manager.schema.scorecard import ChangeImpactReport, LibraryScoreCard, PairScoreCard, SkillScoreCard
from auto_skill_manager.schema.skill import SkillRecord
from auto_skill_manager.scoring import (
    abstraction_score,
    anchor_strength,
    build_pair_index,
    build_summary,
    library_scorecard,
    pair_scorecards,
    recommendation_records,
    skill_scorecard,
    skill_tokens,
)
from auto_skill_manager.scoring.recommendations import AnalysisSummaryRecord

AnalysisSummary = AnalysisSummaryRecord


@dataclass(slots=True)
class AnalysisResult:
    library_scorecard: LibraryScoreCard
    skill_scorecards: list[SkillScoreCard] = field(default_factory=list)
    pair_scorecards: list[PairScoreCard] = field(default_factory=list)
    recommendations: list[RecommendationRecord] = field(default_factory=list)
    change_impact: ChangeImpactReport | None = None
    summary: AnalysisSummaryRecord = field(default_factory=AnalysisSummaryRecord)


class LibraryAnalyzer:
    def __init__(self, max_pairs: int | None = None) -> None:
        self.max_pairs = max_pairs

    def analyze_library(self, library: LibraryRecord) -> AnalysisResult:
        pair_cards = pair_scorecards(library, max_pairs=self.max_pairs)
        pair_index = build_pair_index(pair_cards)
        skill_cards = [skill_scorecard(skill, library, pair_index) for skill in library.skills]
        library_card = library_scorecard(library, skill_cards, pair_cards)
        recommendations = recommendation_records(skill_cards, pair_cards, library.skills)
        summary = build_summary(skill_cards, pair_cards, recommendations, library_card)
        return AnalysisResult(
            library_scorecard=library_card,
            skill_scorecards=skill_cards,
            pair_scorecards=pair_cards,
            recommendations=recommendations,
            change_impact=None,
            summary=summary,
        )

    def optimize(self, library: LibraryRecord, plan: OptimizationPlan) -> OptimizationResult:
        before = self.analyze_library(library)
        optimized_library = apply_plan(library, plan)
        after = self.analyze_library(optimized_library)
        return diff_results(before, after, plan)

    def simulate_add(self, library: LibraryRecord, candidate: SkillRecord) -> AnalysisResult:
        before = self.analyze_library(library)
        projected_library = LibraryRecord(
            library_id=f"{library.library_id}:simulated_add:{candidate.id}",
            skills=[*library.skills, candidate],
            pipeline_edges=list(library.pipeline_edges),
            metadata={**library.metadata, "simulation": "add", "candidate_skill_id": candidate.id},
        )
        after = self.analyze_library(projected_library)
        candidate_impact = self.compare_candidate(library, candidate)
        before_card = before.library_scorecard
        after_card = after.library_scorecard
        after.change_impact = ChangeImpactReport(
            candidate_skill_id=candidate.id,
            delta_competition_density=round(after_card.competition_density - before_card.competition_density, 3),
            delta_danger_zone_mass=round(after_card.danger_zone_mass - before_card.danger_zone_mass, 3),
            delta_anchor_conflict=round(after_card.anchor_weakness_mass - before_card.anchor_weakness_mass, 3),
            delta_routing_stability=round(after_card.predicted_routing_stability - before_card.predicted_routing_stability, 3),
            recommended_action=candidate_impact.recommended_action,
            details={
                "mode": "simulate_add",
                "before_library_id": before_card.library_id,
                "after_library_id": after_card.library_id,
                "before": {
                    "competition_density": before_card.competition_density,
                    "danger_zone_mass": before_card.danger_zone_mass,
                    "anchor_weakness_mass": before_card.anchor_weakness_mass,
                    "predicted_routing_stability": before_card.predicted_routing_stability,
                },
                "after": {
                    "competition_density": after_card.competition_density,
                    "danger_zone_mass": after_card.danger_zone_mass,
                    "anchor_weakness_mass": after_card.anchor_weakness_mass,
                    "predicted_routing_stability": after_card.predicted_routing_stability,
                },
                "candidate_assessment": candidate_impact.details,
            },
        )
        return after

    def compare_candidate(self, library: LibraryRecord, candidate: SkillRecord) -> ChangeImpactReport:
        existing_ids = {skill.id for skill in library.skills}
        duplicate_id = candidate.id in existing_ids
        best_overlap = 0.0
        nearest_skill_id: str | None = None
        same_family_hits = 0
        candidate_terms = skill_tokens(candidate)

        for skill in library.skills:
            overlap = jaccard_similarity(skill_tokens(skill), candidate_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                nearest_skill_id = skill.id
            if skill.family and candidate.family and skill.family == candidate.family:
                same_family_hits += 1

        candidate_anchor_strength = anchor_strength(candidate)
        candidate_abstraction_score = abstraction_score(candidate.description)
        delta_competition_density = min(1.0, 0.02 + 0.55 * best_overlap + 0.06 * same_family_hits)
        delta_danger_zone_mass = min(1.0, 0.01 + 0.5 * best_overlap + 0.25 * candidate_abstraction_score)
        delta_anchor_conflict = min(1.0, 0.02 + 0.5 * (1.0 - candidate_anchor_strength) + 0.25 * candidate_abstraction_score)
        delta_routing_stability = -round(
            min(1.0, 0.02 + 0.4 * delta_competition_density + 0.35 * delta_anchor_conflict), 3
        )

        if duplicate_id or best_overlap >= 0.72:
            recommended_action = "reject"
        elif best_overlap >= 0.5 or candidate_anchor_strength < 0.45:
            recommended_action = "add-with-review"
        else:
            recommended_action = "approve-candidate"

        return ChangeImpactReport(
            candidate_skill_id=candidate.id,
            delta_competition_density=round(delta_competition_density, 3),
            delta_danger_zone_mass=round(delta_danger_zone_mass, 3),
            delta_anchor_conflict=round(delta_anchor_conflict, 3),
            delta_routing_stability=delta_routing_stability,
            recommended_action=recommended_action,
            details={
                "reason": "duplicate_id" if duplicate_id else "candidate_similarity_check",
                "nearest_skill_id": nearest_skill_id,
                "best_overlap": round(best_overlap, 3),
                "same_family_hits": same_family_hits,
                "candidate_anchor_strength": round(candidate_anchor_strength, 3),
                "candidate_abstraction_score": round(candidate_abstraction_score, 3),
            },
        )


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
