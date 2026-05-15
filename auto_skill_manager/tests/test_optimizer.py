from pathlib import Path
import unittest

from auto_skill_manager.analyze import LibraryAnalyzer
from auto_skill_manager.ingest import load_library, load_skill
from auto_skill_manager.optimize import diff_libraries, plan_from_analysis
from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.optimization import OptimizationAction, OptimizationPlan
from auto_skill_manager.schema.skill import SkillRecord


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "examples" / "library.sample.yaml"
CANDIDATE_PATH = ROOT / "examples" / "candidate.sample.yaml"


class OptimizerTests(unittest.TestCase):
    def test_plan_from_analysis_creates_actions(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        analysis = analyzer.analyze_library(library)

        plan = plan_from_analysis(analysis)
        self.assertEqual(plan.library_id, library.library_id)
        self.assertTrue(plan.actions)
        self.assertTrue(all(action.status == "pending" for action in plan.actions))

    def test_plan_includes_auto_rewrite_or_narrow_payload(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        analysis = analyzer.analyze_library(library)

        plan = plan_from_analysis(analysis, skill_id="generic_data_handler")
        action = next(item for item in plan.actions if item.target_skill_ids == ["generic_data_handler"])

        self.assertEqual(action.action_type, "rewrite")
        self.assertEqual(action.action_subtype, "")
        self.assertTrue(action.proposed_changes["description"])
        self.assertTrue(action.proposed_changes["anchors"]["verbs"])
        self.assertIn("do not use for vague catch-all requests", action.proposed_changes["anchors"]["constraints"])

    def test_plan_can_recommend_remove_for_unanchored_black_hole(self) -> None:
        analyzer = LibraryAnalyzer()
        library = LibraryRecord(
            library_id="black-hole-test",
            skills=[
                SkillRecord(
                    id="generic_everything",
                    name="Generic Everything",
                    description="Generic general handle process multiple workflow various data.",
                    family="general",
                    tags=["generic"],
                    anchors={"verbs": [], "objects": [], "constraints": []},
                ),
                SkillRecord(
                    id="specific_report_builder",
                    name="Specific Report Builder",
                    description="Create a monthly sales report from approved revenue tables.",
                    family="reports",
                    tags=["sales", "report"],
                    anchors={
                        "verbs": ["create"],
                        "objects": ["sales report", "revenue tables"],
                        "constraints": ["monthly reports only"],
                    },
                ),
            ],
        )

        plan = plan_from_analysis(analyzer.analyze_library(library))

        remove_actions = [item for item in plan.actions if item.action_type == "remove"]
        self.assertEqual([item.target_skill_ids for item in remove_actions], [["generic_everything"]])
        self.assertEqual(remove_actions[0].action_subtype, "")
        self.assertEqual(remove_actions[0].proposed_changes["routing_policy"], "remove-from-flat-pool")

    def test_plan_includes_pair_boundary_rewrite_payload(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        plan = plan_from_analysis(analyzer.analyze_library(library))

        pair_action = next(
            item for item in plan.actions
            if set(item.target_skill_ids) == {"parse_citation_pdf", "generate_bibliography_entry"}
        )

        self.assertEqual(pair_action.action_type, "rewrite")
        self.assertEqual(pair_action.action_subtype, "boundary_rewrite")
        self.assertEqual(pair_action.proposed_changes["pair"], ["parse_citation_pdf", "generate_bibliography_entry"])
        self.assertIn("parse_citation_pdf", pair_action.proposed_changes["skills"])
        self.assertIn("generate_bibliography_entry", pair_action.proposed_changes["skills"])
        left_constraints = pair_action.proposed_changes["skills"]["parse_citation_pdf"]["anchors"]["constraints"]
        right_constraints = pair_action.proposed_changes["skills"]["generate_bibliography_entry"]["anchors"]["constraints"]
        self.assertTrue(any("Generate Bibliography Entry" in item for item in left_constraints))
        self.assertTrue(any("Parse Citation PDF" in item for item in right_constraints))

    def test_optimize_pair_boundary_rewrite_updates_both_skills(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        plan = plan_from_analysis(analyzer.analyze_library(library))
        pair_action = next(
            item for item in plan.actions
            if set(item.target_skill_ids) == {"parse_citation_pdf", "generate_bibliography_entry"}
        )
        pair_action.status = "applied"

        result = analyzer.optimize(library, plan)

        changed = {item.skill_id for item in result.skill_deltas if item.changed}
        self.assertIn("parse_citation_pdf", changed)
        self.assertIn("generate_bibliography_entry", changed)
        self.assertIn(pair_action.action_id, result.actions_applied)
        impact = next(item for item in result.action_impacts if item.action_id == pair_action.action_id)
        self.assertEqual(impact.action_subtype, "boundary_rewrite")

    def test_plan_can_filter_by_action_subtype(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        plan = plan_from_analysis(analyzer.analyze_library(library), only_action_types=["boundary_rewrite"])

        self.assertTrue(plan.actions)
        self.assertTrue(all(action.action_subtype == "boundary_rewrite" for action in plan.actions))
        self.assertEqual(plan.metadata["only_action_types"], ["boundary_rewrite"])

    def test_optimize_with_pending_actions_keeps_metrics(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        analysis = analyzer.analyze_library(library)
        plan = plan_from_analysis(analysis)

        result = analyzer.optimize(library, plan)
        self.assertEqual(result.actions_applied, [])
        self.assertEqual(result.scorecard_before.competition_density, result.scorecard_after.competition_density)
        self.assertTrue(all(not item.changed for item in result.skill_deltas))
        self.assertTrue(all(item.status == "pending" for item in result.action_impacts))

    def test_optimize_rewrite_changes_skill_metrics(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        plan = OptimizationPlan(
            plan_id="opt-test",
            library_id=library.library_id,
            created_at="2026-04-20T00:00:00+00:00",
            actions=[
                OptimizationAction(
                    action_id="rewrite-generic",
                    action_type="rewrite",
                    target_skill_ids=["generic_data_handler"],
                    status="applied",
                    proposed_changes={
                        "description": "Transform structured tabular input into normalized records for analytics workflows.",
                        "anchors": {
                            "verbs": ["transform", "normalize"],
                            "objects": ["tabular input", "records", "analytics workflows"],
                            "constraints": ["structured data only"],
                        },
                    },
                )
            ],
        )

        result = analyzer.optimize(library, plan)
        changed = {item.skill_id: item for item in result.skill_deltas if item.changed}
        self.assertIn("generic_data_handler", changed)
        self.assertEqual(result.actions_applied, ["rewrite-generic"])
        self.assertEqual(result.action_impacts[0].changed_skill_ids, ["generic_data_handler"])
        self.assertIn("predicted_routing_stability", result.action_impacts[0].metric_impacts)

    def test_compare_candidate_returns_change_impact(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        candidate = load_skill(CANDIDATE_PATH)
        impact = analyzer.compare_candidate(library, candidate)

        self.assertEqual(impact.candidate_skill_id, candidate.id)
        self.assertIn(impact.recommended_action, {"reject", "add-with-review", "approve-candidate"})
        self.assertIn("nearest_skill_id", impact.details)

    def test_simulate_add_changes_library_identity(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        candidate = load_skill(CANDIDATE_PATH)
        result = analyzer.simulate_add(library, candidate)

        self.assertTrue(result.library_scorecard.library_id.endswith(candidate.id))
        self.assertIsNotNone(result.change_impact)
        self.assertEqual(result.change_impact.details.get("mode"), "simulate_add")

    def test_diff_libraries_tracks_added_skill(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        candidate = load_skill(CANDIDATE_PATH)
        extended_library = LibraryRecord(
            library_id=f"{library.library_id}:candidate",
            skills=[*library.skills, candidate],
            pipeline_edges=list(library.pipeline_edges),
            metadata=dict(library.metadata),
        )

        diff = diff_libraries(analyzer.analyze_library(library), analyzer.analyze_library(extended_library))
        self.assertEqual(diff.base_library_id, library.library_id)
        self.assertEqual(diff.candidate_library_id, extended_library.library_id)
        self.assertIn(candidate.id, diff.added_skill_ids)
        self.assertFalse(diff.removed_skill_ids)
