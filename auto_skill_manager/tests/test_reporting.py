from pathlib import Path
import unittest

from auto_skill_manager.analyze import LibraryAnalyzer
from auto_skill_manager.ingest import load_library, load_skill
from auto_skill_manager.optimize import diff_libraries, plan_from_analysis
from auto_skill_manager.reporting.json_report import build_report_payload, render_diff_json_report, render_optimization_json_report
from auto_skill_manager.schema.optimization import OptimizationAction, OptimizationPlan


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "examples" / "library.sample.yaml"
CANDIDATE_PATH = ROOT / "examples" / "candidate.sample.yaml"


class ReportingTests(unittest.TestCase):
    def test_analysis_report_payload_contains_frontend_views(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        result = analyzer.analyze_library(library)

        payload = build_report_payload(result)
        self.assertEqual(payload["report_type"], "auto_skill_manager_analysis")
        self.assertIn("risk_table", payload["views"])
        self.assertIn("conflict_queue", payload["views"])
        self.assertIn("recommendation_queue", payload["views"])
        self.assertIsNone(payload["views"]["skill_focus"])

    def test_analysis_recommendation_queue_exposes_decision_ready_fields(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        result = analyzer.analyze_library(library)

        payload = build_report_payload(result)
        self.assertTrue(payload["views"]["recommendation_queue"])
        recommendation = payload["views"]["recommendation_queue"][0]

        self.assertIn("id", recommendation)
        self.assertIn("summary", recommendation)
        self.assertIn("skill_ids", recommendation)
        self.assertIn("mapped_action", recommendation)
        self.assertNotIn("decision_status", recommendation)
        self.assertNotIn("decision_updated_at", recommendation)

    def test_diff_report_json_marks_added_skill_status(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        candidate = load_skill(CANDIDATE_PATH)
        extended_library = library.__class__(
            library_id=f"{library.library_id}:candidate",
            skills=[*library.skills, candidate],
            pipeline_edges=list(library.pipeline_edges),
            metadata=dict(library.metadata),
        )

        diff = diff_libraries(analyzer.analyze_library(library), analyzer.analyze_library(extended_library))
        payload = __import__("json").loads(render_diff_json_report(diff))
        added_row = next(item for item in payload["views"]["skill_delta_table"] if item["skill_id"] == candidate.id)

        self.assertEqual(payload["report_type"], "auto_skill_manager_library_diff")
        self.assertEqual(added_row["status"], "added")
        self.assertEqual(payload["highlights"]["added_skill_count"], 1)

    def test_optimization_report_json_contains_action_summary_and_metric_cards(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        plan = OptimizationPlan(
            plan_id="opt-report-test",
            library_id=library.library_id,
            created_at="2026-04-21T00:00:00+00:00",
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
        payload = __import__("json").loads(render_optimization_json_report(result))

        self.assertEqual(payload["report_type"], "auto_skill_manager_optimization")
        self.assertTrue(payload["highlights"]["is_optimization"])
        self.assertIn("action_summary", payload["views"])
        self.assertIn("action_impact_cards", payload["views"])
        self.assertIn("metric_delta_cards", payload["views"])
        self.assertEqual(payload["views"]["action_summary"]["applied"], ["rewrite-generic"])
        self.assertTrue(payload["views"]["metric_delta_cards"])
        self.assertIn("action_subtype", payload["views"]["action_impact_cards"][0])

    def test_plan_from_analysis_filter_metadata_survives_reporting_workflow(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        analysis = analyzer.analyze_library(library)

        plan = plan_from_analysis(analysis, only_action_types=["rewrite"], skill_id="generic_data_handler")

        self.assertEqual(plan.metadata["only_action_types"], ["rewrite"])
        self.assertEqual(plan.metadata["skill_id"], "generic_data_handler")
        self.assertTrue(all(action.action_type == "rewrite" for action in plan.actions))
        self.assertTrue(all("generic_data_handler" in action.target_skill_ids for action in plan.actions))


if __name__ == "__main__":
    unittest.main()
