from pathlib import Path
import unittest

from auto_skill_manager.analyze import LibraryAnalyzer
from auto_skill_manager.ingest import load_library, load_skill


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "examples" / "library.sample.yaml"
CANDIDATE_PATH = ROOT / "examples" / "candidate.sample.yaml"


class AnalyzerTests(unittest.TestCase):
    def test_analyze_library_returns_recommendations(self) -> None:
        analyzer = LibraryAnalyzer()
        library = load_library(LIBRARY_PATH)
        result = analyzer.analyze_library(library)

        recommendation_types = {item.type for item in result.recommendations}
        self.assertEqual(result.library_scorecard.library_id, "sample_skill_library")
        self.assertIn(result.summary.health_status, {"healthy", "watch", "risky"})
        self.assertIn("rewrite", recommendation_types)

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
