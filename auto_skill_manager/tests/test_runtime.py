from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_skill_manager.ingest.loaders import load_library
from auto_skill_manager.runtime import build_selection_payload, build_skill_context, check_required_outputs, infer_required_outputs, select_skills


ROOT = Path(__file__).resolve().parent.parent


class RuntimeTests(unittest.TestCase):
    def test_domain_gated_context_selects_matching_skills(self) -> None:
        library = load_library(ROOT / "examples" / "library.sample.yaml")
        selected = select_skills(
            library,
            instruction="Convert citation metadata into a bibliography entry.",
            task_id="doc-001",
            limit=250,
            mode="domain-profile",
        )
        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual(selected[0].skill.id, "generate_bibliography_entry")
        self.assertLessEqual(len(selected), 60)

    def test_context_renders_selection_metadata(self) -> None:
        library = load_library(ROOT / "examples" / "library.sample.yaml")
        context = build_skill_context(
            library,
            instruction="Summarize this PDF.",
            task_id="doc-002",
            mode="domain-profile",
        )
        self.assertIn("selection_mode: domain-profile", context)
        self.assertIn("estimated_context_chars:", context)
        self.assertIn("selected_tool_count:", context)
        self.assertIn("summarize_pdf", context)

    def test_selection_payload_includes_reasons(self) -> None:
        library = load_library(ROOT / "examples" / "library.sample.yaml")
        payload = build_selection_payload(
            library,
            instruction="Create a BibTeX entry from citation metadata.",
            task_id="doc-001",
            mode="domain-profile",
        )
        self.assertEqual(payload["mode"], "domain-profile")
        self.assertGreaterEqual(payload["selected_tool_count"], 1)
        self.assertIn("reason", payload["skills"][0])

    def test_infer_required_outputs_uses_explicit_and_instruction_patterns(self) -> None:
        self.assertEqual(infer_required_outputs("doc-002", ""), ["output.html"])
        self.assertEqual(infer_required_outputs("db-002", ""), ["query.sql", "results.json"])
        self.assertEqual(
            infer_required_outputs("custom-001", "Save the answer as `results.json`."),
            ["results.json"],
        )

    def test_check_required_outputs_reports_missing_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "empty.json").write_text("")
            check = check_required_outputs(workspace, ["missing.json", "empty.json"])
            self.assertFalse(check.ok)
            self.assertEqual(check.missing, ["missing.json"])
            self.assertEqual(check.empty, ["empty.json"])
            self.assertIn("Missing required output files", check.feedback())


if __name__ == "__main__":
    unittest.main()
