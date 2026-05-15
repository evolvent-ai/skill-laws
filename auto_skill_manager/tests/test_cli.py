import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "examples" / "library.sample.yaml"
CANDIDATE_PATH = ROOT / "examples" / "candidate.sample.yaml"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "auto_skill_manager.cli.main",
        *args,
    ]
    return subprocess.run(command, capture_output=True, text=True, check=True)


class CliTests(unittest.TestCase):
    def test_analyze_command_outputs_report(self) -> None:
        result = run_cli("analyze", "library", str(LIBRARY_PATH))
        self.assertIn("Skill Manager Report", result.stdout)
        self.assertIn("Recommendations", result.stdout)

    def test_inspect_command_outputs_skill_focus(self) -> None:
        result = run_cli("inspect", "skill", str(LIBRARY_PATH), "--id", "generic_data_handler")
        self.assertIn("## Skill Focus", result.stdout)
        self.assertIn("Skill: generic_data_handler", result.stdout)
        self.assertIn("## Skill Risk Card", result.stdout)
        self.assertIn("## Related Conflict Pairs", result.stdout)

    def test_inspect_command_json_contains_skill_focus(self) -> None:
        result = run_cli("inspect", "skill", str(LIBRARY_PATH), "--id", "generic_data_handler", "--format", "json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["report_type"], "auto_skill_manager_skill_inspection")
        self.assertTrue(payload["highlights"]["is_skill_inspection"])
        self.assertEqual(payload["highlights"]["focus_skill_id"], "generic_data_handler")
        self.assertEqual(payload["views"]["skill_focus"]["skill_id"], "generic_data_handler")

    def test_report_json_contains_frontend_sections(self) -> None:
        result = run_cli("report", "library", str(LIBRARY_PATH), "--format", "json")
        payload = json.loads(result.stdout)
        self.assertIn("views", payload)
        self.assertIn("risk_table", payload["views"])
        self.assertIn("sections", payload)

    def test_compare_and_simulate_outputs_differ(self) -> None:
        compare_result = run_cli("compare", "candidate", str(LIBRARY_PATH), "--file", str(CANDIDATE_PATH))
        simulate_result = run_cli("simulate", "add", str(LIBRARY_PATH), "--file", str(CANDIDATE_PATH))
        self.assertNotEqual(compare_result.stdout, simulate_result.stdout)
        self.assertIn("simulate_add", simulate_result.stdout)

    def test_optimize_plan_outputs_actions(self) -> None:
        result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["library_id"], "sample_skill_library")
        self.assertIn("actions", payload)
        self.assertTrue(payload["actions"])

    def test_diff_library_outputs_summary(self) -> None:
        base_text = LIBRARY_PATH.read_text(encoding="utf-8").rstrip()
        appended_skill = """
  - id: parse_reference_section
    name: Parse Reference Section
    description: Extract individual bibliography entries from a paper's references section and return structured citation records.
    examples:
      - Parse the references section into individual citations.
    family: citation_tools
    tags: [citation, bibliography, parsing]
    inputs:
      reference_text:
        type: string
        required: true
    outputs:
      citations:
        type: array
    anchors:
      verbs: [extract, parse]
      objects: [references, bibliography entries, citations]
      constraints: [references section only, structured citation records]
    metadata:
      owner: research
      source: candidate
      version: v1
      created_at: null
      updated_at: null
""".rstrip()
        candidate_library_text = base_text.replace(
            "pipeline_edges:\n",
            f"{appended_skill}\n\npipeline_edges:\n",
            1,
        )
        candidate_library_path = None
        result = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
                handle.write(candidate_library_text)
                candidate_library_path = Path(handle.name)
            result = run_cli("diff", "library", str(LIBRARY_PATH), str(candidate_library_path))
        finally:
            if candidate_library_path is not None:
                candidate_library_path.unlink(missing_ok=True)
        self.assertIsNotNone(result)
        self.assertIn("# Skill Manager Diff:", result.stdout)
        self.assertIn("Added skills", result.stdout)

    def test_diff_library_json_contains_viewer_sections(self) -> None:
        base_text = LIBRARY_PATH.read_text(encoding="utf-8").rstrip()
        appended_skill = """
  - id: parse_reference_section
    name: Parse Reference Section
    description: Extract individual bibliography entries from a paper's references section and return structured citation records.
    examples:
      - Parse the references section into individual citations.
    family: citation_tools
    tags: [citation, bibliography, parsing]
    inputs:
      reference_text:
        type: string
        required: true
    outputs:
      citations:
        type: array
    anchors:
      verbs: [extract, parse]
      objects: [references, bibliography entries, citations]
      constraints: [references section only, structured citation records]
    metadata:
      owner: research
      source: candidate
      version: v1
      created_at: null
      updated_at: null
""".rstrip()
        candidate_library_text = base_text.replace(
            "pipeline_edges:\n",
            f"{appended_skill}\n\npipeline_edges:\n",
            1,
        )
        candidate_library_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
                handle.write(candidate_library_text)
                candidate_library_path = Path(handle.name)
            result = run_cli("diff", "library", str(LIBRARY_PATH), str(candidate_library_path), "--format", "json")
        finally:
            if candidate_library_path is not None:
                candidate_library_path.unlink(missing_ok=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["report_type"], "auto_skill_manager_library_diff")
        self.assertIn("metric_delta_cards", payload["views"])
        self.assertIn("skill_delta_table", payload["views"])
        self.assertEqual(payload["highlights"]["added_skill_count"], 1)

    def test_optimize_plan_only_rewrite_filters_actions(self) -> None:
        result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json", "--only", "rewrite")
        payload = json.loads(result.stdout)
        self.assertTrue(payload["actions"])
        self.assertTrue(all(action["action_type"] == "rewrite" for action in payload["actions"]))
        self.assertEqual(payload["metadata"]["only_action_types"], ["rewrite"])

    def test_optimize_plan_only_boundary_rewrite_filters_actions(self) -> None:
        result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json", "--only", "boundary_rewrite")
        payload = json.loads(result.stdout)
        self.assertTrue(payload["actions"])
        self.assertTrue(all(action["action_subtype"] == "boundary_rewrite" for action in payload["actions"]))
        self.assertEqual(payload["metadata"]["only_action_types"], ["boundary_rewrite"])

    def test_optimize_plan_skill_filters_actions(self) -> None:
        result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json", "--skill", "generic_data_handler")
        payload = json.loads(result.stdout)
        self.assertTrue(payload["actions"])
        self.assertTrue(all("generic_data_handler" in action["target_skill_ids"] for action in payload["actions"]))
        self.assertEqual(payload["metadata"]["skill_id"], "generic_data_handler")

    def test_optimize_apply_outputs_optimization_report(self) -> None:
        plan_result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json")
        plan = json.loads(plan_result.stdout)
        plan["actions"][0]["status"] = "applied"
        plan["actions"][0]["proposed_changes"] = {
            "description": "Extract precise citation metadata from academic PDFs.",
            "anchors": {
                "verbs": ["extract", "parse"],
                "objects": ["citation", "metadata", "academic pdf"],
                "constraints": ["structured fields only"],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(plan, handle)
            plan_path = Path(handle.name)
        try:
            result = run_cli("optimize", "apply", str(LIBRARY_PATH), "--plan", str(plan_path), "--format", "json")
        finally:
            plan_path.unlink(missing_ok=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["report_type"], "auto_skill_manager_optimization")
        self.assertTrue(payload["highlights"]["is_optimization"])
        self.assertIn("metric_delta_cards", payload["views"])
        self.assertIn("skill_delta_table", payload["views"])
        self.assertIn("action_impact_cards", payload["views"])
        self.assertTrue(payload["views"]["action_impact_cards"])

    def test_optimize_apply_can_write_optimized_library(self) -> None:
        plan_result = run_cli("optimize", "plan", str(LIBRARY_PATH), "--format", "json", "--skill", "generic_data_handler")
        plan = json.loads(plan_result.stdout)
        plan["actions"][0]["status"] = "applied"
        plan["actions"][0]["proposed_changes"] = {
            "description": "Transform structured tabular input into normalized records for analytics workflows.",
            "anchors": {
                "verbs": ["transform", "normalize"],
                "objects": ["tabular input", "records"],
                "constraints": ["structured data only"],
            },
        }
        plan_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump(plan, handle)
                plan_path = Path(handle.name)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
                output_path = Path(handle.name)
            result = run_cli(
                "optimize",
                "apply",
                str(LIBRARY_PATH),
                "--plan",
                str(plan_path),
                "--write-library",
                str(output_path),
                "--format",
                "json",
            )
            payload = json.loads(result.stdout)
            optimized_library = __import__("yaml").safe_load(output_path.read_text(encoding="utf-8"))
        finally:
            if plan_path is not None:
                plan_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

        self.assertEqual(payload["report_type"], "auto_skill_manager_optimization")
        target = next(skill for skill in optimized_library["skills"] if skill["id"] == "generic_data_handler")
        self.assertEqual(target["description"], "Transform structured tabular input into normalized records for analytics workflows.")
        self.assertEqual(target["anchors"]["verbs"], ["transform", "normalize"])
