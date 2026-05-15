from pathlib import Path
import unittest

from auto_skill_manager.ingest import load_library, load_skill


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "examples" / "library.sample.yaml"
CANDIDATE_PATH = ROOT / "examples" / "candidate.sample.yaml"


class LoaderTests(unittest.TestCase):
    def test_load_library_reads_sample_file(self) -> None:
        library = load_library(LIBRARY_PATH)
        self.assertEqual(library.library_id, "sample_skill_library")
        self.assertEqual(len(library.skills), 4)
        self.assertEqual(len(library.pipeline_edges), 2)

    def test_load_skill_reads_candidate_file(self) -> None:
        candidate = load_skill(CANDIDATE_PATH)
        self.assertEqual(candidate.id, "parse_reference_section")
        self.assertEqual(candidate.family, "citation_tools")
