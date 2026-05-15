from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from skill_law.runtime import LLMRouter, LibrarySpec, SkillSpec, TaskSpec, load_env


@dataclass(frozen=True)
class SmokeCase:
    instruction: str
    expected: str


CASES = [
    SmokeCase("Summarize this long meeting transcript into decisions and action items.", "summarize_text"),
    SmokeCase("Translate this user message from English to Chinese.", "translate_text"),
    SmokeCase("Extract the dates, prices, and product names from this invoice.", "extract_entities"),
    SmokeCase("Classify this support ticket as billing, technical, or account access.", "classify_ticket"),
    SmokeCase("Rewrite this paragraph in a concise professional tone.", "rewrite_text"),
    SmokeCase("Generate a SQL query that counts orders by customer region.", "write_sql"),
    SmokeCase("Find the likely bug in this Python traceback.", "debug_python"),
    SmokeCase("Create a short unit test plan for this function.", "write_tests"),
    SmokeCase("Convert this JSON object into a markdown table.", "format_markdown"),
    SmokeCase("Check whether this answer follows all listed constraints.", "validate_constraints"),
]


LIBRARY = LibrarySpec(
    id="smoke_library",
    skills=[
        SkillSpec(id="summarize_text", description="Summarize long text into concise decisions, actions, or key points."),
        SkillSpec(id="translate_text", description="Translate text between natural languages while preserving meaning."),
        SkillSpec(id="extract_entities", description="Extract structured entities such as dates, prices, names, and identifiers."),
        SkillSpec(id="classify_ticket", description="Classify support requests into predefined categories."),
        SkillSpec(id="rewrite_text", description="Rewrite text for tone, clarity, brevity, or style."),
        SkillSpec(id="write_sql", description="Write SQL queries from data-analysis requests."),
        SkillSpec(id="debug_python", description="Diagnose Python errors, stack traces, and likely code bugs."),
        SkillSpec(id="write_tests", description="Design or write tests for code behavior."),
        SkillSpec(id="format_markdown", description="Format information as markdown tables, lists, or sections."),
        SkillSpec(id="validate_constraints", description="Check whether an answer satisfies explicit requirements and constraints."),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 10-case skill_law routing smoke test.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
    router = LLMRouter(model_name=model)
    cases = CASES[: args.limit]
    correct = 0
    for idx, case in enumerate(cases, start=1):
        chosen, meta = router.route(TaskSpec(id=f"smoke_{idx:02d}", instruction=case.instruction), LIBRARY)
        ok = chosen == case.expected
        correct += int(ok)
        detail = meta.get("raw") or meta.get("error", "")
        print(f"{idx:02d} expected={case.expected} chosen={chosen} ok={ok} detail={detail}")
    print(f"accuracy={correct}/{len(cases)}")
    return 0 if correct == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
