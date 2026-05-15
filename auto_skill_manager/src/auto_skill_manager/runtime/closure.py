from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ClosureCheck:
    ok: bool
    missing: list[str]
    empty: list[str]
    expected: list[str]

    def feedback(self) -> str:
        if self.ok:
            return "All required output artifacts are present and non-empty."
        parts = ["The task is not complete yet."]
        if self.missing:
            parts.append(f"Missing required output files: {', '.join(self.missing)}.")
        if self.empty:
            parts.append(f"Empty required output files: {', '.join(self.empty)}.")
        parts.append("Continue working until these artifacts exist in the workspace and contain final output.")
        return " ".join(parts)


EXPLICIT_REQUIRED_OUTPUTS = {
    "db-002": ["query.sql", "results.json"],
    "db-004": ["migrate.sql"],
    "debug-001": ["fixed.py"],
    "doc-002": ["output.html"],
    "eml-001": ["headers.json"],
    "xdom-012": ["api_docs.md", "architecture.md", "getting_started.md", "index.json"],
}


def infer_required_outputs(task_id: str, instruction: str) -> list[str]:
    lowered = task_id.lower()
    if lowered in EXPLICIT_REQUIRED_OUTPUTS:
        return list(EXPLICIT_REQUIRED_OUTPUTS[lowered])

    patterns = [
        r"Save(?: the [^`]+)? as `(?:[^`/]*/)?([^`/]+)`",
        r"Write(?: the [^`]+)? to `(?:[^`/]*/)?([^`/]+)`",
        r"Create `(?:[^`/]*/)?([^`/]+)`",
    ]
    seen: set[str] = set()
    outputs: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, instruction, flags=re.IGNORECASE):
            item = match.strip()
            if item and item not in seen:
                seen.add(item)
                outputs.append(item)
    return outputs


def check_required_outputs(workspace: Path, required_outputs: list[str]) -> ClosureCheck:
    missing: list[str] = []
    empty: list[str] = []
    for rel in required_outputs:
        path = workspace / rel
        if not path.exists():
            missing.append(rel)
        elif path.is_file() and path.stat().st_size == 0:
            empty.append(rel)
    return ClosureCheck(
        ok=not missing and not empty,
        missing=missing,
        empty=empty,
        expected=list(required_outputs),
    )
