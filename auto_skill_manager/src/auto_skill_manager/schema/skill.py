from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillRecord:
    id: str
    name: str
    description: str
    label: str = ""
    examples: list[str] = field(default_factory=list)
    family: str | None = None
    tags: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    anchors: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
