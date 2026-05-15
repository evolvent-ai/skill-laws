from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RecommendationRecord:
    id: str
    type: str
    priority: str
    confidence: float
    entities: dict[str, list[str]] = field(default_factory=dict)
    summary: str = ""
    rationale: dict[str, Any] = field(default_factory=dict)
    suggested_action: dict[str, Any] = field(default_factory=dict)
