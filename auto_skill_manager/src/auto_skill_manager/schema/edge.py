from dataclasses import dataclass


@dataclass(slots=True)
class PipelineEdgeRecord:
    upstream_skill: str
    downstream_skill: str
    dependency_type: str
    weight: float = 1.0
    notes: str | None = None
