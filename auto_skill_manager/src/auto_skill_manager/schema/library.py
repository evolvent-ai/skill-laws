from dataclasses import dataclass, field
from typing import Any

from .edge import PipelineEdgeRecord
from .skill import SkillRecord


@dataclass(slots=True)
class LibraryRecord:
    library_id: str
    skills: list[SkillRecord]
    pipeline_edges: list[PipelineEdgeRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
