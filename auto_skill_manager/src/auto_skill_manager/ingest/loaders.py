from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from auto_skill_manager.schema.edge import PipelineEdgeRecord
from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.skill import SkillRecord


class SchemaError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"Expected mapping at top level in {path}")
    return data


def load_library(path: Path) -> LibraryRecord:
    raw = load_yaml(path)
    skills_raw = raw.get("skills", [])
    if not isinstance(skills_raw, list) or not skills_raw:
        raise SchemaError("Library file must contain a non-empty 'skills' list")

    skills = [parse_skill_record(item) for item in skills_raw]
    edges_raw = raw.get("pipeline_edges", [])
    edges = [parse_pipeline_edge(item) for item in edges_raw]

    return LibraryRecord(
        library_id=str(raw.get("library_id") or path.stem),
        skills=skills,
        pipeline_edges=edges,
        metadata=_ensure_dict(raw.get("metadata")),
    )


def load_skill(path: Path) -> SkillRecord:
    return parse_skill_record(load_yaml(path))


def write_library(library: LibraryRecord, path: Path) -> None:
    payload = to_plain_data(library)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def parse_skill_record(raw: dict[str, Any]) -> SkillRecord:
    if not isinstance(raw, dict):
        raise SchemaError("Skill record must be a mapping")
    skill_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    description = str(raw.get("description") or "").strip()
    if not skill_id or not name or not description:
        raise SchemaError("Skill record requires non-empty id, name, and description")

    return SkillRecord(
        id=skill_id,
        name=name,
        description=description,
        label=_optional_str(raw.get("label")) or "",
        examples=_ensure_str_list(raw.get("examples")),
        family=_optional_str(raw.get("family")),
        tags=_ensure_str_list(raw.get("tags")),
        inputs=_ensure_dict(raw.get("inputs")),
        outputs=_ensure_dict(raw.get("outputs")),
        anchors=_normalize_anchors(raw.get("anchors")),
        metadata=_ensure_dict(raw.get("metadata")),
    )


def parse_pipeline_edge(raw: dict[str, Any]) -> PipelineEdgeRecord:
    if not isinstance(raw, dict):
        raise SchemaError("Pipeline edge must be a mapping")

    upstream = str(raw.get("upstream_skill") or "").strip()
    downstream = str(raw.get("downstream_skill") or "").strip()
    dependency_type = str(raw.get("dependency_type") or "").strip()
    if not upstream or not downstream or not dependency_type:
        raise SchemaError("Pipeline edge requires upstream_skill, downstream_skill, dependency_type")

    return PipelineEdgeRecord(
        upstream_skill=upstream,
        downstream_skill=downstream,
        dependency_type=dependency_type,
        weight=float(raw.get("weight", 1.0)),
        notes=_optional_str(raw.get("notes")),
    )


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def _normalize_anchors(raw: Any) -> dict[str, list[str]]:
    anchors = _ensure_dict(raw)
    return {
        "verbs": _ensure_str_list(anchors.get("verbs")),
        "objects": _ensure_str_list(anchors.get("objects")),
        "constraints": _ensure_str_list(anchors.get("constraints")),
    }


def _ensure_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SchemaError("Expected mapping")
    return dict(value)


def _ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaError("Expected list")
    return [str(item) for item in value]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
