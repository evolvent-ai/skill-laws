from __future__ import annotations

import re
from dataclasses import dataclass

from auto_skill_manager.schema.library import LibraryRecord
from auto_skill_manager.schema.skill import SkillRecord

LOCAL_ARTIFACT_TASK_PREFIXES = {
    "acct",
    "cal",
    "code",
    "comm",
    "db",
    "debug",
    "doc",
    "ds",
    "eml",
    "file",
    "fin",
    "law",
    "mag",
    "math",
    "mem",
    "plan",
    "reg",
    "sec",
    "sys",
    "web",
    "wfl",
    "xdom",
}
LOCAL_ARTIFACT_SKILL_LIMIT = 60
TOOL_ROUTING_SKILL_LIMIT = 140
DOMAIN_PROFILES = {
    "db": {
        "positive": {"sql", "sqlite", "database", "query", "schema", "migration", "table", "join"},
        "negative": {"slack", "calendar", "email", "pdf", "browser", "webhook", "kubernetes"},
    },
    "doc": {
        "positive": {"markdown", "html", "document", "convert", "format", "heading", "link"},
        "negative": {"sql", "database", "calendar", "security", "kubernetes", "terraform"},
    },
    "xdom": {
        "positive": {"documentation", "api", "architecture", "module", "python", "index", "guide"},
        "negative": {"calendar", "email", "spreadsheet", "terraform", "kubernetes"},
    },
    "code": {
        "positive": {"python", "code", "test", "function", "class", "debug", "implementation"},
        "negative": {"calendar", "email", "crm", "slack", "pdf"},
    },
    "debug": {
        "positive": {"debug", "fix", "test", "error", "traceback", "performance", "python"},
        "negative": {"calendar", "email", "crm", "slack", "pdf"},
    },
    "tool": {
        "positive": {"tool", "api", "workflow", "integration", "github", "yaml", "automation"},
        "negative": set(),
    },
}


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    skill: SkillRecord
    score: float
    reason: str


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_./+-]+", text.lower()) if len(token) >= 2}


def task_prefix(task_id: str) -> str:
    return task_id.lower().split("-", 1)[0]


def is_local_artifact_task(task_id: str) -> bool:
    return task_prefix(task_id) in LOCAL_ARTIFACT_TASK_PREFIXES


def domain_skill_limit(task_id: str, limit: int) -> int:
    if is_local_artifact_task(task_id):
        return min(limit, LOCAL_ARTIFACT_SKILL_LIMIT)
    return min(limit, TOOL_ROUTING_SKILL_LIMIT)


def score_skill(skill: SkillRecord, instruction_tokens: set[str], task_id: str) -> tuple[float, str]:
    name_tokens = tokenize(skill.name)
    desc_tokens = tokenize(skill.description)
    family_tokens = tokenize(skill.family or "")
    tag_tokens = set()
    for tag in skill.tags:
        tag_tokens |= tokenize(tag)

    anchor_tokens = set()
    for key in ("verbs", "objects", "constraints"):
        for item in skill.anchors.get(key) or []:
            anchor_tokens |= tokenize(item)

    score = 0.0
    name_hits = name_tokens & instruction_tokens
    tag_hits = tag_tokens & instruction_tokens
    anchor_hits = anchor_tokens & instruction_tokens
    desc_hits = desc_tokens & instruction_tokens
    family_hits = family_tokens & instruction_tokens

    score += 4.0 * len(name_hits)
    score += 2.0 * len(tag_hits)
    score += 1.5 * len(anchor_hits)
    score += 1.0 * len(desc_hits)
    score += 0.5 * len(family_hits)

    lowered_task = task_id.lower()
    prefix = task_prefix(task_id)
    if prefix and prefix in family_tokens:
        score += 3.0
    if lowered_task in tag_tokens or lowered_task in name_tokens:
        score += 6.0

    all_skill_tokens = name_tokens | desc_tokens | family_tokens | tag_tokens | anchor_tokens
    profile = DOMAIN_PROFILES.get(prefix, {})
    positive_hits = set(profile.get("positive", set())) & all_skill_tokens
    negative_hits = set(profile.get("negative", set())) & all_skill_tokens
    if positive_hits:
        score += 1.25 * len(positive_hits)
    if negative_hits and not (name_hits or tag_hits or anchor_hits):
        score -= 1.0 * len(negative_hits)

    reason_parts = []
    if name_hits:
        reason_parts.append(f"name={','.join(sorted(name_hits)[:4])}")
    if tag_hits:
        reason_parts.append(f"tags={','.join(sorted(tag_hits)[:4])}")
    if anchor_hits:
        reason_parts.append(f"anchors={','.join(sorted(anchor_hits)[:4])}")
    if desc_hits:
        reason_parts.append(f"desc={','.join(sorted(desc_hits)[:4])}")
    if positive_hits:
        reason_parts.append(f"domain={','.join(sorted(positive_hits)[:4])}")
    if negative_hits and not (name_hits or tag_hits or anchor_hits):
        reason_parts.append(f"penalty={','.join(sorted(negative_hits)[:4])}")
    return score, "; ".join(reason_parts) or "tie-break"


def select_skills(
    library: LibraryRecord,
    instruction: str,
    task_id: str,
    *,
    limit: int = 250,
    mode: str = "domain-gated",
) -> list[RuntimeSelection]:
    if mode not in {"full", "domain-gated", "domain-profile"}:
        raise ValueError("mode must be 'full', 'domain-gated', or 'domain-profile'")

    instruction_tokens = tokenize(instruction) | tokenize(task_id)
    scored = []
    for skill in library.skills:
        score, reason = score_skill(skill, instruction_tokens, task_id)
        scored.append(RuntimeSelection(skill=skill, score=score, reason=reason))
    if mode in {"domain-gated", "domain-profile"}:
        limit = domain_skill_limit(task_id, limit)
        scored = [item for item in scored if item.score > 0]

    return sorted(scored, key=lambda item: (item.score, item.skill.name), reverse=True)[:limit]


def runtime_guidance(task_id: str, mode: str) -> str:
    if mode in {"domain-gated", "domain-profile"} and is_local_artifact_task(task_id):
        return (
            "For local workspace artifact tasks, the task instruction and files in the workspace are authoritative. "
            "Use skill context only when it gives a direct routing or format hint; do not let unrelated tool/vendor "
            "skills override requested filenames, schemas, or verifier-visible outputs."
        )
    return (
        "Use skill context as routing guidance. Prefer concrete skills whose names, descriptions, anchors, or tags "
        "match the task. Avoid being distracted by nearby but wrong product/vendor matches."
    )


def build_skill_context(
    library: LibraryRecord,
    instruction: str,
    task_id: str,
    *,
    limit: int = 250,
    mode: str = "domain-gated",
) -> str:
    selected = select_skills(library, instruction, task_id, limit=limit, mode=mode)
    lines = [
        "Available skill repository context:",
        f"library_id: {library.library_id}",
        f"tool_count: {len(library.skills)}",
        f"selection_mode: {mode}",
        f"selected_tool_count: {len(selected)}",
        f"estimated_context_chars: {sum(len(item.skill.description) + len(item.skill.name) + 120 for item in selected)}",
        runtime_guidance(task_id, mode),
        "",
    ]
    for item in selected:
        skill = item.skill
        anchor_parts = []
        for key in ("verbs", "objects", "constraints"):
            values = skill.anchors.get(key) or []
            if values:
                anchor_parts.append(f"{key}={', '.join(values[:4])}")
        tags = ", ".join(skill.tags[:5])
        lines.append(
            f"- id: {skill.id} | name: {skill.name} | score: {item.score:.2f} | "
            f"reason: {item.reason} | desc: {skill.description} | anchors: {'; '.join(anchor_parts) or 'none'} | tags: {tags}"
        )
    return "\n".join(lines)


def build_selection_payload(
    library: LibraryRecord,
    instruction: str,
    task_id: str,
    *,
    limit: int = 250,
    mode: str = "domain-profile",
) -> dict:
    selected = select_skills(library, instruction, task_id, limit=limit, mode=mode)
    return {
        "library_id": library.library_id,
        "task_id": task_id,
        "mode": mode,
        "requested_limit": limit,
        "effective_limit": domain_skill_limit(task_id, limit) if mode != "full" else limit,
        "selected_tool_count": len(selected),
        "is_local_artifact_task": is_local_artifact_task(task_id),
        "skills": [
            {
                "id": item.skill.id,
                "name": item.skill.name,
                "family": item.skill.family,
                "score": round(item.score, 4),
                "reason": item.reason,
                "tags": item.skill.tags,
            }
            for item in selected
        ],
    }
