from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill_law.paths import ENV_FILE, SKILLS_DIR


@dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str | None = None
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class LibrarySpec:
    id: str
    skills: list[SkillSpec]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    instruction: str
    required_skills: list[Any] | None = None
    gold_trace: list[Any] | None = None
    input: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


def load_env(path: Path | None = None, *, override: bool = True) -> None:
    env_path = path or ENV_FILE
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if override or clean_key not in os.environ:
            os.environ[clean_key] = clean_value


def _extract_description(content: str) -> str:
    match = re.search(r"^description:\s*(.*?)(?=\n[a-zA-Z_]+:|\n---|\Z)", content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("|").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("---")]
    return " ".join(lines[:3])[:1000]


def load_skills(skills_dir: Path | None = None) -> list[SkillSpec]:
    root = skills_dir or SKILLS_DIR
    if not root.exists():
        raise FileNotFoundError(f"Skill directory not found: {root}. Set SKILL_LAW_SKILLS_DIR.")
    skills: list[SkillSpec] = []
    for path in sorted(root.glob("*/SKILL.md")):
        content = path.read_text(encoding="utf-8", errors="replace")
        skill_id = path.parent.name
        name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
        skills.append(
            SkillSpec(
                id=skill_id,
                name=name_match.group(1).strip() if name_match else skill_id,
                description=_extract_description(content),
                input_schema={},
                output_schema={},
            )
        )
    return skills


def load_skill_specs(skills_dir: Path | None = None) -> dict[str, SkillSpec]:
    return {skill.id: skill for skill in load_skills(skills_dir)}


def parse_skill_id(skill_id: str) -> dict[str, str]:
    parts = [part for part in skill_id.replace("_", "-").split("-") if part]
    descriptor = parts[0] if parts else skill_id
    function_root = "-".join(parts[1:]) if len(parts) > 1 else skill_id
    return {"descriptor": descriptor, "function_root": function_root}


def build_skill_to_task(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        for item in task.get("required_skills", []):
            sid = item.get("name") if isinstance(item, dict) else item
            if sid:
                mapping.setdefault(str(sid), []).append(task)
    return mapping


def choose_task_desc(target: str, task_map: dict[str, list[dict[str, Any]]], skills_map: dict[str, SkillSpec], *, use_function_keywords: bool = False) -> str:
    tasks = task_map.get(target) or []
    if tasks:
        return str(tasks[0].get("task_desc") or tasks[0].get("instruction") or "")
    parsed = parse_skill_id(target)
    if use_function_keywords:
        return f"Use the {parsed['function_root'].replace('-', ' ')} capability with {target}."
    skill = skills_map.get(target)
    return skill.description if skill and skill.description else f"Use {target}."


class LLMRouter:
    def __init__(self, model_name: str = "gpt-5.4-mini") -> None:
        self.model_name = model_name

    def _build_prompt(self, task: TaskSpec, library: LibrarySpec) -> tuple[str, str]:
        tools = "\n".join(f"- {skill.id}: {skill.description}" for skill in library.skills)
        system = f"Select exactly one tool ID from the available tools.\n\nAvailable tools:\n{tools}\n\nRespond only with the selected tool ID."
        return system, task.instruction

    def route(self, task: TaskSpec, library: LibrarySpec) -> tuple[str, dict[str, Any]]:
        load_env()
        try:
            import openai

            client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                timeout=120.0,
            )
            system, user = self._build_prompt(task, library)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            return "ERROR", {"error": str(exc)}
        ids = {skill.id for skill in library.skills}
        if raw in ids:
            return raw, {"raw": raw}
        match = re.search(r"[A-Za-z0-9_.:-]+", raw)
        return (match.group(0) if match else raw), {"raw": raw}


class StrictLLMRouter(LLMRouter):
    def _build_prompt(self, task: TaskSpec, library: LibrarySpec) -> tuple[str, str]:
        tools_json = json.dumps(
            [{"name": skill.id, "description": skill.description} for skill in library.skills],
            ensure_ascii=False,
        )
        system = (
            "You are a tool routing agent. You must call exactly one tool.\n\n"
            f"Available tools (JSON):\n{tools_json}\n\n"
            "Respond with ONLY the tool name you would call."
        )
        return system, task.instruction


class NameOnlyRouter(LLMRouter):
    def _build_prompt(self, task: TaskSpec, library: LibrarySpec) -> tuple[str, str]:
        names = "\n".join(f"- {skill.id}" for skill in library.skills)
        system = (
            "Select exactly one tool ID from the available tools.\n\n"
            f"Available tools:\n{names}\n\n"
            "Respond only with the selected tool ID."
        )
        return system, task.instruction


class DescriptionOnlyRouter(LLMRouter):
    def _build_prompt(self, task: TaskSpec, library: LibrarySpec) -> tuple[str, str]:
        descs = "\n".join(
            f"- Tool_{i+1}: {skill.description}" for i, skill in enumerate(library.skills)
        )
        id_map = "\n".join(
            f"  Tool_{i+1} = {skill.id}" for i, skill in enumerate(library.skills)
        )
        system = (
            "Select exactly one tool from the descriptions below.\n\n"
            f"Available tools:\n{descs}\n\n"
            f"ID mapping:\n{id_map}\n\n"
            "Respond only with the tool ID (from the mapping above)."
        )
        return system, task.instruction
