from __future__ import annotations

from auto_skill_manager.schema.skill import SkillRecord


def tokenize(text: str) -> set[str]:
    return {token.strip(".,:;()[]{}\"'").lower() for token in text.split() if token.strip()}


def abstraction_score(description: str) -> float:
    description_lower = description.lower()
    abstraction_terms = ["generic", "general", "handle", "process", "multiple", "workflow", "various"]
    abstraction_hits = sum(1 for token in abstraction_terms if token in description_lower)
    return min(1.0, abstraction_hits / 4.0)


def anchor_strength(skill: SkillRecord) -> float:
    anchors = skill.anchors or {}
    verbs = anchors.get("verbs", [])
    objects = anchors.get("objects", [])
    constraints = anchors.get("constraints", [])
    examples = skill.examples or []
    description_tokens = tokenize(skill.description)
    action_specificity = min(1.0, 0.18 * len(verbs))
    object_specificity = min(1.0, 0.16 * len(objects))
    constraint_density = min(1.0, 0.14 * len(constraints))
    example_support = min(1.0, 0.1 * len(examples))
    length_bonus = 0.1 if 8 <= len(description_tokens) <= 30 else 0.0
    abstraction_penalty = 0.22 * abstraction_score(skill.description)
    strength = action_specificity + object_specificity + constraint_density + example_support + length_bonus - abstraction_penalty
    return max(0.0, min(1.0, strength))
