from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from skill_law.paths import DATA_ROOT, SKILLS_DIR, data_path, finding_path


GROUPS = [
    "file-search",
    "log-analyze",
    "sql-query",
    "api-test",
    "config-edit",
    "report-build",
    "security-scan",
    "data-clean",
    "chart-plot",
    "deploy-check",
]
DESCRIPTORS = ["fast", "deep", "secure", "smart", "batch", "quick", "advanced", "auto"]


def skill_name(group_idx: int, descriptor_idx: int, variant: int) -> str:
    return f"{DESCRIPTORS[descriptor_idx]}-{GROUPS[group_idx]}-{variant:02d}"


def skill_description(name: str, group: str, descriptor: str) -> str:
    words = group.replace("-", " ")
    return (
        f"{name} handles {words} tasks with a {descriptor} routing profile. "
        f"Use it when the user needs {descriptor} {words}, concrete inputs, and a checkable output artifact."
    )


def build_skill_rows() -> list[dict]:
    rows = []
    for gi, group in enumerate(GROUPS):
        for di, descriptor in enumerate(DESCRIPTORS):
            for variant in range(4):
                name = skill_name(gi, di, variant)
                rows.append(
                    {
                        "name": name,
                        "base_name": group,
                        "description": skill_description(name, group, descriptor),
                    }
                )
    return rows


def ensure_skills(rows: list[dict]) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = SKILLS_DIR / row["name"] / "SKILL.md"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"name: {row['name']}",
                    f"description: {row['description']}",
                    "---",
                    "",
                    f"Accept a concrete request and produce the requested {row['base_name'].replace('-', ' ')} output.",
                    "Validate inputs, return a concise artifact, and preserve relevant intermediate state.",
                ]
            ),
            encoding="utf-8",
        )


def ensure_processed(rows: list[dict]) -> None:
    processed = data_path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "high_similarity_skills.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (processed / "synthetic_skill_library.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    names = sorted(row["name"] for row in rows)
    row_by_name = {row["name"]: row for row in rows}
    rng = np.random.default_rng(20260516)
    dim = 48
    group_centers = {group: rng.normal(size=dim) for group in GROUPS}
    descriptor_centers = {desc: rng.normal(size=dim) for desc in DESCRIPTORS}
    embs = []
    for name in names:
        row = row_by_name[name]
        descriptor = name.split("-", 1)[0]
        vec = 1.7 * group_centers[row["base_name"]] + 0.7 * descriptor_centers[descriptor] + 0.15 * rng.normal(size=dim)
        vec = vec / max(np.linalg.norm(vec), 1e-12)
        embs.append(vec)
    np.savez(processed / "bge_skill_embeddings.npz", embs=np.asarray(embs, dtype=np.float32))


def task_row(idx: int, ids: list[str]) -> dict:
    groups = [" ".join(s.split("-")[1:-1]) for s in ids]
    desc = " then ".join(f"perform {g}" for g in groups)
    return {
        "idx": idx,
        "task_desc": f"Complete a workflow that must {desc} and return the final artifact.",
        "required_steps": len(ids),
        "required_skills": [{"id": sid, "name": sid} for sid in ids],
        "gold_trace": ids,
        "human_verified": True,
    }


def ensure_benchmark(rows: list[dict]) -> None:
    out = finding_path("F01", "data", "benchmark_tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    names_by_group = {}
    for row in rows:
        names_by_group.setdefault(row["base_name"], []).append(row["name"])
    tasks = []
    idx = 0
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
        for offset in range(4):
            ids = []
            for step in range(k):
                group = GROUPS[(offset + step) % len(GROUPS)]
                ids.append(names_by_group[group][(offset + step) % len(names_by_group[group])])
            tasks.append(task_row(idx, ids))
            idx += 1
    out.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def ensure_f06_targets(rows: list[dict]) -> None:
    out = finding_path("F06", "analysis", "F06_counterexample_intervention_targets.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    by_group = {}
    for row in rows:
        by_group.setdefault(row["base_name"], []).append(row["name"])
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_key", "group_key", "comparison"])
        writer.writeheader()
        for group in GROUPS[:6]:
            ids = by_group[group]
            writer.writerow({"pair_key": f"{ids[0]}__{ids[1]}", "group_key": group, "comparison": "descriptor_pair"})


def rubric() -> list[dict]:
    return [
        {"criterion": "uses required input", "weight": 0.25, "pass_condition": "uses the supplied state", "fail_condition": "ignores the supplied state"},
        {"criterion": "correct artifact", "weight": 0.35, "pass_condition": "returns the requested artifact", "fail_condition": "returns an unrelated artifact"},
        {"criterion": "complete output", "weight": 0.25, "pass_condition": "covers all requested parts", "fail_condition": "omits key parts"},
        {"criterion": "valid format", "weight": 0.15, "pass_condition": "is directly usable", "fail_condition": "is malformed"},
    ]


def ensure_execution_pairs(rows: list[dict]) -> None:
    by_group = {}
    for row in rows:
        by_group.setdefault(row["base_name"], []).append(row)
    pairs = []
    deps = ["tight", "loose", "independent"]
    for i, dep in enumerate(deps * 2):
        a = by_group[GROUPS[i]][0]
        b = by_group[GROUPS[(i + 1) % len(GROUPS)]][1]
        pairs.append(
            {
                "a_id": a["name"],
                "b_id": b["name"],
                "a_desc": a["description"],
                "b_desc": b["description"],
                "dependency": dep,
                "task_desc": f"Use {a['base_name'].replace('-', ' ')} output as state for {b['base_name'].replace('-', ' ')}.",
                "rubric": rubric(),
                "rubric_a": rubric(),
                "rubric_b": rubric(),
            }
        )
    f11 = finding_path("F11", "data", "F11_pairs_v4_rubrics.json")
    f12 = finding_path("F12", "data", "F12_pairs.json")
    f11.parent.mkdir(parents=True, exist_ok=True)
    f12.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(pairs, indent=2)
    f11.write_text(payload, encoding="utf-8")
    f12.write_text(payload, encoding="utf-8")


def ensure_f09_independence(rows: list[dict]) -> None:
    out_dir = finding_path("F09", "data", "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:12]
    records = []
    for idx in range(0, len(selected) - 1, 2):
        a = selected[idx]
        b = selected[idx + 1]
        p_a = 0.55 + 0.05 * ((idx // 2) % 3)
        p_b = 0.50 + 0.04 * ((idx // 2) % 4)
        p_c = max(0.0, min(1.0, p_a * p_b + (-0.01 + 0.005 * ((idx // 2) % 5))))
        records.append(
            {
                "skill_a": a["name"],
                "skill_b": b["name"],
                "p_a": p_a,
                "p_b": p_b,
                "p_c": p_c,
            }
        )
    (out_dir / "cooperation_large_scale_gpt-5.4-mini.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def ensure_demo_data() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_skill_rows()
    ensure_skills(rows)
    ensure_processed(rows)
    ensure_benchmark(rows)
    ensure_f06_targets(rows)
    ensure_execution_pairs(rows)
    ensure_f09_independence(rows)
