from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SKILL_LAW_DATA_ROOT", ROOT / "data")).resolve()
SKILLS_DIR = Path(os.environ.get("SKILL_LAW_SKILLS_DIR", DATA_ROOT / "skills")).resolve()


def _default_env_file() -> Path:
    for base in (Path.cwd(), ROOT, *ROOT.parents):
        candidate = base / ".env"
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / ".env").resolve()


ENV_FILE = Path(os.environ.get("SKILL_LAW_ENV_FILE", _default_env_file())).resolve()


def finding_path(finding_id: str, *parts: str) -> Path:
    return DATA_ROOT.joinpath(finding_id, *parts)


def finding_data_path(finding_id: str, *parts: str) -> Path:
    return finding_path(finding_id, *parts)


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)
