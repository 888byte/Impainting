"""Small logging helpers used by the refactored CLI."""
from __future__ import annotations

import json
from typing import Any, Dict


def dump_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")
