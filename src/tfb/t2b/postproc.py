"""Durable write of a JSON record.
"""
from __future__ import annotations
import json, os
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str)); os.replace(tmp, path)
