"""configs/final.yaml: the single source of the final recipe's parameters.

python -m tfb.final_config   -> shell assignments (CFG_TASK1_LGBM_ROUNDS=20000 ...) for scripts/reproduce.sh
get(key, default)            -> the yaml value when TFB_CONFIG is set (reproduce.sh exports it), else the code default
"""
from __future__ import annotations
import os
import shlex
from functools import lru_cache
from pathlib import Path
import yaml
from . import io

PATH = io.PROJECT / "configs" / "final.yaml"


@lru_cache(maxsize=None)
def load(path=None) -> dict:
    return yaml.safe_load(Path(path or os.environ.get("TFB_CONFIG") or PATH).read_text())


def get(key: str, default=None):
    if not os.environ.get("TFB_CONFIG"):
        return default
    value = load()
    for k in key.split("."):
        value = value[k]          # a missing key fails loudly
    return value


def flat(d: dict, prefix="CFG"):
    for k, v in d.items():
        name = f"{prefix}_{k}".upper()
        if isinstance(v, dict): yield from flat(v, name)
        else: yield name, " ".join(map(str, v)) if isinstance(v, list) else str(v)


if __name__ == "__main__":
    print("\n".join(f"{k}={shlex.quote(v)}" for k, v in flat(load())))
