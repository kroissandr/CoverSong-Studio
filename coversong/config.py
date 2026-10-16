import copy
import os

import yaml

from . import ROOT


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_config(path=None, overrides=()):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value.strip())
    for k, v in cfg.get("paths", {}).items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.normpath(os.path.join(ROOT, v))
    return cfg
