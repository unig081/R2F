from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default if default is not None else "")

    previous = None
    current = text
    for _ in range(8):
        if current == previous:
            break
        previous = current
        current = _ENV_PATTERN.sub(replace, current)
    return current


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(_expand_env(f.read()))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def deep_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def deep_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    smoke = cfg.get("smoke", {})
    if not smoke:
        return cfg

    if "max_length" in smoke:
        deep_set(cfg, "model.max_length", smoke["max_length"])
    if "max_forget_samples" in smoke:
        deep_set(cfg, "decoder_samples.max_forget_samples", smoke["max_forget_samples"])
        deep_set(cfg, "unlearning.max_steps", smoke["max_forget_samples"])
    if "coords_per_module" in smoke:
        deep_set(cfg, "decoder_samples.coords_per_module", smoke["coords_per_module"])
    if "target_modules" in smoke:
        deep_set(cfg, "unlearning.target_modules", smoke["target_modules"])
    if "train_layers" in smoke:
        deep_set(cfg, "unlearning.train_layers", smoke["train_layers"])
    if "decoder_steps" in smoke:
        deep_set(cfg, "decoder.max_steps", smoke["decoder_steps"])
    if "r2f_gradient_capture_steps" in smoke:
        deep_set(cfg, "r2f.gradient_capture_steps", smoke["r2f_gradient_capture_steps"])
    if "r2f_eta_grid" in smoke:
        deep_set(cfg, "r2f.eta_grid", smoke["r2f_eta_grid"])
    if "eval_max_forget" in smoke:
        deep_set(cfg, "evaluation.max_forget", smoke["eval_max_forget"])
    if "eval_max_retain" in smoke:
        deep_set(cfg, "evaluation.max_retain", smoke["eval_max_retain"])
    if "max_retain_samples" in smoke:
        deep_set(cfg, "unlearning.max_retain_samples", smoke["max_retain_samples"])
    return cfg


def resolve_output_path(cfg: dict[str, Any], path: str) -> Path:
    value = deep_get(cfg, path)
    if value is None:
        raise KeyError(path)
    return Path(str(value)).expanduser()
