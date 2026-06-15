from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .model_families import default_model_path, get_model_family, normalize_family_name


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
    apply_model_family_defaults(cfg)
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


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _looks_like_other_family_path(value: Any, family: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    lowered = value.lower()
    markers = {
        "llama": ("llama", "llama3"),
        "phi": ("phi", "phi4"),
        "qwen": ("qwen", "qwen3"),
    }
    return any(
        marker in lowered
        for other, other_markers in markers.items()
        if other != family
        for marker in other_markers
    )


def apply_model_family_defaults(cfg: dict[str, Any]) -> None:
    family_name = normalize_family_name(deep_get(cfg, "model_family", deep_get(cfg, "model.family", "llama")))
    family = get_model_family(family_name)
    root = deep_get(cfg, "root", ".")
    deep_set(cfg, "model_family", family_name)
    deep_set(cfg, "model.family", family_name)
    deep_set(cfg, "model.family_name", family.display_name)
    if _is_missing(deep_get(cfg, "model.trust_remote_code")):
        deep_set(cfg, "model.trust_remote_code", family.trust_remote_code)
    if _is_missing(deep_get(cfg, "model.attn_implementation")) and family.attn_implementation:
        deep_set(cfg, "model.attn_implementation", family.attn_implementation)

    source_model = deep_get(cfg, "paths.source_model")
    if _is_missing(source_model) or _looks_like_other_family_path(source_model, family_name):
        deep_set(cfg, "paths.source_model", default_model_path(root, family_name, "proxy"))
    target_model = deep_get(cfg, "paths.target_model")
    if _is_missing(target_model) or _looks_like_other_family_path(target_model, family_name):
        deep_set(cfg, "paths.target_model", default_model_path(root, family_name, "target"))

    target_modules = deep_get(cfg, "unlearning.target_modules")
    if _is_missing(target_modules):
        deep_set(cfg, "unlearning.target_modules", list(family.target_modules))


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
