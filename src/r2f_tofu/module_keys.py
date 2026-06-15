from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

MODULE_TO_ID = {name: i for i, name in enumerate(DEFAULT_TARGET_MODULES)}
ID_TO_MODULE = {i: name for name, i in MODULE_TO_ID.items()}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class ModuleKey:
    layer_idx: int
    module_type: str

    @property
    def module_id(self) -> int:
        return MODULE_TO_ID[self.module_type]

    def as_string(self) -> str:
        return f"layer{self.layer_idx}.{self.module_type}"


def parse_module_key(name: str, target_modules: Iterable[str]) -> ModuleKey | None:
    layer_match = LAYER_RE.search(name)
    if layer_match is None:
        return None
    layer_idx = int(layer_match.group(1))
    for module_type in target_modules:
        if re.search(rf"(?:^|\.){re.escape(module_type)}(?:\.|$)", name):
            return ModuleKey(layer_idx=layer_idx, module_type=module_type)
    return None
