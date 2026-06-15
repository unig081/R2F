from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelEndpoint:
    local_subdir: str
    hf_id: str


@dataclass(frozen=True)
class ModelFamily:
    name: str
    display_name: str
    proxy: ModelEndpoint
    target: ModelEndpoint
    target_modules: tuple[str, ...]
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    chat_template_kwargs: dict[str, Any] | None = None


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "llama": ModelFamily(
        name="llama",
        display_name="Llama 3.2",
        proxy=ModelEndpoint(
            local_subdir="model/proxy/llama3.2_1B",
            hf_id="meta-llama/Llama-3.2-1B-Instruct",
        ),
        target=ModelEndpoint(
            local_subdir="model/target/llama3.2_3B",
            hf_id="meta-llama/Llama-3.2-3B-Instruct",
        ),
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        attn_implementation=None,
        chat_template_kwargs={"date_string": "10 Apr 2025"},
    ),
    "phi": ModelFamily(
        name="phi",
        display_name="Phi-4",
        proxy=ModelEndpoint(
            local_subdir="model/proxy/phi4_3B",
            hf_id="microsoft/Phi-4-mini-instruct",
        ),
        target=ModelEndpoint(
            local_subdir="model/target/phi4_14B",
            hf_id="microsoft/phi-4",
        ),
        target_modules=("qkv_proj", "o_proj"),
        attn_implementation=None,
    ),
    "qwen": ModelFamily(
        name="qwen",
        display_name="Qwen3",
        proxy=ModelEndpoint(
            local_subdir="model/proxy/qwen3_1.7B",
            hf_id="Qwen/Qwen3-1.7B",
        ),
        target=ModelEndpoint(
            local_subdir="model/target/qwen3_8B",
            hf_id="Qwen/Qwen3-8B",
        ),
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        attn_implementation=None,
    ),
}


def normalize_family_name(name: str | None) -> str:
    family = (name or "llama").strip().lower()
    aliases = {
        "llama3": "llama",
        "llama3.2": "llama",
        "llama_3_2": "llama",
        "phi4": "phi",
        "qwen3": "qwen",
    }
    family = aliases.get(family, family)
    if family not in MODEL_FAMILIES:
        choices = ", ".join(sorted(MODEL_FAMILIES))
        raise ValueError(f"Unsupported model_family {name!r}; expected one of: {choices}")
    return family


def get_model_family(name: str | None) -> ModelFamily:
    return MODEL_FAMILIES[normalize_family_name(name)]


def default_model_path(root: str | Path, family_name: str | None, role: str) -> str:
    family = get_model_family(family_name)
    endpoint = family.proxy if role == "proxy" else family.target
    return str(Path(root) / endpoint.local_subdir)


def default_train_model_id(family_name: str | None) -> str:
    return get_model_family(family_name).proxy.hf_id


def chat_template_kwargs(family_name: str | None) -> dict[str, Any]:
    return dict(get_model_family(family_name).chat_template_kwargs or {})
