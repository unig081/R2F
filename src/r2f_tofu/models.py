from __future__ import annotations

from typing import Any, Iterable

import torch

from .module_keys import LAYER_RE, ModuleKey, parse_module_key
from .utils import dtype_from_name


def load_tokenizer(model_path: str, trust_remote_code: bool = True) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_path: str,
    dtype_name: str | None = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(dtype_name),
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def add_lora(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> torch.nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def infer_num_layers(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(cfg, attr, None)
        if value is not None:
            return int(value)
    max_layer = -1
    for name, _module in model.named_modules():
        match = LAYER_RE.search(name)
        if match:
            max_layer = max(max_layer, int(match.group(1)))
    if max_layer < 0:
        raise ValueError("Unable to infer number of Transformer layers")
    return max_layer + 1


def should_keep_key(
    key: ModuleKey,
    train_layers: Iterable[int] | None = None,
    train_modules: Iterable[str] | None = None,
) -> bool:
    if train_layers is not None and key.layer_idx not in set(int(x) for x in train_layers):
        return False
    if train_modules is not None and key.module_type not in set(str(x) for x in train_modules):
        return False
    return True


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def set_dense_trainable(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    train_layers: Iterable[int] | None = None,
    train_modules: Iterable[str] | None = None,
) -> dict[ModuleKey, torch.nn.Parameter]:
    freeze_all_parameters(model)
    selected: dict[ModuleKey, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        key = parse_module_key(name, target_modules)
        if key is None or not should_keep_key(key, train_layers, train_modules):
            continue
        param.requires_grad = True
        selected[key] = param
    if not selected:
        raise ValueError("No dense target parameters were selected")
    return selected


def set_lora_trainable_filter(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    train_layers: Iterable[int] | None = None,
    train_modules: Iterable[str] | None = None,
) -> None:
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            continue
        key = parse_module_key(name, target_modules)
        param.requires_grad = bool(key and should_keep_key(key, train_layers, train_modules))


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
