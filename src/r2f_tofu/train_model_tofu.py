from __future__ import annotations

import argparse
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import deep_get, load_config
from .data import DataCollatorForR2F, load_tofu_file
from .models import load_tokenizer
from .utils import dtype_from_name, ensure_dir, set_seed, write_json


class TOFUSupervisedDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        if not rows:
            raise ValueError("TOFU supervised training dataset is empty")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def _world_size() -> int:
    for name in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(name)
        if value and value.isdigit():
            return max(int(value), 1)
    if torch.cuda.is_available():
        return max(torch.cuda.device_count(), 1)
    return 1


def _resolve_base_model(cfg: dict[str, Any]) -> str:
    configured = deep_get(cfg, "train.model_name_or_path") or deep_get(cfg, "paths.model_path")
    if configured:
        return str(configured)
    return str(deep_get(cfg, "paths.source_model"))


def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    configured = deep_get(cfg, "train.output_dir")
    if configured:
        return Path(str(configured))
    output_root = Path(str(deep_get(cfg, "paths.output_dir", "results")))
    family = str(deep_get(cfg, "model.family", deep_get(cfg, "model_family", "llama")))
    split = str(deep_get(cfg, "train.tofu_split", "full"))
    return output_root / "tofu_finetune" / f"{family}_{split}"


def _load_tofu_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    data_file = deep_get(cfg, "train.data_file")
    split = str(deep_get(cfg, "train.tofu_split", "full"))
    limit = deep_get(cfg, "train.max_train_samples")
    limit_int = int(limit) if limit is not None else None
    if data_file:
        return load_tofu_file(str(data_file), limit=limit_int)

    data_dir = deep_get(cfg, "train.tofu_data_dir") or deep_get(cfg, "evaluation.tofu_data_dir")
    if data_dir:
        candidate = Path(str(data_dir)) / f"{split}.json"
        if candidate.exists():
            return load_tofu_file(candidate, limit=limit_int)

    from datasets import load_dataset

    dataset = load_dataset("locuslab/TOFU", name=split, split="train")
    rows = [{"question": str(row["question"]), "answer": str(row["answer"])} for row in dataset]
    return rows[:limit_int] if limit_int is not None else rows


def _training_args_kwargs(cfg: dict[str, Any], output_dir: Path, dataset_len: int) -> dict[str, Any]:
    from transformers import TrainingArguments

    per_device_train_batch_size = int(deep_get(cfg, "train.per_device_train_batch_size", 4))
    gradient_accumulation_steps = int(deep_get(cfg, "train.gradient_accumulation_steps", 4))
    warmup_steps = deep_get(cfg, "train.warmup_steps")
    warmup_epochs = deep_get(cfg, "train.warmup_epochs")
    if warmup_steps is None and warmup_epochs is not None:
        denom = max(per_device_train_batch_size * gradient_accumulation_steps * _world_size(), 1)
        warmup_steps = int((float(warmup_epochs) * dataset_len) // denom)

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": bool(deep_get(cfg, "train.overwrite_output_dir", False)),
        "do_train": True,
        "do_eval": bool(deep_get(cfg, "train.do_eval", False)),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": int(deep_get(cfg, "train.per_device_eval_batch_size", 16)),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(deep_get(cfg, "train.learning_rate", 1e-5)),
        "weight_decay": float(deep_get(cfg, "train.weight_decay", 0.01)),
        "adam_beta1": float(deep_get(cfg, "train.adam_beta1", 0.9)),
        "adam_beta2": float(deep_get(cfg, "train.adam_beta2", 0.999)),
        "adam_epsilon": float(deep_get(cfg, "train.adam_epsilon", 1e-8)),
        "max_grad_norm": float(deep_get(cfg, "train.max_grad_norm", 1.0)),
        "num_train_epochs": float(deep_get(cfg, "train.num_train_epochs", 5)),
        "max_steps": int(deep_get(cfg, "train.max_steps", -1)),
        "lr_scheduler_type": str(deep_get(cfg, "train.lr_scheduler_type", "linear")),
        "warmup_steps": int(warmup_steps or 0),
        "logging_steps": int(deep_get(cfg, "train.logging_steps", 5)),
        "logging_strategy": str(deep_get(cfg, "train.logging_strategy", "steps")),
        "save_strategy": str(deep_get(cfg, "train.save_strategy", "no")),
        "save_steps": int(deep_get(cfg, "train.save_steps", 500)),
        "save_total_limit": deep_get(cfg, "train.save_total_limit"),
        "save_safetensors": bool(deep_get(cfg, "train.save_safetensors", True)),
        "seed": int(deep_get(cfg, "seed", deep_get(cfg, "train.seed", 0))),
        "bf16": bool(deep_get(cfg, "train.bf16", True)),
        "bf16_full_eval": bool(deep_get(cfg, "train.bf16_full_eval", True)),
        "fp16": bool(deep_get(cfg, "train.fp16", False)),
        "gradient_checkpointing": bool(deep_get(cfg, "train.gradient_checkpointing", True)),
        "optim": str(deep_get(cfg, "train.optim", "paged_adamw_32bit")),
        "remove_unused_columns": bool(deep_get(cfg, "train.remove_unused_columns", True)),
        "dataloader_num_workers": int(deep_get(cfg, "train.dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(deep_get(cfg, "train.dataloader_pin_memory", True)),
        "group_by_length": bool(deep_get(cfg, "train.group_by_length", False)),
        "report_to": deep_get(cfg, "train.report_to", ["tensorboard"]),
        "run_name": str(deep_get(cfg, "train.run_name") or output_dir),
        "load_best_model_at_end": bool(deep_get(cfg, "train.load_best_model_at_end", False)),
        "save_only_model": bool(deep_get(cfg, "train.save_only_model", True)),
        "ddp_find_unused_parameters": deep_get(cfg, "train.ddp_find_unused_parameters", True),
        "deepspeed": deep_get(cfg, "train.deepspeed"),
    }
    eval_strategy = deep_get(cfg, "train.eval_strategy", deep_get(cfg, "train.evaluation_strategy"))
    if eval_strategy is not None:
        kwargs["eval_strategy"] = str(eval_strategy)
        kwargs["evaluation_strategy"] = str(eval_strategy)
    eval_steps = deep_get(cfg, "train.eval_steps")
    if eval_steps is not None:
        kwargs["eval_steps"] = int(eval_steps)

    valid = {field.name for field in fields(TrainingArguments)}
    return {key: value for key, value in kwargs.items() if key in valid and value is not None}


def train_model_tofu(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(deep_get(cfg, "seed", deep_get(cfg, "train.seed", 0))))

    model_family = str(deep_get(cfg, "model.family", deep_get(cfg, "model_family", "llama")))
    model_name_or_path = _resolve_base_model(cfg)
    output_dir = ensure_dir(_resolve_output_dir(cfg))
    train_rows = _load_tofu_rows(cfg)
    eval_rows = train_rows[: int(deep_get(cfg, "train.max_eval_samples", 0) or 0)]
    train_dataset = TOFUSupervisedDataset(train_rows)
    eval_dataset = TOFUSupervisedDataset(eval_rows) if eval_rows else None

    tokenizer = load_tokenizer(
        model_name_or_path,
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
    )
    collator = DataCollatorForR2F(
        tokenizer=tokenizer,
        max_length=int(deep_get(cfg, "model.max_length", deep_get(cfg, "train.max_length", 512))),
        model_family=model_family,
    )

    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name_or_path,
        "torch_dtype": dtype_from_name(str(deep_get(cfg, "model.dtype", "bfloat16"))),
        "trust_remote_code": bool(deep_get(cfg, "model.trust_remote_code", True)),
    }
    device_map = deep_get(cfg, "model.device_map")
    if device_map:
        model_kwargs["device_map"] = device_map
    attn_implementation = deep_get(cfg, "model.attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if bool(deep_get(cfg, "train.gradient_checkpointing", True)):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    args = TrainingArguments(**_training_args_kwargs(cfg, output_dir, len(train_dataset)))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / max(args.per_device_train_batch_size * args.gradient_accumulation_steps * _world_size(), 1)
    )
    try:
        training_args = args.to_dict()
    except Exception:
        training_args = dict(getattr(args, "__dict__", {}))
    stats = {
        "model_family": model_family,
        "model_name_or_path": model_name_or_path,
        "output_dir": str(output_dir),
        "tofu_split": str(deep_get(cfg, "train.tofu_split", "full")),
        "train_samples": len(train_dataset),
        "effective_global_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * _world_size(),
        "steps_per_epoch_estimate": steps_per_epoch,
        "train_metrics": train_result.metrics,
        "training_args": training_args,
    }
    write_json(output_dir / "tofu_train_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_model_tofu.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    stats = train_model_tofu(cfg)
    print(stats)


if __name__ == "__main__":
    main()
