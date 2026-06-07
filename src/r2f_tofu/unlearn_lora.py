from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .data import build_paired_loader
from .grad_capture import capture_lora_gradients, save_shape_report, shape_report_from_records
from .losses import ga_gd_loss
from .models import (
    add_lora,
    get_model_device,
    load_causal_lm,
    load_tokenizer,
    set_lora_trainable_filter,
)
from .utils import (
    accumulation_group_size,
    cuda_memory_summary,
    ensure_dir,
    is_accumulation_boundary,
    move_to_device,
    set_seed,
    write_json,
)


def train_lora_baseline(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))

    target_model = str(deep_get(cfg, "paths.target_model"))
    forget_file = str(deep_get(cfg, "paths.forget_file"))
    retain_file = str(deep_get(cfg, "paths.retain_file"))
    target_modules = list(deep_get(cfg, "unlearning.target_modules"))
    train_layers = deep_get(cfg, "unlearning.train_layers")
    train_modules = deep_get(cfg, "unlearning.train_modules")
    max_steps = int(deep_get(cfg, "unlearning.max_steps", 512))
    max_forget_samples = deep_get(cfg, "decoder_samples.max_forget_samples")
    batch_size = int(deep_get(cfg, "unlearning.batch_size", 1))
    max_length = int(deep_get(cfg, "model.max_length", 1024))
    grad_accum_steps = max(1, int(deep_get(cfg, "unlearning.grad_accum_steps", 1)))

    output_dir = ensure_dir(Path(deep_get(cfg, "paths.output_dir")) / "lora_gagd_3b")
    adapter_dir = ensure_dir(output_dir / "adapter")

    tokenizer = load_tokenizer(
        target_model,
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
    )
    loader = build_paired_loader(
        tokenizer=tokenizer,
        forget_file=forget_file,
        retain_file=retain_file,
        max_length=max_length,
        batch_size=batch_size,
        max_forget_samples=int(max_forget_samples) if max_forget_samples is not None else None,
        seed=int(cfg.get("seed", 42)),
    )

    base = load_causal_lm(
        target_model,
        dtype_name=str(deep_get(cfg, "model.dtype", "bfloat16")),
        device_map=deep_get(cfg, "model.device_map", "auto"),
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
        attn_implementation=deep_get(cfg, "model.attn_implementation"),
    )
    model = add_lora(
        base,
        target_modules=target_modules,
        r=int(deep_get(cfg, "lora.r", 8)),
        lora_alpha=int(deep_get(cfg, "lora.alpha", 16)),
        lora_dropout=float(deep_get(cfg, "lora.dropout", 0.0)),
    )
    set_lora_trainable_filter(
        model,
        target_modules,
        train_layers=train_layers,
        train_modules=train_modules,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA parameters are trainable")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(deep_get(cfg, "unlearning.lora_learning_rate", 1e-4)),
    )

    device = get_model_device(model)
    losses: list[dict[str, float]] = []
    shape_report_written = False
    model.train()
    total_steps = min(len(loader), max_steps)
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="LoRA-GA+GD-3B"), start=1):
        if step > max_steps:
            break
        batch = move_to_device(batch, device)
        loss, metrics = ga_gd_loss(
            model,
            batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        group_size = accumulation_group_size(step, total_steps, grad_accum_steps)
        (loss / group_size).backward()
        if not shape_report_written:
            lora_records = capture_lora_gradients(
                model,
                target_modules=target_modules,
                train_layers=train_layers,
                train_modules=train_modules,
            )
            save_shape_report(
                output_dir / "lora_shape_report.json",
                shape_report_from_records(lora_records),
            )
            shape_report_written = True
        if is_accumulation_boundary(step, total_steps, grad_accum_steps):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        metrics["step"] = step
        metrics["optimizer_steps"] = optimizer_steps
        metrics["grad_accum_group_size"] = group_size
        losses.append(metrics)

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    stats = {
        "adapter_dir": str(adapter_dir),
        "steps": len(losses),
        "optimizer_steps": optimizer_steps,
        "grad_accum_steps": grad_accum_steps,
        "last_loss": losses[-1] if losses else {},
        "losses_tail": losses[-20:],
        "target_modules": target_modules,
        "train_layers": train_layers,
        "smoke": smoke,
        **cuda_memory_summary(),
    }
    write_json(output_dir / "train_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    stats = train_lora_baseline(cfg, smoke=args.smoke)
    print(stats)


if __name__ == "__main__":
    main()
