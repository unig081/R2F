from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .data import build_paired_loader
from .grad_capture import (
    DECODER_FEATURE_VERSION,
    DecoderSampleWriter,
    capture_dense_gradients,
    capture_lora_gradients,
    find_lora_target_modules,
    lora_scale_from_module,
    sample_paired_gradients,
    save_shape_report,
    shape_report_from_records,
)
from .losses import ga_gd_loss
from .models import (
    add_lora,
    get_model_device,
    infer_num_layers,
    load_causal_lm,
    load_tokenizer,
    set_dense_trainable,
    set_lora_trainable_filter,
)
from .utils import (
    accumulation_group_size,
    cuda_memory_summary,
    ensure_parent,
    is_accumulation_boundary,
    move_to_device,
    set_seed,
)


def _sync_dense_params_from_lora_state(
    lora_modules: dict[Any, torch.nn.Module],
    dense_params: dict[Any, torch.nn.Parameter],
) -> None:
    with torch.no_grad():
        for key, param in dense_params.items():
            if key not in lora_modules:
                continue
            module = lora_modules[key]
            base_layer = getattr(module, "base_layer", module)
            if not hasattr(base_layer, "weight"):
                raise ValueError(f"LoRA module {key.as_string()} does not expose a base weight")
            base = base_layer.weight.detach().to(device=param.device, dtype=torch.float32)
            a = module.lora_A["default"].weight.detach().to(device=param.device, dtype=torch.float32)
            b = module.lora_B["default"].weight.detach().to(device=param.device, dtype=torch.float32)
            merged = torch.addmm(base, b, a, alpha=lora_scale_from_module(module))
            param.data.copy_(merged.to(dtype=param.dtype))


def capture_1b_decoder_samples(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))

    source_model = str(deep_get(cfg, "paths.source_model"))
    forget_file = str(deep_get(cfg, "paths.forget_file"))
    retain_file = str(deep_get(cfg, "paths.retain_file"))
    target_modules = list(deep_get(cfg, "unlearning.target_modules"))
    train_layers = deep_get(cfg, "unlearning.train_layers")
    train_modules = deep_get(cfg, "unlearning.train_modules")
    max_steps = int(deep_get(cfg, "unlearning.max_steps", 512))
    max_forget_samples = int(deep_get(cfg, "decoder_samples.max_forget_samples", max_steps))
    max_retain_samples = deep_get(cfg, "unlearning.max_retain_samples")
    coords_per_module = int(deep_get(cfg, "decoder_samples.coords_per_module", 8192))
    projection_ridge = float(deep_get(cfg, "decoder.projection_ridge", 1e-4))
    deterministic_capture = bool(deep_get(cfg, "decoder_samples.deterministic_capture", True))
    batch_size = int(deep_get(cfg, "unlearning.batch_size", 1))
    max_length = int(deep_get(cfg, "model.max_length", 1024))
    grad_accum_steps = max(1, int(deep_get(cfg, "unlearning.grad_accum_steps", 1)))

    tokenizer = load_tokenizer(
        source_model,
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
    )
    loader = build_paired_loader(
        tokenizer=tokenizer,
        forget_file=forget_file,
        retain_file=retain_file,
        max_length=max_length,
        batch_size=batch_size,
        max_forget_samples=max_forget_samples,
        seed=int(cfg.get("seed", 42)),
        max_retain_samples=int(max_retain_samples) if max_retain_samples is not None else None,
    )

    common_model_kwargs = {
        "dtype_name": str(deep_get(cfg, "model.dtype", "bfloat16")),
        "device_map": deep_get(cfg, "model.device_map", "auto"),
        "trust_remote_code": bool(deep_get(cfg, "model.trust_remote_code", True)),
        "attn_implementation": deep_get(cfg, "model.attn_implementation"),
    }

    lora_base = load_causal_lm(source_model, **common_model_kwargs)
    lora_model = add_lora(
        lora_base,
        target_modules=target_modules,
        r=int(deep_get(cfg, "lora.r", 8)),
        lora_alpha=int(deep_get(cfg, "lora.alpha", 16)),
        lora_dropout=float(deep_get(cfg, "lora.dropout", 0.0)),
    )
    set_lora_trainable_filter(
        lora_model,
        target_modules,
        train_layers=train_layers,
        train_modules=train_modules,
    )
    lora_modules = find_lora_target_modules(
        lora_model,
        target_modules=target_modules,
        train_layers=train_layers,
        train_modules=train_modules,
    )
    lora_trainable = [p for p in lora_model.parameters() if p.requires_grad]
    lora_optimizer = torch.optim.AdamW(
        lora_trainable,
        lr=float(deep_get(cfg, "unlearning.lora_learning_rate", 1e-4)),
    )

    dense_model = load_causal_lm(source_model, **common_model_kwargs)
    dense_params = set_dense_trainable(
        dense_model,
        target_modules=target_modules,
        train_layers=train_layers,
        train_modules=train_modules,
    )

    num_layers = infer_num_layers(dense_model)
    output_path = ensure_parent(deep_get(cfg, "decoder_samples.output_path"))
    writer = DecoderSampleWriter(
        output_path=output_path,
        metadata={
            "source_model": source_model,
            "forget_file": forget_file,
            "retain_file": retain_file,
            "target_modules": target_modules,
            "train_layers": train_layers,
            "rank": int(deep_get(cfg, "lora.r", 8)),
            "num_layers": num_layers,
            "coords_per_module": coords_per_module,
            "pairing_mode": "same_lora_state_merged_dense",
            "feature_version": DECODER_FEATURE_VERSION,
            "projection_ridge": projection_ridge,
            "deterministic_capture": deterministic_capture,
            "smoke": smoke,
        },
        flush_every_steps=int(deep_get(cfg, "decoder_samples.flush_every_steps", 8)),
    )

    lora_device = get_model_device(lora_model)
    dense_device = get_model_device(dense_model)
    losses: list[dict[str, float]] = []
    shape_report_written = False
    group_lora_metrics: list[dict[str, float]] = []
    group_dense_metrics: list[dict[str, float]] = []
    total_micro_steps = min(len(loader), max_steps)
    optimizer_steps = 0

    if deterministic_capture:
        lora_model.eval()
        dense_model.eval()
    else:
        lora_model.train()
        dense_model.train()
    lora_optimizer.zero_grad(set_to_none=True)
    dense_model.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="1B paired gradient capture"), start=1):
        if step > max_steps:
            break

        if not group_lora_metrics:
            _sync_dense_params_from_lora_state(lora_modules, dense_params)
            dense_model.zero_grad(set_to_none=True)

        lora_batch = move_to_device(batch, lora_device)
        lora_loss, lora_metrics = ga_gd_loss(
            lora_model,
            lora_batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        group_size = accumulation_group_size(step, total_micro_steps, grad_accum_steps)
        (lora_loss / group_size).backward()
        group_lora_metrics.append(lora_metrics)

        dense_batch = move_to_device(batch, dense_device)
        dense_loss, dense_metrics = ga_gd_loss(
            dense_model,
            dense_batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        (dense_loss / group_size).backward()
        group_dense_metrics.append(dense_metrics)

        if not is_accumulation_boundary(step, total_micro_steps, grad_accum_steps):
            continue

        lora_records = capture_lora_gradients(
            lora_model,
            target_modules=target_modules,
            train_layers=train_layers,
            train_modules=train_modules,
        )
        dense_records = capture_dense_gradients(dense_params)

        samples = sample_paired_gradients(
            lora_records,
            dense_records,
            coords_per_module=coords_per_module,
            num_layers=num_layers,
            seed=int(cfg.get("seed", 42)) + (optimizer_steps + 1) * 7919,
            projection_ridge=projection_ridge,
        )
        writer.add(samples)

        if not shape_report_written:
            shape_report = shape_report_from_records(lora_records, dense_records)
            save_shape_report(
                Path(output_path).parent / "llama1b_gradient_shape_report.json",
                shape_report,
            )
            shape_report_written = True

        torch.nn.utils.clip_grad_norm_(lora_trainable, 1.0)
        lora_optimizer.step()
        lora_optimizer.zero_grad(set_to_none=True)
        dense_model.zero_grad(set_to_none=True)
        optimizer_steps += 1

        def mean_metric(rows: list[dict[str, float]], key: str) -> float:
            return float(sum(row[key] for row in rows) / max(len(rows), 1))

        metrics = {
            "optimizer_step": optimizer_steps,
            "micro_step_end": step,
            "grad_accum_group_size": len(group_lora_metrics),
            "lora_loss": mean_metric(group_lora_metrics, "loss"),
            "lora_forget_ce": mean_metric(group_lora_metrics, "forget_ce"),
            "lora_retain_ce": mean_metric(group_lora_metrics, "retain_ce"),
            "dense_loss": mean_metric(group_dense_metrics, "loss"),
            "dense_forget_ce": mean_metric(group_dense_metrics, "forget_ce"),
            "dense_retain_ce": mean_metric(group_dense_metrics, "retain_ce"),
            "samples_total": writer.total_samples,
        }
        losses.append(metrics)
        group_lora_metrics.clear()
        group_dense_metrics.clear()

    writer.close()
    stats = {
        "optimizer_steps": optimizer_steps,
        "micro_steps": total_micro_steps,
        "grad_accum_steps": grad_accum_steps,
        "total_samples": writer.total_samples,
        "output_path": str(output_path),
        "last_loss": losses[-1] if losses else {},
        "losses_tail": losses[-20:],
        "target_modules": target_modules,
        "train_layers": train_layers,
        "pairing_mode": "same_lora_state_merged_dense",
        "feature_version": DECODER_FEATURE_VERSION,
        "projection_ridge": projection_ridge,
        "deterministic_capture": deterministic_capture,
        "smoke": smoke,
        **cuda_memory_summary(),
    }
    writer.write_stats(deep_get(cfg, "decoder_samples.stats_path"), stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    stats = capture_1b_decoder_samples(cfg, smoke=args.smoke)
    print(stats)


if __name__ == "__main__":
    main()
