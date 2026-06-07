from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .data import build_paired_loader
from .grad_capture import (
    DecoderSampleWriter,
    capture_dense_gradients,
    capture_lora_gradients,
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
from .utils import cuda_memory_summary, ensure_parent, move_to_device, set_seed


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
    coords_per_module = int(deep_get(cfg, "decoder_samples.coords_per_module", 8192))
    batch_size = int(deep_get(cfg, "unlearning.batch_size", 1))
    max_length = int(deep_get(cfg, "model.max_length", 1024))

    tokenizer = load_tokenizer(source_model, trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)))
    loader = build_paired_loader(
        tokenizer=tokenizer,
        forget_file=forget_file,
        retain_file=retain_file,
        max_length=max_length,
        batch_size=batch_size,
        max_forget_samples=max_forget_samples,
        seed=int(cfg.get("seed", 42)),
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
    set_lora_trainable_filter(lora_model, target_modules, train_layers=train_layers, train_modules=train_modules)
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
    dense_optimizer = torch.optim.AdamW(
        [p for p in dense_model.parameters() if p.requires_grad],
        lr=float(deep_get(cfg, "unlearning.learning_rate", 3e-5)),
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
            "smoke": smoke,
        },
        flush_every_steps=int(deep_get(cfg, "decoder_samples.flush_every_steps", 8)),
    )

    lora_device = get_model_device(lora_model)
    dense_device = get_model_device(dense_model)
    losses: list[dict[str, float]] = []
    shape_report_written = False

    lora_model.train()
    dense_model.train()
    for step, batch in enumerate(tqdm(loader, desc="1B paired gradient capture"), start=1):
        if step > max_steps:
            break

        lora_batch = move_to_device(batch, lora_device)
        lora_optimizer.zero_grad(set_to_none=True)
        lora_loss, lora_metrics = ga_gd_loss(
            lora_model,
            lora_batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        lora_loss.backward()
        lora_records = capture_lora_gradients(
            lora_model,
            target_modules=target_modules,
            train_layers=train_layers,
            train_modules=train_modules,
        )

        dense_batch = move_to_device(batch, dense_device)
        dense_optimizer.zero_grad(set_to_none=True)
        dense_loss, dense_metrics = ga_gd_loss(
            dense_model,
            dense_batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        dense_loss.backward()
        dense_records = capture_dense_gradients(dense_params)

        samples = sample_paired_gradients(
            lora_records,
            dense_records,
            coords_per_module=coords_per_module,
            num_layers=num_layers,
            seed=int(cfg.get("seed", 42)) + step * 7919,
        )
        writer.add(samples)

        if not shape_report_written:
            shape_report = shape_report_from_records(lora_records, dense_records)
            save_shape_report(Path(output_path).parent / "llama1b_gradient_shape_report.json", shape_report)
            shape_report_written = True

        torch.nn.utils.clip_grad_norm_(lora_trainable, 1.0)
        torch.nn.utils.clip_grad_norm_([p for p in dense_model.parameters() if p.requires_grad], 1.0)
        lora_optimizer.step()
        dense_optimizer.step()

        metrics = {
            "step": step,
            "lora_loss": lora_metrics["loss"],
            "lora_forget_ce": lora_metrics["forget_ce"],
            "lora_retain_ce": lora_metrics["retain_ce"],
            "dense_loss": dense_metrics["loss"],
            "dense_forget_ce": dense_metrics["forget_ce"],
            "dense_retain_ce": dense_metrics["retain_ce"],
            "samples_total": writer.total_samples,
        }
        losses.append(metrics)

    writer.close()
    stats = {
        "steps": len(losses),
        "total_samples": writer.total_samples,
        "output_path": str(output_path),
        "last_loss": losses[-1] if losses else {},
        "losses_tail": losses[-20:],
        "target_modules": target_modules,
        "train_layers": train_layers,
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
