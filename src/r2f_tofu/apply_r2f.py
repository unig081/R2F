from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .data import build_paired_loader
from .decoder import GradientDecoder
from .grad_capture import capture_lora_gradients, save_shape_report, shape_report_from_records
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
from .module_keys import ModuleKey
from .utils import (
    accumulation_group_size,
    cuda_memory_summary,
    ensure_dir,
    is_accumulation_boundary,
    move_to_device,
    set_seed,
    write_json,
)


def _load_decoder(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GradientDecoder, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if ckpt.get("format") != "r2f_gradient_decoder_v1":
        raise ValueError(f"Unsupported decoder checkpoint: {checkpoint_path}")
    model_cfg = ckpt["model_config"]
    decoder = GradientDecoder(
        rank=int(model_cfg["rank"]),
        num_modules=int(model_cfg.get("num_modules", 7)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        module_embedding_dim=int(model_cfg.get("module_embedding_dim", 16)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
    )
    decoder.load_state_dict(ckpt["model_state"])
    decoder.to(device)
    decoder.eval()
    return decoder, ckpt


def _iter_coordinate_blocks(
    out_dim: int,
    in_dim: int,
    block_rows: int,
    max_elements: int = 262144,
) -> Iterator[tuple[int, int, int, int]]:
    for row_start in range(0, out_dim, block_rows):
        row_end = min(row_start + block_rows, out_dim)
        rows = row_end - row_start
        block_cols = max(1, min(in_dim, max_elements // max(rows, 1)))
        for col_start in range(0, in_dim, block_cols):
            col_end = min(col_start + block_cols, in_dim)
            yield row_start, row_end, col_start, col_end


def _resolve_grad_rms(
    key: ModuleKey,
    target_num_layers: int,
    checkpoint: dict[str, Any],
) -> float:
    normalization = checkpoint.get("normalization", {})
    layer_stats = normalization.get("layer_module_grad_rms", {})
    module_stats = normalization.get("module_grad_rms", {})
    source_num_layers = int(
        checkpoint.get("metadata", {}).get("num_layers")
        or checkpoint["model_config"].get("num_layers", 1)
    )
    mapped_layer = round(
        key.layer_idx / max(target_num_layers - 1, 1) * max(source_num_layers - 1, 1)
    )
    mapped_key = f"{mapped_layer}.{key.module_type}"
    if mapped_key in layer_stats:
        return float(layer_stats[mapped_key])
    if key.module_type in module_stats:
        return float(module_stats[key.module_type])
    return float(normalization.get("global_grad_rms", 1.0))


def _predict_and_update_param(
    param: torch.nn.Parameter,
    key: ModuleKey,
    lora: dict[str, torch.Tensor],
    decoder: GradientDecoder,
    checkpoint: dict[str, Any],
    eta: float,
    target_num_layers: int,
    block_rows: int,
    gradient_path: Path,
) -> dict[str, Any]:
    a = lora["A"]
    b = lora["B"]
    da = lora["dA"]
    db = lora["dB"]
    out_dim, in_dim = tuple(param.shape)
    if a.shape[1] != in_dim or b.shape[0] != out_dim:
        raise ValueError(
            f"{key.as_string()} LoRA/dense shape mismatch: "
            f"A={tuple(a.shape)} B={tuple(b.shape)} W={tuple(param.shape)}"
        )

    device = next(decoder.parameters()).device
    grad_rms = _resolve_grad_rms(key, target_num_layers, checkpoint)
    grad_cpu = torch.empty((out_dim, in_dim), dtype=torch.float32)
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0

    with torch.no_grad():
        for row_start, row_end, col_start, col_end in _iter_coordinate_blocks(
            out_dim, in_dim, block_rows=block_rows
        ):
            rows = torch.arange(row_start, row_end, dtype=torch.long)
            cols = torch.arange(col_start, col_end, dtype=torch.long)
            o_idx = rows.repeat_interleave(len(cols))
            i_idx = cols.repeat(len(rows))
            batch = {
                "A_col": a[:, i_idx].transpose(0, 1).contiguous(),
                "B_row": b[o_idx, :].contiguous(),
                "dA_col": da[:, i_idx].transpose(0, 1).contiguous(),
                "dB_row": db[o_idx, :].contiguous(),
                "layer_idx": torch.full((len(o_idx),), key.layer_idx, dtype=torch.long),
                "module_id": torch.full((len(o_idx),), key.module_id, dtype=torch.long),
                "num_layers": torch.full((len(o_idx),), target_num_layers, dtype=torch.long),
            }
            batch = move_to_device(batch, device)
            pred_norm = decoder(batch)
            d_w = (pred_norm.float().cpu() * grad_rms).reshape(
                row_end - row_start,
                col_end - col_start,
            )
            dense_delta = -eta * d_w
            grad_cpu[row_start:row_end, col_start:col_end] = d_w
            param.data[row_start:row_end, col_start:col_end].add_(
                dense_delta.to(device=param.device, dtype=param.dtype)
            )
            total_abs += float(d_w.abs().sum().item())
            total_sq += float(d_w.pow(2).sum().item())
            total_n += int(d_w.numel())

    torch.save(
        {
            "format": "r2f_predicted_dense_gradient_shard_v1",
            "module": key.as_string(),
            "eta": eta,
            "dW_hat": grad_cpu,
            "delta_formula": "dense_delta = -eta * dW_hat",
        },
        gradient_path,
    )
    return {
        "module": key.as_string(),
        "shape": [out_dim, in_dim],
        "grad_rms_used": grad_rms,
        "gradient_path": str(gradient_path),
        "delta_formula": "dense_delta = -eta * dW_hat",
        "pred_abs_mean": total_abs / max(total_n, 1),
        "pred_rms": (total_sq / max(total_n, 1)) ** 0.5,
    }


def _train_and_capture_target_lora_gradients(
    cfg: dict[str, Any],
) -> tuple[dict[ModuleKey, dict[str, torch.Tensor]], int, dict[str, Any]]:
    target_model = str(deep_get(cfg, "paths.target_model"))
    forget_file = str(deep_get(cfg, "paths.forget_file"))
    retain_file = str(deep_get(cfg, "paths.retain_file"))
    target_modules = list(deep_get(cfg, "unlearning.target_modules"))
    train_layers = deep_get(cfg, "unlearning.train_layers")
    train_modules = deep_get(cfg, "unlearning.train_modules")
    lora_steps = int(deep_get(cfg, "r2f.lora_steps", deep_get(cfg, "unlearning.max_steps", 512)))
    capture_steps = int(
        deep_get(cfg, "r2f.gradient_capture_steps", deep_get(cfg, "unlearning.grad_accum_steps", 1))
    )
    grad_accum_steps = max(1, int(deep_get(cfg, "unlearning.grad_accum_steps", 1)))
    if lora_steps < 1:
        raise ValueError("r2f.lora_steps must be >= 1")
    if capture_steps < 1:
        raise ValueError("r2f.gradient_capture_steps must be >= 1")

    tokenizer = load_tokenizer(
        target_model,
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
    )
    train_loader = build_paired_loader(
        tokenizer=tokenizer,
        forget_file=forget_file,
        retain_file=retain_file,
        max_length=int(deep_get(cfg, "model.max_length", 1024)),
        batch_size=int(deep_get(cfg, "unlearning.batch_size", 1)),
        max_forget_samples=lora_steps,
        seed=int(cfg.get("seed", 42)) + 17,
    )
    base = load_causal_lm(
        target_model,
        dtype_name=str(deep_get(cfg, "model.dtype", "bfloat16")),
        device_map=deep_get(cfg, "model.device_map", "auto"),
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
        attn_implementation=deep_get(cfg, "model.attn_implementation"),
    )
    lora_model = add_lora(
        base,
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
    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No 3B LoRA parameters are trainable for R2F capture")
    device = get_model_device(lora_model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(deep_get(cfg, "unlearning.lora_learning_rate", 1e-4)),
    )
    lora_model.train()
    total_train_steps = min(len(train_loader), lora_steps)
    optimizer_steps = 0
    train_losses: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(train_loader, desc="3B LoRA GA+GD for R2F"), start=1):
        if step > lora_steps:
            break
        batch = move_to_device(batch, device)
        loss, _metrics = ga_gd_loss(
            lora_model,
            batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        group_size = accumulation_group_size(step, total_train_steps, grad_accum_steps)
        (loss / group_size).backward()
        if is_accumulation_boundary(step, total_train_steps, grad_accum_steps):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        train_losses.append(
            {
                "step": step,
                "optimizer_steps": optimizer_steps,
                "grad_accum_group_size": group_size,
                **_metrics,
            }
        )

    target_num_layers = infer_num_layers(lora_model)
    output_dir = ensure_dir(deep_get(cfg, "r2f.output_dir"))
    adapter_dir = ensure_dir(output_dir / "lora_gagd_3b_for_r2f_adapter")
    lora_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    capture_loader = build_paired_loader(
        tokenizer=tokenizer,
        forget_file=forget_file,
        retain_file=retain_file,
        max_length=int(deep_get(cfg, "model.max_length", 1024)),
        batch_size=int(deep_get(cfg, "unlearning.batch_size", 1)),
        max_forget_samples=capture_steps,
        seed=int(cfg.get("seed", 42)) + 1009,
    )
    total_capture_steps = min(len(capture_loader), capture_steps)
    capture_losses: list[dict[str, float]] = []
    lora_model.train()
    lora_model.zero_grad(set_to_none=True)
    capture_iter = tqdm(capture_loader, desc="3B LoRA gradient capture for decoder")
    for step, batch in enumerate(capture_iter, start=1):
        if step > capture_steps:
            break
        batch = move_to_device(batch, device)
        loss, metrics = ga_gd_loss(
            lora_model,
            batch,
            gamma=float(deep_get(cfg, "unlearning.gamma", 1.0)),
            alpha=float(deep_get(cfg, "unlearning.alpha", 1.0)),
        )
        (loss / max(total_capture_steps, 1)).backward()
        metrics["step"] = step
        capture_losses.append(metrics)

    lora_records = capture_lora_gradients(
        lora_model,
        target_modules=target_modules,
        train_layers=train_layers,
        train_modules=train_modules,
    )
    save_shape_report(
        output_dir / "llama3b_lora_gradient_shape_report.json",
        shape_report_from_records(lora_records),
    )

    stats = {
        "target_model": target_model,
        "adapter_dir": str(adapter_dir),
        "lora_train_micro_steps": total_train_steps,
        "lora_optimizer_steps": optimizer_steps,
        "grad_accum_steps": grad_accum_steps,
        "gradient_capture_steps": total_capture_steps,
        "train_losses_tail": train_losses[-20:],
        "capture_losses": capture_losses,
        "captured_modules": [
            key.as_string()
            for key in sorted(lora_records, key=lambda x: (x.layer_idx, x.module_type))
        ],
    }
    write_json(output_dir / "lora_gradient_capture_stats.json", stats)
    del lora_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lora_records, target_num_layers, stats


def apply_r2f(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(deep_get(cfg, "r2f.output_dir"))
    decoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, checkpoint = _load_decoder(deep_get(cfg, "decoder.checkpoint_path"), decoder_device)

    (
        lora_records,
        target_num_layers,
        lora_gradient_stats,
    ) = _train_and_capture_target_lora_gradients(cfg)
    eta_grid = [float(x) for x in deep_get(cfg, "r2f.eta_grid", [1e-6])]
    target_model = str(deep_get(cfg, "paths.target_model"))
    target_modules = list(deep_get(cfg, "unlearning.target_modules"))
    train_layers = deep_get(cfg, "unlearning.train_layers")
    train_modules = deep_get(cfg, "unlearning.train_modules")
    block_rows = int(deep_get(cfg, "r2f.block_rows", 512))
    save_updated_model = bool(deep_get(cfg, "r2f.save_updated_model", True))

    eta_stats: list[dict[str, Any]] = []
    tokenizer = load_tokenizer(
        target_model,
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
    )
    for eta in eta_grid:
        eta_tag = f"eta_{eta:.0e}".replace("-", "m")
        eta_dir = ensure_dir(output_dir / eta_tag)
        shard_dir = ensure_dir(eta_dir / "predicted_dense_gradient_shards")

        dense_model = load_causal_lm(
            target_model,
            dtype_name=str(deep_get(cfg, "model.dtype", "bfloat16")),
            device_map=deep_get(cfg, "model.device_map", "auto"),
            trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
            attn_implementation=deep_get(cfg, "model.attn_implementation"),
        )
        dense_params = set_dense_trainable(
            dense_model,
            target_modules=target_modules,
            train_layers=train_layers,
            train_modules=train_modules,
        )
        module_stats: list[dict[str, Any]] = []
        for key, param in tqdm(
            sorted(dense_params.items(), key=lambda item: (item[0].layer_idx, item[0].module_type)),
            desc=f"R2F dense update eta={eta:g}",
        ):
            if key not in lora_records:
                continue
            gradient_path = shard_dir / f"{key.as_string()}.pt"
            module_stats.append(
                _predict_and_update_param(
                    param=param,
                    key=key,
                    lora=lora_records[key],
                    decoder=decoder,
                    checkpoint=checkpoint,
                    eta=eta,
                    target_num_layers=target_num_layers,
                    block_rows=block_rows,
                    gradient_path=gradient_path,
                )
            )

        manifest = {
            "format": "r2f_predicted_dense_gradient_manifest_v1",
            "eta": eta,
            "target_model": target_model,
            "target_num_layers": target_num_layers,
            "delta_formula": "dense_delta = -eta * dW_hat",
            "modules": module_stats,
        }
        torch.save(manifest, eta_dir / "dense_gradient.pt")
        torch.save(manifest, eta_dir / "dense_delta.pt")

        updated_model_dir = eta_dir / "updated_model"
        if save_updated_model:
            dense_model.save_pretrained(updated_model_dir)
            tokenizer.save_pretrained(updated_model_dir)

        stats = {
            "eta": eta,
            "eta_dir": str(eta_dir),
            "dense_gradient_manifest": str(eta_dir / "dense_gradient.pt"),
            "dense_delta_manifest": str(eta_dir / "dense_delta.pt"),
            "updated_model_dir": str(updated_model_dir) if save_updated_model else None,
            "modules_updated": len(module_stats),
            "module_stats": module_stats,
            "smoke": smoke,
            **cuda_memory_summary(),
        }
        write_json(eta_dir / "update_stats.json", stats)
        eta_stats.append(stats)

        del dense_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "output_dir": str(output_dir),
        "lora_gradient_capture": lora_gradient_stats,
        "etas": eta_stats,
        "smoke": smoke,
    }
    write_json(output_dir / "update_stats.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    stats = apply_r2f(cfg, smoke=args.smoke)
    print(stats)


if __name__ == "__main__":
    main()
