from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .decoder import GradientDecoder, decoder_loss
from .grad_capture import load_decoder_sample_tensors
from .module_keys import ID_TO_MODULE
from .utils import ensure_parent, move_to_device, set_seed, write_json


class TensorDictDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        n = int(next(iter(tensors.values())).shape[0])
        for key, value in tensors.items():
            if int(value.shape[0]) != n:
                raise ValueError(f"Tensor {key} length mismatch: {value.shape[0]} != {n}")
        self.tensors = tensors
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}


def collate_tensor_dict(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _module_grad_rms(samples: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    module_ids = samples["module_id"].long()
    grad_rms = samples["grad_rms"].float()
    for module_id in sorted(module_ids.unique().tolist()):
        mask = module_ids == module_id
        values = grad_rms[mask]
        result[ID_TO_MODULE.get(int(module_id), str(module_id))] = float(values.median().item())
    return result


def _layer_module_grad_rms(samples: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    module_ids = samples["module_id"].long()
    layer_idx = samples["layer_idx"].long()
    grad_rms = samples["grad_rms"].float()
    for layer in sorted(layer_idx.unique().tolist()):
        for module_id in sorted(module_ids.unique().tolist()):
            mask = (layer_idx == layer) & (module_ids == module_id)
            if mask.any():
                module = ID_TO_MODULE.get(int(module_id), str(module_id))
                result[f"{int(layer)}.{module}"] = float(grad_rms[mask].median().item())
    return result


def _safe_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = torch.sqrt(pred_centered.pow(2).sum() * target_centered.pow(2).sum())
    if float(denom.item()) == 0.0:
        return 0.0
    return float((pred_centered * target_centered).sum().div(denom).item())


def _prediction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sign_threshold: float = 0.0,
) -> dict[str, float]:
    pred = pred.float().detach()
    target = target.float().detach()
    diff = pred - target
    mse = diff.pow(2).mean()
    target_var = target.var(unbiased=False)
    r2 = 0.0 if float(target_var.item()) == 0.0 else 1.0 - float(mse.div(target_var).item())
    strong_mask = target.abs() >= float(sign_threshold)
    if strong_mask.any():
        sign_acc_strong = (
            torch.sign(pred[strong_mask]) == torch.sign(target[strong_mask])
        ).float().mean()
    else:
        sign_acc_strong = torch.tensor(0.0, device=pred.device)
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(diff.abs().mean().item()),
        "corr": _safe_corrcoef(pred, target),
        "r2": r2,
        "pred_mean": float(pred.mean().item()),
        "pred_std": float(pred.std(unbiased=False).item()),
        "target_mean": float(target.mean().item()),
        "target_std": float(target.std(unbiased=False).item()),
        "sign_acc": float((torch.sign(pred) == torch.sign(target)).float().mean().item()),
        "sign_acc_strong": float(sign_acc_strong.item()),
        "strong_frac": float(strong_mask.float().mean().item()),
    }


def _normalization_stats(samples: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "module_grad_rms": _module_grad_rms(samples),
        "layer_module_grad_rms": _layer_module_grad_rms(samples),
        "global_grad_rms": float(samples["grad_rms"].float().median().item()),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_decoder(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    sample_path = Path(deep_get(cfg, "decoder_samples.output_path"))
    max_samples = deep_get(cfg, "decoder.max_samples")
    if smoke:
        max_samples = min(int(max_samples or 16384), 16384)

    samples, metadata = load_decoder_sample_tensors(sample_path, max_samples=max_samples)
    required_projection_keys = {"pinv_dB_norm", "pinv_dA_norm", "pinv_mean_norm", "pinv_diff_norm"}
    missing_projection = sorted(required_projection_keys - set(samples))
    if missing_projection:
        raise ValueError(
            f"Decoder samples are missing projection_v2 features: {missing_projection}. "
            "Regenerate samples with r2f_tofu.unlearn_dense."
        )
    if metadata.get("feature_version") != "projection_v2":
        raise ValueError(
            f"Decoder sample feature_version must be projection_v2, got {metadata.get('feature_version')!r}"
        )
    n = int(next(iter(samples.values())).shape[0])
    rank = int(samples["A_col"].shape[1])
    num_layers = int(samples["num_layers"].max().item()) if "num_layers" in samples else int(metadata.get("num_layers", 1))

    dataset = TensorDictDataset(samples)
    val_size = max(1, int(0.05 * len(dataset))) if len(dataset) > 20 else 0
    if val_size:
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
        )
    else:
        train_ds, val_ds = dataset, None

    batch_size = int(deep_get(cfg, "decoder.batch_size", 8192))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_tensor_dict)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tensor_dict)
        if val_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GradientDecoder(
        rank=rank,
        num_layers=num_layers,
        module_embedding_dim=int(deep_get(cfg, "decoder.module_embedding_dim", 16)),
        hidden_dim=int(deep_get(cfg, "decoder.hidden_dim", 512)),
        dropout=float(deep_get(cfg, "decoder.dropout", 0.05)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(cfg, "decoder.lr", 3e-4)),
        weight_decay=float(deep_get(cfg, "decoder.weight_decay", 1e-4)),
    )
    epochs = int(deep_get(cfg, "decoder.epochs", 3))
    max_steps = deep_get(cfg, "decoder.max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    sign_weight = float(deep_get(cfg, "decoder.sign_loss_weight", 0.01))
    target_clip = deep_get(cfg, "decoder.target_clip", 8.0)
    target_clip = float(target_clip) if target_clip is not None else None
    sign_threshold = float(deep_get(cfg, "decoder.sign_threshold", 0.05))
    checkpoint_every_steps = max(1, int(deep_get(cfg, "decoder.checkpoint_every_steps", 250)))
    fail_fast = bool(deep_get(cfg, "decoder.fail_fast_on_weak_baseline", True))
    baseline_min_corr = float(deep_get(cfg, "decoder.baseline_min_abs_corr", 0.02))
    baseline_min_sign = float(deep_get(cfg, "decoder.baseline_min_sign_acc", 0.51))
    checkpoint_path = ensure_parent(deep_get(cfg, "decoder.checkpoint_path"))
    normalization = _normalization_stats(samples)
    baseline_metrics = _prediction_metrics(
        samples["pinv_mean_norm"],
        samples["target_norm"],
        sign_threshold=sign_threshold,
    )
    if (
        fail_fast
        and not smoke
        and abs(baseline_metrics["corr"]) < baseline_min_corr
        and baseline_metrics["sign_acc"] < baseline_min_sign
    ):
        raise RuntimeError(
            "Projection baseline is too weak for full decoder training: "
            f"corr={baseline_metrics['corr']:.4f}, sign_acc={baseline_metrics['sign_acc']:.4f}. "
            "Regenerate same-state paired samples before training the decoder."
        )

    def save_checkpoint(complete: bool, reason: str) -> None:
        checkpoint = {
            "format": "r2f_gradient_decoder_v1",
            "complete": bool(complete),
            "checkpoint_reason": reason,
            "feature_version": "projection_v2",
            "model_state": _cpu_state_dict(model),
            "model_config": model.config_dict(),
            "normalization": normalization,
            "metadata": metadata,
            "train_samples": n,
            "baseline": baseline_metrics,
            "curve_tail": curve[-20:],
        }
        torch.save(checkpoint, checkpoint_path)

    curve: list[dict[str, float]] = []
    step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"decoder epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            step += 1
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss, metrics = decoder_loss(
                pred,
                batch["target_norm"],
                sign_loss_weight=sign_weight,
                target_clip=target_clip,
                sign_threshold=sign_threshold,
            )
            metrics.update(_prediction_metrics(pred, batch["target_norm"], sign_threshold=sign_threshold))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics["step"] = step
            metrics["epoch"] = epoch + 1
            curve.append(metrics)
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                sign=f"{metrics['sign_acc_strong']:.3f}",
                corr=f"{metrics['corr']:.3f}",
            )
            if step % checkpoint_every_steps == 0:
                save_checkpoint(complete=False, reason=f"step_{step}")
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    val_metrics: dict[str, float] = {}
    if val_loader is not None:
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device)
                pred = model(batch)
                _loss, metrics = decoder_loss(
                    pred,
                    batch["target_norm"],
                    sign_loss_weight=sign_weight,
                    target_clip=target_clip,
                    sign_threshold=sign_threshold,
                )
                metrics.update(_prediction_metrics(pred, batch["target_norm"], sign_threshold=sign_threshold))
                vals.append(metrics)
        if vals:
            val_metrics = {
                f"val_{key}": float(sum(row[key] for row in vals) / len(vals))
                for key in (
                    "loss",
                    "huber",
                    "sign_loss",
                    "sign_acc",
                    "sign_acc_strong",
                    "corr",
                    "r2",
                    "pred_std",
                )
            }

    save_checkpoint(complete=True, reason="training_complete")

    stats = {
        "checkpoint_path": str(checkpoint_path),
        "sample_path": str(sample_path),
        "samples_loaded": n,
        "rank": rank,
        "num_layers": num_layers,
        "steps": step,
        "last_train": curve[-1] if curve else {},
        "baseline": baseline_metrics,
        "feature_version": "projection_v2",
        "complete": True,
        **val_metrics,
        "normalization": normalization,
    }
    write_json(deep_get(cfg, "decoder.stats_path"), stats)
    write_json(deep_get(cfg, "decoder.curve_path"), curve)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    stats = train_decoder(cfg, smoke=args.smoke)
    print(stats)


if __name__ == "__main__":
    main()
