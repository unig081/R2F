from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .decoder import GradientDecoder, decoder_loss
from .grad_capture import load_decoder_sample_tensors
from .module_keys import ID_TO_MODULE
from .train_decoder import TensorDictDataset, collate_tensor_dict
from .utils import ensure_parent, move_to_device, set_seed, write_json


def _load_decoder_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("format") != "r2f_gradient_decoder_v1":
        raise ValueError(f"Unsupported decoder checkpoint: {checkpoint_path}")
    return ckpt


def _build_decoder(ckpt: dict[str, Any], rank: int, num_layers: int, cfg: dict[str, Any]) -> GradientDecoder:
    model_cfg = dict(ckpt.get("model_config") or {})
    model = GradientDecoder(
        rank=int(model_cfg.get("rank", rank)),
        num_modules=int(model_cfg.get("num_modules", 7)),
        num_layers=int(model_cfg.get("num_layers", num_layers)),
        module_embedding_dim=int(
            model_cfg.get("module_embedding_dim", deep_get(cfg, "decoder.module_embedding_dim", 16))
        ),
        hidden_dim=int(model_cfg.get("hidden_dim", deep_get(cfg, "decoder.hidden_dim", 512))),
        dropout=float(deep_get(cfg, "decoder.dropout", 0.05)),
    )
    model.load_state_dict(ckpt["model_state"])
    return model


def _make_splits(dataset: Dataset[Any], seed: int) -> dict[str, Dataset[Any]]:
    splits: dict[str, Dataset[Any]] = {"all": dataset}
    val_size = max(1, int(0.05 * len(dataset))) if len(dataset) > 20 else 0
    if val_size:
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        splits["train_split"] = train_ds
        splits["val_split"] = val_ds
    return splits


def _safe_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = torch.sqrt(pred_centered.pow(2).sum() * target_centered.pow(2).sum())
    if float(denom.item()) == 0.0:
        return 0.0
    return float((pred_centered * target_centered).sum().div(denom).item())


def _regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    diff = pred - target
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    target_var = target.var(unbiased=False)
    r2 = 0.0 if float(target_var.item()) == 0.0 else 1.0 - float(mse.div(target_var).item())
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(mae.item()),
        "corr": _safe_corrcoef(pred, target),
        "r2": r2,
        "pred_mean": float(pred.mean().item()),
        "pred_std": float(pred.std(unbiased=False).item()),
        "target_mean": float(target.mean().item()),
        "target_std": float(target.std(unbiased=False).item()),
        "pred_abs_mean": float(pred.abs().mean().item()),
        "target_abs_mean": float(target.abs().mean().item()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
        if isinstance(rows[0][key], float)
    }


def _module_metrics(pred: torch.Tensor, target: torch.Tensor, module_ids: torch.Tensor) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for module_id in sorted(module_ids.long().unique().tolist()):
        mask = module_ids.long() == int(module_id)
        if not mask.any():
            continue
        module = ID_TO_MODULE.get(int(module_id), str(module_id))
        result[module] = {
            "samples": int(mask.sum().item()),
            **_regression_metrics(pred[mask], target[mask]),
        }
    return result


def _evaluate_dataset(
    model: GradientDecoder,
    dataset: Dataset[Any],
    batch_size: int,
    sign_loss_weight: float,
    device: torch.device,
    desc: str,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tensor_dict)
    loss_rows: list[dict[str, float]] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    module_ids: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = move_to_device(batch, device)
            pred = model(batch)
            _loss, loss_metrics = decoder_loss(pred, batch["target_norm"], sign_loss_weight=sign_loss_weight)
            loss_rows.append(loss_metrics)
            preds.append(pred.detach().cpu())
            targets.append(batch["target_norm"].detach().cpu())
            module_ids.append(batch["module_id"].detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    module_id_all = torch.cat(module_ids, dim=0)
    return {
        "samples": int(target_all.numel()),
        **_mean_metrics(loss_rows),
        **_regression_metrics(pred_all, target_all),
        "by_module": _module_metrics(pred_all, target_all, module_id_all),
    }


def eval_decoder(
    cfg: dict[str, Any],
    smoke: bool = False,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    sample_path = Path(deep_get(cfg, "decoder_samples.output_path"))
    if max_samples is None:
        cfg_max_samples = deep_get(cfg, "decoder.eval_max_samples", deep_get(cfg, "decoder.max_samples"))
        max_samples = int(cfg_max_samples) if cfg_max_samples is not None else None
    if smoke:
        max_samples = min(int(max_samples or 16384), 16384)

    samples, metadata = load_decoder_sample_tensors(sample_path, max_samples=max_samples)
    n = int(next(iter(samples.values())).shape[0])
    rank = int(samples["A_col"].shape[1])
    num_layers = int(samples["num_layers"].max().item()) if "num_layers" in samples else int(metadata.get("num_layers", 1))

    checkpoint_path = Path(deep_get(cfg, "decoder.checkpoint_path"))
    ckpt = _load_decoder_checkpoint(checkpoint_path)
    model = _build_decoder(ckpt, rank=rank, num_layers=num_layers, cfg=cfg)

    dataset = TensorDictDataset(samples)
    splits = _make_splits(dataset, seed=int(cfg.get("seed", 42)))

    batch_size = int(deep_get(cfg, "decoder.batch_size", 8192))
    sign_weight = float(deep_get(cfg, "decoder.sign_loss_weight", 0.01))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    split_metrics = {
        name: _evaluate_dataset(
            model=model,
            dataset=split,
            batch_size=batch_size,
            sign_loss_weight=sign_weight,
            device=device,
            desc=f"eval decoder {name}",
        )
        for name, split in splits.items()
    }

    result = {
        "checkpoint_path": str(checkpoint_path),
        "sample_path": str(sample_path),
        "samples_loaded": n,
        "rank": rank,
        "num_layers": num_layers,
        "smoke": smoke,
        "metadata": metadata,
        "checkpoint_train_samples": ckpt.get("train_samples"),
        "splits": split_metrics,
    }

    out = Path(output_path) if output_path is not None else checkpoint_path.parent / "eval_stats.json"
    write_json(ensure_parent(out), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    stats = eval_decoder(
        cfg,
        smoke=args.smoke,
        max_samples=args.max_samples,
        output_path=args.output,
    )
    print(stats)


if __name__ == "__main__":
    main()
