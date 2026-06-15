from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def write_json(path: str | Path, data: Any) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def move_to_device(batch: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.float().pow(2)) + 1e-12)


def cuda_memory_summary() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"cuda_peak_allocated_gb": 0.0, "cuda_peak_reserved_gb": 0.0}
    return {
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def accumulation_group_size(step: int, total_steps: int, grad_accum_steps: int) -> int:
    grad_accum_steps = max(int(grad_accum_steps), 1)
    group_start = ((int(step) - 1) // grad_accum_steps) * grad_accum_steps + 1
    group_end = min(group_start + grad_accum_steps - 1, int(total_steps))
    return max(group_end - group_start + 1, 1)


def is_accumulation_boundary(step: int, total_steps: int, grad_accum_steps: int) -> bool:
    grad_accum_steps = max(int(grad_accum_steps), 1)
    return int(step) % grad_accum_steps == 0 or int(step) >= int(total_steps)


def sine_cosine_depth(layer_idx: torch.Tensor, num_layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = max(num_layers - 1, 1)
    rel = layer_idx.float() / denom
    return rel, torch.sin(2 * math.pi * rel), torch.cos(2 * math.pi * rel)
