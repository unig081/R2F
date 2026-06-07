from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .module_keys import ModuleKey, parse_module_key
from .utils import ensure_dir, ensure_parent, tensor_rms, write_json


def capture_lora_gradients(
    model: torch.nn.Module,
    target_modules: list[str],
    train_layers: list[int] | None = None,
    train_modules: list[str] | None = None,
) -> dict[ModuleKey, dict[str, torch.Tensor]]:
    records: dict[ModuleKey, dict[str, torch.Tensor]] = {}
    layer_filter = set(train_layers) if train_layers is not None else None
    module_filter = set(train_modules) if train_modules is not None else None

    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        key = parse_module_key(name, target_modules)
        if key is None:
            continue
        if layer_filter is not None and key.layer_idx not in layer_filter:
            continue
        if module_filter is not None and key.module_type not in module_filter:
            continue

        lora_a = module.lora_A["default"]
        lora_b = module.lora_B["default"]
        if lora_a.weight.grad is None or lora_b.weight.grad is None:
            continue
        records[key] = {
            "A": lora_a.weight.detach().float().cpu(),
            "B": lora_b.weight.detach().float().cpu(),
            "dA": lora_a.weight.grad.detach().float().cpu(),
            "dB": lora_b.weight.grad.detach().float().cpu(),
        }

    if not records:
        raise RuntimeError("No LoRA gradients were captured")
    return records


def capture_dense_gradients(
    dense_params: dict[ModuleKey, torch.nn.Parameter],
) -> dict[ModuleKey, dict[str, torch.Tensor]]:
    records: dict[ModuleKey, dict[str, torch.Tensor]] = {}
    for key, param in dense_params.items():
        if param.grad is None:
            continue
        records[key] = {
            "W": param.detach().float().cpu(),
            "dW": param.grad.detach().float().cpu(),
        }
    if not records:
        raise RuntimeError("No dense gradients were captured")
    return records


def _assert_pair_shapes(key: ModuleKey, lora: dict[str, torch.Tensor], dense: dict[str, torch.Tensor]) -> None:
    a, b, da, db, dw = lora["A"], lora["B"], lora["dA"], lora["dB"], dense["dW"]
    if a.shape != da.shape:
        raise ValueError(f"{key.as_string()} A/dA shape mismatch: {a.shape} vs {da.shape}")
    if b.shape != db.shape:
        raise ValueError(f"{key.as_string()} B/dB shape mismatch: {b.shape} vs {db.shape}")
    if dw.shape != (b.shape[0], a.shape[1]):
        raise ValueError(
            f"{key.as_string()} dW shape {dw.shape} does not match B/A {(b.shape[0], a.shape[1])}"
        )


def sample_coordinates_for_module(
    key: ModuleKey,
    lora: dict[str, torch.Tensor],
    dense: dict[str, torch.Tensor],
    coords_per_module: int,
    num_layers: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    _assert_pair_shapes(key, lora, dense)
    a = lora["A"]
    b = lora["B"]
    da = lora["dA"]
    db = lora["dB"]
    dw = dense["dW"]

    out_dim, in_dim = dw.shape
    n = min(coords_per_module, out_dim * in_dim)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    o_idx = torch.randint(0, out_dim, (n,), generator=gen)
    i_idx = torch.randint(0, in_dim, (n,), generator=gen)

    grad_rms = tensor_rms(dw).clamp_min(1e-12)
    target_raw = dw[o_idx, i_idx].float()
    target_norm = target_raw / grad_rms

    return {
        "A_col": a[:, i_idx].transpose(0, 1).contiguous(),
        "B_row": b[o_idx, :].contiguous(),
        "dA_col": da[:, i_idx].transpose(0, 1).contiguous(),
        "dB_row": db[o_idx, :].contiguous(),
        "target_norm": target_norm.contiguous(),
        "target_raw": target_raw.contiguous(),
        "grad_rms": torch.full((n,), float(grad_rms), dtype=torch.float32),
        "layer_idx": torch.full((n,), key.layer_idx, dtype=torch.long),
        "module_id": torch.full((n,), key.module_id, dtype=torch.long),
        "coord_o": o_idx.long(),
        "coord_i": i_idx.long(),
        "num_layers": torch.full((n,), num_layers, dtype=torch.long),
    }


def sample_paired_gradients(
    lora_records: dict[ModuleKey, dict[str, torch.Tensor]],
    dense_records: dict[ModuleKey, dict[str, torch.Tensor]],
    coords_per_module: int,
    num_layers: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    chunks: list[dict[str, torch.Tensor]] = []
    for offset, key in enumerate(sorted(lora_records, key=lambda x: (x.layer_idx, x.module_type))):
        if key not in dense_records:
            continue
        chunks.append(
            sample_coordinates_for_module(
                key,
                lora_records[key],
                dense_records[key],
                coords_per_module=coords_per_module,
                num_layers=num_layers,
                seed=seed + offset * 1009,
            )
        )
    if not chunks:
        raise RuntimeError("No paired LoRA/dense gradient records were available")
    return concat_sample_chunks(chunks)


def concat_sample_chunks(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = chunks[0].keys()
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}


class DecoderSampleWriter:
    def __init__(
        self,
        output_path: str | Path,
        metadata: dict[str, Any],
        flush_every_steps: int = 8,
    ) -> None:
        self.output_path = ensure_parent(output_path)
        stem = self.output_path.name
        if stem.endswith(".pt"):
            stem = stem[:-3]
        self.chunk_dir = ensure_dir(self.output_path.parent / f"{stem}_chunks")
        self.metadata = metadata
        self.flush_every_steps = max(int(flush_every_steps), 1)
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.chunks: list[str] = []
        self.steps_since_flush = 0
        self.total_samples = 0

    def add(self, samples: dict[str, torch.Tensor]) -> None:
        self.buffer.append(samples)
        self.total_samples += int(next(iter(samples.values())).shape[0])
        self.steps_since_flush += 1
        if self.steps_since_flush >= self.flush_every_steps:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        chunk = concat_sample_chunks(self.buffer)
        chunk_path = self.chunk_dir / f"chunk_{len(self.chunks):06d}.pt"
        torch.save(chunk, chunk_path)
        self.chunks.append(str(chunk_path))
        self.buffer.clear()
        self.steps_since_flush = 0

    def close(self) -> None:
        self.flush()
        manifest = {
            "format": "r2f_decoder_samples_manifest_v1",
            "metadata": self.metadata,
            "chunks": self.chunks,
            "total_samples": self.total_samples,
        }
        torch.save(manifest, self.output_path)

    def write_stats(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "output_path": str(self.output_path),
            "chunk_dir": str(self.chunk_dir),
            "chunks": len(self.chunks),
            "total_samples": self.total_samples,
            "metadata": self.metadata,
        }
        if extra:
            payload.update(extra)
        write_json(path, payload)


def load_decoder_sample_tensors(
    path: str | Path,
    max_samples: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == "r2f_decoder_samples_manifest_v1":
        chunks = []
        remaining = max_samples
        for chunk_path in payload["chunks"]:
            chunk = torch.load(chunk_path, map_location="cpu")
            if remaining is not None:
                n = int(next(iter(chunk.values())).shape[0])
                if remaining <= 0:
                    break
                if n > remaining:
                    chunk = {k: v[:remaining] for k, v in chunk.items()}
                    remaining = 0
                else:
                    remaining -= n
            chunks.append(chunk)
        if not chunks:
            raise ValueError(f"No samples loaded from {path}")
        return concat_sample_chunks(chunks), payload.get("metadata", {})
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        if max_samples is not None:
            samples = {k: v[:max_samples] for k, v in samples.items()}
        return samples, payload.get("metadata", {})
    if isinstance(payload, dict) and "A_col" in payload:
        if max_samples is not None:
            payload = {k: v[:max_samples] for k, v in payload.items()}
        return payload, {}
    raise ValueError(f"Unsupported decoder sample file: {path}")


def shape_report_from_records(
    lora_records: dict[ModuleKey, dict[str, torch.Tensor]],
    dense_records: dict[ModuleKey, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key, rec in sorted(lora_records.items(), key=lambda item: (item[0].layer_idx, item[0].module_type)):
        item: dict[str, Any] = {
            "A": list(rec["A"].shape),
            "B": list(rec["B"].shape),
            "dA": list(rec["dA"].shape),
            "dB": list(rec["dB"].shape),
        }
        if dense_records and key in dense_records:
            item["W"] = list(dense_records[key]["W"].shape)
            item["dW"] = list(dense_records[key]["dW"].shape)
        report[key.as_string()] = item
    return report


def save_shape_report(path: str | Path, report: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
