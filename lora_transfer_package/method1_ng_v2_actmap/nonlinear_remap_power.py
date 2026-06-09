#!/usr/bin/env python3
"""Apply signed-power post-refinement to a transferred LoRA adapter."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default="./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg",
        help="Input adapter directory.",
    )
    parser.add_argument(
        "--dst_dir",
        default="./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_nlpow15",
        help="Output adapter directory.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.5,
        help="Signed-power exponent applied to lora_B tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(src_dir):
        if fname != "adapter_model.safetensors":
            shutil.copy2(src_dir / fname, dst_dir / fname)

    state = load_file(src_dir / "adapter_model.safetensors")
    new_state = {}
    changed = 0

    for k, v in state.items():
        if ".lora_B." in k:
            t = v.float()
            # Data-free nonlinear remap: signed power, then per-tensor norm preserve.
            t_new = torch.sign(t) * torch.pow(torch.abs(t) + EPS, args.gamma)
            n_old = torch.norm(t)
            n_new = torch.norm(t_new)
            if n_new > EPS and n_old > EPS:
                t_new = t_new * (n_old / n_new)
            new_state[k] = t_new.to(v.dtype)
            changed += 1
        else:
            new_state[k] = v

    save_file(new_state, dst_dir / "adapter_model.safetensors")
    print(f"Saved nonlinear remapped adapter to {dst_dir}")
    print(f"Gamma: {args.gamma}")
    print(f"Changed lora_B tensors: {changed}")


if __name__ == "__main__":
    main()
