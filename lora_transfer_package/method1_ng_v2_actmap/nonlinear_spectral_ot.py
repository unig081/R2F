#!/usr/bin/env python3
"""
nonlinear_spectral_ot.py

Data-free nonlinear transfer via 1D monotone optimal transport in spectrum space:
1) Collect singular values from matched old/new base weights.
2) Build monotone map g(s)=Q_new(F_old(s)) by empirical quantiles.
3) Apply g to each LoRA delta singular value, then refactorize to B/A.

This is a grounded nonlinear method (quantile transport / histogram matching),
not relying on forget/retain labels.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SRC_MODEL_DIR = "./modelzoo/qwen3_1_7B/"
TGT_MODEL_DIR = "./modelzoo/qwen3_8B/"
SRC_ADAPTER_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg"
OUT_ADAPTER_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_specot"

LAYER_MAPPING = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19,
                 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34]

MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

TOPK = 32
EPS = 1e-8


def build_shard_index(model_dir: str):
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        return {k: str(Path(model_dir) / v) for k, v in idx["weight_map"].items()}
    single = Path(model_dir) / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())
        return {k: str(single) for k in keys}
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def load_tensor(idx, key):
    with safe_open(idx[key], framework="pt") as f:
        return f.get_tensor(key).float()


def collect_spectrum_samples(idx_old, idx_new):
    s_old_all = []
    s_new_all = []

    for old_i, new_j in enumerate(LAYER_MAPPING):
        for mod in MODULES:
            k_old = f"model.layers.{old_i}.{mod}.weight"
            k_new = f"model.layers.{new_j}.{mod}.weight"
            if k_old not in idx_old or k_new not in idx_new:
                continue

            w_old = load_tensor(idx_old, k_old)
            w_new = load_tensor(idx_new, k_new)

            s_old = torch.linalg.svdvals(w_old)
            s_new = torch.linalg.svdvals(w_new)
            k = min(TOPK, s_old.numel(), s_new.numel())

            s_old_all.append(s_old[:k].clamp_min(EPS))
            s_new_all.append(s_new[:k].clamp_min(EPS))

    s_old_cat = torch.cat(s_old_all)
    s_new_cat = torch.cat(s_new_all)

    # Build empirical quantiles for monotone OT map.
    old_sorted, _ = torch.sort(s_old_cat)
    new_sorted, _ = torch.sort(s_new_cat)
    return old_sorted, new_sorted


def quantile_map(x, old_sorted, new_sorted):
    """
    Monotone map g(x)=Q_new(F_old(x)) via empirical CDF with linear interpolation.
    x can be tensor of any shape.
    """
    old = old_sorted
    new = new_sorted
    n = old.numel()

    x_flat = x.reshape(-1)
    idx = torch.searchsorted(old, x_flat)
    idx = torch.clamp(idx, 1, n - 1)

    x0 = old[idx - 1]
    x1 = old[idx]
    y0 = new[idx - 1]
    y1 = new[idx]

    t = (x_flat - x0) / (x1 - x0 + EPS)
    y = y0 + t * (y1 - y0)
    return y.reshape(x.shape)


def low_rank_svd_from_factors(B, A):
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")
    M = Rb @ Ra.T
    Um, S, Vhm = torch.linalg.svd(M, full_matrices=False)
    U = Qb @ Um
    Vh = Vhm @ Qa.T
    return U, S, Vh


def apply_spec_ot(in_path, out_path, old_sorted, new_sorted):
    state = load_file(in_path)
    out = {}
    touched = 0

    for k, v in state.items():
        if not k.endswith("lora_B.weight"):
            out[k] = v
            continue

        a_key = k.replace("lora_B.weight", "lora_A.weight")
        if a_key not in state:
            out[k] = v
            continue

        B = state[k].float()
        A = state[a_key].float()

        U, S, Vh = low_rank_svd_from_factors(B, A)
        S_new = quantile_map(S.clamp_min(EPS), old_sorted, new_sorted).clamp_min(EPS)

        # keep module energy stable
        n_old = torch.norm(S)
        n_new = torch.norm(S_new)
        if n_old > EPS and n_new > EPS:
            S_new = S_new * (n_old / n_new)

        s_sqrt = torch.sqrt(S_new)
        B_new = (U * s_sqrt.unsqueeze(0)).to(v.dtype)
        A_new = (s_sqrt.unsqueeze(1) * Vh).to(state[a_key].dtype)

        out[k] = B_new.contiguous()
        out[a_key] = A_new.contiguous()
        touched += 1

    for k, v in state.items():
        if k not in out:
            out[k] = v

    save_file(out, out_path)
    print(f"Saved adapter: {out_path}")
    print(f"Touched LoRA pairs: {touched}")


def main():
    os.makedirs(OUT_ADAPTER_DIR, exist_ok=True)

    for fn in os.listdir(SRC_ADAPTER_DIR):
        src = os.path.join(SRC_ADAPTER_DIR, fn)
        dst = os.path.join(OUT_ADAPTER_DIR, fn)
        if fn != "adapter_model.safetensors":
            shutil.copy2(src, dst)

    print("Building model shard indexes...")
    idx_old = build_shard_index(SRC_MODEL_DIR)
    idx_new = build_shard_index(TGT_MODEL_DIR)

    print("Collecting spectral samples for OT map...")
    old_sorted, new_sorted = collect_spectrum_samples(idx_old, idx_new)
    print(f"Quantile samples: {old_sorted.numel()}")

    in_adapter = os.path.join(SRC_ADAPTER_DIR, "adapter_model.safetensors")
    out_adapter = os.path.join(OUT_ADAPTER_DIR, "adapter_model.safetensors")

    print("Applying OT spectral map to LoRA...")
    apply_spec_ot(in_adapter, out_adapter, old_sorted, new_sorted)


if __name__ == "__main__":
    main()
