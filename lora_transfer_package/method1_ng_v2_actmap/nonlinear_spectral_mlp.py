#!/usr/bin/env python3
"""
nonlinear_spectral_mlp.py

Data-free nonlinear LoRA transfer refinement:
1) Learn a nonlinear singular-value map g from base-weight spectra (old->new)
   using a tiny MLP in log-spectrum space.
2) Apply g to each LoRA delta via exact low-rank SVD from (B, A) factors.
3) Re-factorize back to LoRA B/A with preserved rank.

This is motivated by spectral calibration (spcal): instead of linear or identity
spectrum handling, learn a smooth nonlinear transport between source/target
module spectra.
"""

import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from safetensors import safe_open

SRC_MODEL_DIR = "./modelzoo/qwen3_1_7B/"
TGT_MODEL_DIR = "./modelzoo/qwen3_8B/"
SRC_ADAPTER_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg"
OUT_ADAPTER_DIR = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_specmlp"

LAYER_MAPPING = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19,
                 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34]

MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

EPS = 1e-8
TOPK = 32
EPOCHS = 400
LR = 2e-3
SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def collect_spectrum_pairs(idx_old, idx_new):
    x_vals = []
    y_vals = []

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
            s_old = s_old[:k].clamp_min(EPS)
            s_new = s_new[:k].clamp_min(EPS)

            x_vals.append(torch.log(s_old))
            y_vals.append(torch.log(s_new))

            del w_old, w_new, s_old, s_new

    x = torch.cat(x_vals, dim=0).unsqueeze(1)
    y = torch.cat(y_vals, dim=0).unsqueeze(1)
    return x, y


class SpecMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_spec_mlp(x, y):
    model = SpecMLP()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Sort inputs once to impose monotonicity regularization.
    xs, order = torch.sort(x.squeeze(1))
    ys = y.squeeze(1)[order]
    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(1)

    for ep in range(EPOCHS):
        pred = model(xs)
        mse = ((pred - ys) ** 2).mean()

        # Monotonic prior: g should be non-decreasing on singular values.
        diff = pred[1:] - pred[:-1]
        mono_penalty = torch.relu(-diff).mean()

        # Smoothness prior.
        smooth_penalty = ((diff[1:] - diff[:-1]) ** 2).mean() if diff.numel() > 1 else 0.0

        loss = mse + 0.2 * mono_penalty + 0.05 * smooth_penalty
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == EPOCHS - 1:
            print(f"[train] ep={ep:03d} mse={mse.item():.6f} mono={float(mono_penalty):.6f}")

    return model


def low_rank_svd_from_factors(B, A):
    """
    Exact compact SVD of dW=B@A using QR+small SVD.
    B: (m, r), A: (r, n)
    Returns U:(m,r), S:(r,), Vh:(r,n)
    """
    # B = Qb Rb
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    # A^T = Qa Ra => A = Ra^T Qa^T
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    M = Rb @ Ra.T  # (r, r)
    Um, S, Vhm = torch.linalg.svd(M, full_matrices=False)

    U = Qb @ Um
    Vh = Vhm @ Qa.T
    return U, S, Vh


def apply_spec_map_to_adapter(spec_model, in_path, out_path):
    state = load_file(in_path)
    new_state = {}

    touched = 0
    for k, v in state.items():
        if not k.endswith("lora_B.weight"):
            new_state[k] = v
            continue

        a_key = k.replace("lora_B.weight", "lora_A.weight")
        if a_key not in state:
            new_state[k] = v
            continue

        B = state[k].float()
        A = state[a_key].float()
        r = B.shape[1]

        U, S, Vh = low_rank_svd_from_factors(B, A)

        # Nonlinear spectral transport in log space
        logS = torch.log(S.clamp_min(EPS)).unsqueeze(1)
        with torch.no_grad():
            logS_new = spec_model(logS).squeeze(1)
        S_new = torch.exp(logS_new).clamp_min(EPS)

        # Preserve module-level delta norm to avoid uncontrolled scale drift
        old_norm = torch.norm(S)
        new_norm = torch.norm(S_new)
        if old_norm > EPS and new_norm > EPS:
            S_new = S_new * (old_norm / new_norm)

        S_sqrt = torch.sqrt(S_new)
        B_new = (U * S_sqrt.unsqueeze(0)).to(v.dtype)
        A_new = (S_sqrt.unsqueeze(1) * Vh).to(state[a_key].dtype)

        new_state[k] = B_new.contiguous()
        new_state[a_key] = A_new.contiguous()
        touched += 1

    # Pass through any untouched tensors
    for k, v in state.items():
        if k not in new_state:
            new_state[k] = v

    save_file(new_state, out_path)
    print(f"Saved mapped adapter: {out_path}")
    print(f"Touched LoRA pairs: {touched}")


def main():
    set_seed(SEED)
    os.makedirs(OUT_ADAPTER_DIR, exist_ok=True)

    # copy metadata/config files
    for fn in os.listdir(SRC_ADAPTER_DIR):
        src = os.path.join(SRC_ADAPTER_DIR, fn)
        dst = os.path.join(OUT_ADAPTER_DIR, fn)
        if fn != "adapter_model.safetensors":
            shutil.copy2(src, dst)

    print("Building model shard indexes...")
    idx_old = build_shard_index(SRC_MODEL_DIR)
    idx_new = build_shard_index(TGT_MODEL_DIR)

    print("Collecting old/new spectral pairs from base weights...")
    x, y = collect_spectrum_pairs(idx_old, idx_new)
    print(f"Training samples: {x.shape[0]}")

    print("Training nonlinear spectral mapper (tiny MLP)...")
    spec_model = train_spec_mlp(x, y)

    in_adapter = os.path.join(SRC_ADAPTER_DIR, "adapter_model.safetensors")
    out_adapter = os.path.join(OUT_ADAPTER_DIR, "adapter_model.safetensors")

    print("Applying spectral map to LoRA tensors...")
    apply_spec_map_to_adapter(spec_model, in_adapter, out_adapter)


if __name__ == "__main__":
    main()
