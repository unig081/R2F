#!/usr/bin/env python3
"""
transfer_lora_actmap.py
=======================
Transfer 1.7B wmdp-cyber unlearning LoRA to 8B using activation-aligned mappings:

  - R_l : (2048, 4096) per-layer hidden-state alignment (already pre-computed)
      h_1.7B @ R_l ≈ h_8B   (row-vector convention)
      In column convention:  R_l^T @ h_1.7B ≈ h_8B

  - P_l : (12288, 6144) per-layer FFN intermediate alignment, derived analytically:
      P_l = W_g_8B_l @ R_l^T @ pinv(W_g_1.7B_l)
    So that: v_inter_8B ≈ P_l @ v_inter_1.7B  (column vectors)

LoRA transformation rules (rank=32):
  - hidden-dim A  (rank × 2048 → rank × 4096):  A_8B = A_1.7B @ R_l
  - hidden-dim B  (2048 × rank → 4096 × rank):  B_8B = R_l.T @ B_1.7B
  - inter-dim B   (6144 × rank → 12288 × rank):  B_8B = P_l @ B_1.7B
  - inter-dim A   (rank × 6144 → rank × 12288):  A_8B = A_1.7B @ P_l.T
  - KV-dim B      (1024 × rank → 1024 × rank):   COPY (same KV heads)

Layer mapping (from act_align meta.json):
  source 1.7B layer i → target 8B layer LAYER_MAPPING[i]
  For unmapped 8B layers, use nearest source layer.
"""

import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SRC_MODEL_DIR   = "./modelzoo/qwen3_1_7B/"
TGT_MODEL_DIR   = "./modelzoo/qwen3_8B/"
SRC_LORA_DIR    = "./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/"
R_L_DIR         = "./tmp/act_align_qwen3_cyber59/"   # use cyber-domain alignment
OUTPUT_DIR      = "./trained_models/xTransform/qwen3_8B_cyber_actmap_v4"

# From act_align meta.json — source layer i → target layer LAYER_MAPPING[i]
LAYER_MAPPING = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19,
                 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34]
N_SRC = 28
N_TGT = 36
UNMAPPED_TGT_LAYERS = {4, 8, 13, 17, 22, 26, 31, 35}

DTYPE = torch.float32   # use float32 for precision during computation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: load safetensors weight by key
# ─────────────────────────────────────────────────────────────────────────────

def build_shard_index(model_dir):
    """Return dict: tensor_name -> safetensors file path."""
    index_file = Path(model_dir) / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            idx = json.load(f)
        return {k: str(Path(model_dir) / v) for k, v in idx["weight_map"].items()}
    # Single-file model
    single = Path(model_dir) / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())
        return {k: str(single) for k in keys}
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def load_tensor(shard_index, key, dtype=DTYPE):
    """Load a single tensor from the sharded model."""
    path = shard_index[key]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Layer-mapping utilities
# ─────────────────────────────────────────────────────────────────────────────

def tgt_to_src(tgt_j):
    """For target layer j, return the best matching source layer i."""
    diffs = [abs(lm - tgt_j) for lm in LAYER_MAPPING]
    return int(diffs.index(min(diffs)))


def layer_transfer_scale(tgt_j):
    """No attenuation — v4 uses per-module norm matching instead."""
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# P_l computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_P(W_g_8B, R_l, W_g_1_7B):
    """
    P_l = W_g_8B @ R_l.T @ pinv(W_g_1.7B)

    W_g_8B  : (12288, 4096)   gate_proj.weight of 8B at target layer
    R_l     : (2048,  4096)   hidden-state alignment for source layer (row-vec convention)
    W_g_1_7B: (6144,  2048)   gate_proj.weight of 1.7B at source layer

    Returns P_l: (12288, 6144)
    """
    W_g_8B   = W_g_8B.float()
    R_l      = R_l.float()
    W_g_1_7B = W_g_1_7B.float()

    # Step 1: W_g_8B @ R_l.T  →  (12288, 4096) @ (4096, 2048) = (12288, 2048)
    A = W_g_8B @ R_l.T          # (12288, 2048)

    # Step 2: pinv(W_g_1.7B)  →  W_g_1.7B is (6144, 2048), pinv is (2048, 6144)
    # Use lstsq for numerical stability (equivalent to Moore-Penrose with truncation)
    # We want pinv(W_g_1_7B): (2048, 6144), so we solve W_g_1_7B.T @ X = I_{2048}
    # Or directly: X = pinv(W_g_1_7B) = (W_g_1_7B.T @ W_g_1_7B)^{-1} @ W_g_1_7B.T
    # Use torch.linalg.lstsq: argmin_X ||W_g_1_7B @ X - I||  (tall system, overconstrained)
    # Better: pinv via SVD
    # Use larger rcond for numerical stability
    W_pinv = torch.linalg.pinv(W_g_1_7B, rcond=1e-2)   # (2048, 6144)

    # Step 3: P_l = A @ W_pinv  →  (12288, 2048) @ (2048, 6144) = (12288, 6144)
    P_l = A @ W_pinv

    # Clip per-column norm to prevent explosion (each column maps one 1.7B neuron to 8B space)
    col_norms = P_l.norm(dim=0, keepdim=True).clamp(min=1e-8)   # (1, 6144)
    # Target: average col norm of a weight matrix with similar scale
    # Gate_proj weight col norm is roughly ||W_g_8B[:,j]|| / ||W_g_1_7B[:,j]|| ≈ 1-3
    # Clip any column norm exceeding 3x the median
    median_norm = col_norms.median()
    max_norm = (3.0 * median_norm).clamp(min=1.0)
    scale = (max_norm / col_norms).clamp(max=1.0)   # only shrink, never amplify
    P_l = P_l * scale

    return P_l   # (12288, 6144)


# ─────────────────────────────────────────────────────────────────────────────
# Main transfer
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build weight shard indexes
    print("Building shard indexes …")
    idx_src = build_shard_index(SRC_MODEL_DIR)
    idx_tgt = build_shard_index(TGT_MODEL_DIR)

    # Load source LoRA weights
    print("Loading source LoRA …")
    lora_src = {}
    with safe_open(SRC_LORA_DIR + "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            lora_src[k] = f.get_tensor(k).float()

    # Determine LoRA rank and alpha from config
    with open(SRC_LORA_DIR + "adapter_config.json") as f:
        src_cfg = json.load(f)
    rank       = src_cfg["r"]
    lora_alpha = src_cfg["lora_alpha"]
    print(f"Source LoRA rank={rank}, alpha={lora_alpha}")

    # ─────────────────────────────────────────────────────────────────────
    # Pre-load R_l matrices indexed by SOURCE layer
    # ─────────────────────────────────────────────────────────────────────
    print("Loading R_l matrices …")
    R_l_dict = {}
    for i in range(N_SRC):
        path = Path(R_L_DIR) / f"R_l_{i}.pt"
        if path.exists():
            R_l_dict[i] = torch.load(str(path), map_location="cpu").float()
        else:
            print(f"  WARNING: R_l_{i}.pt not found, skipping")

    # ─────────────────────────────────────────────────────────────────────
    # Process each target layer
    # ─────────────────────────────────────────────────────────────────────
    new_lora = {}

    for tgt_j in range(N_TGT):
        src_i = tgt_to_src(tgt_j)
        l_scale = layer_transfer_scale(tgt_j)
        print(f"Target layer {tgt_j:2d}  ←  source layer {src_i}  (8B mapped={LAYER_MAPPING[src_i]}, scale={l_scale:.3f})")

        # Fetch R_l for this source layer
        R_l = R_l_dict[src_i]   # (2048, 4096)

        # Fetch gate_proj weights for P_l computation
        src_gate_key = f"model.layers.{src_i}.mlp.gate_proj.weight"
        tgt_gate_key = f"model.layers.{tgt_j}.mlp.gate_proj.weight"
        W_g_1_7B = load_tensor(idx_src, src_gate_key)   # (6144, 2048)
        W_g_8B   = load_tensor(idx_tgt, tgt_gate_key)   # (12288, 4096)

        # Compute P_l: (12288, 6144)
        P_l = compute_P(W_g_8B, R_l, W_g_1_7B)
        del W_g_1_7B, W_g_8B  # free memory

        # ── Helper closures ──────────────────────────────────────────────
        def transform_A_hidden(A):
            """A: (rank, 2048) → (rank, 4096) via A @ R_l"""
            return (A.float() @ R_l).to(torch.bfloat16)

        def transform_B_hidden(B):
            """B: (2048, rank) → (4096, rank) via R_l.T @ B"""
            return (R_l.T @ B.float()).to(torch.bfloat16)

        def transform_A_inter(A):
            """A: (rank, 6144) → (rank, 12288) via A @ P_l.T"""
            return (A.float() @ P_l.T).to(torch.bfloat16)

        def transform_B_inter(B):
            """B: (6144, rank) → (12288, rank) via P_l @ B"""
            return (P_l @ B.float()).to(torch.bfloat16)

        def copy_tensor(T):
            return T.to(torch.bfloat16)

        # ── Key prefix in source LoRA ─────────────────────────────────
        src_pfx = f"base_model.model.model.layers.{src_i}"
        tgt_pfx = f"base_model.model.model.layers.{tgt_j}"

        # ── Module-by-module transformation ──────────────────────────
        #   Each entry: (src_suffix, tgt_suffix, transform_fn)
        transforms = [
            # Attention
            ("self_attn.q_proj.lora_A.weight", "self_attn.q_proj.lora_A.weight", transform_A_hidden),
            ("self_attn.q_proj.lora_B.weight", "self_attn.q_proj.lora_B.weight", transform_B_hidden),
            ("self_attn.k_proj.lora_A.weight", "self_attn.k_proj.lora_A.weight", transform_A_hidden),
            ("self_attn.k_proj.lora_B.weight", "self_attn.k_proj.lora_B.weight", copy_tensor),   # KV dim unchanged
            ("self_attn.v_proj.lora_A.weight", "self_attn.v_proj.lora_A.weight", transform_A_hidden),
            ("self_attn.v_proj.lora_B.weight", "self_attn.v_proj.lora_B.weight", copy_tensor),   # KV dim unchanged
            ("self_attn.o_proj.lora_A.weight", "self_attn.o_proj.lora_A.weight", transform_A_hidden),
            ("self_attn.o_proj.lora_B.weight", "self_attn.o_proj.lora_B.weight", transform_B_hidden),
            # FFN
            ("mlp.gate_proj.lora_A.weight", "mlp.gate_proj.lora_A.weight", transform_A_hidden),
            ("mlp.gate_proj.lora_B.weight", "mlp.gate_proj.lora_B.weight", transform_B_inter),
            ("mlp.up_proj.lora_A.weight",   "mlp.up_proj.lora_A.weight",   transform_A_hidden),
            ("mlp.up_proj.lora_B.weight",   "mlp.up_proj.lora_B.weight",   transform_B_inter),
            ("mlp.down_proj.lora_A.weight", "mlp.down_proj.lora_A.weight", transform_A_inter),
            ("mlp.down_proj.lora_B.weight", "mlp.down_proj.lora_B.weight", transform_B_hidden),
        ]

        for src_suf, tgt_suf, fn in transforms:
            src_key = f"{src_pfx}.{src_suf}"
            tgt_key = f"{tgt_pfx}.{tgt_suf}"
            if src_key in lora_src:
                t = fn(lora_src[src_key])
                if l_scale != 1.0:
                    t = t * math.sqrt(l_scale)
                # ── Per-module norm matching ──────────────────────────────
                # Rescale B matrix so that ||B_8B @ A_8B|| matches
                # (alpha/rank) * ||B_1.7B @ A_1.7B|| from the source LoRA.
                # This compensates for the ~100x norm drop introduced by the
                # linear projection through R_l and P_l.
                if tgt_suf.endswith("lora_B.weight"):
                    mod_base = tgt_suf.replace("lora_B.weight", "")
                    # get corresponding A tensor (already stored or compute on the fly)
                    a_tgt_key = f"{tgt_pfx}.{mod_base}lora_A.weight"
                    a_src_key = f"{src_pfx}.{mod_base}lora_A.weight"
                    # A_8B may already be in new_lora (processed earlier in same layer)
                    if a_tgt_key in new_lora:
                        A_8B = new_lora[a_tgt_key].float()
                        B_8B = t.float()
                        dW_tgt_norm = (B_8B @ A_8B).norm().item()

                        # Compute source dW norm
                        B_17 = lora_src[f"{src_pfx}.{mod_base}lora_B.weight"].float()
                        A_17 = lora_src[a_src_key].float()
                        lora_scale = lora_alpha / rank
                        dW_src_norm = (lora_scale * (B_17 @ A_17)).norm().item()

                        if dW_tgt_norm > 1e-12 and dW_src_norm > 1e-12:
                            rescale = dW_src_norm / dW_tgt_norm
                            # Cap rescale for unmapped/deep layers to avoid instability
                            if tgt_j in UNMAPPED_TGT_LAYERS:
                                rescale = min(rescale, 20.0)
                            t = (B_8B * rescale).to(torch.bfloat16)
                new_lora[tgt_key] = t
            else:
                print(f"  WARNING: {src_key} not in source LoRA")

        del P_l  # free memory

    # ─────────────────────────────────────────────────────────────────────
    # Save adapter
    # ─────────────────────────────────────────────────────────────────────
    print(f"\nSaving {len(new_lora)} tensors to {OUTPUT_DIR} …")
    save_file(new_lora, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))

    # Write adapter_config.json for 8B
    tgt_cfg = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": TGT_MODEL_DIR,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False
    }
    with open(os.path.join(OUTPUT_DIR, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_cfg, f, indent=2)

    print("Done!")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
