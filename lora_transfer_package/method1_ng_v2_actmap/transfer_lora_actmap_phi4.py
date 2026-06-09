#!/usr/bin/env python3
"""
transfer_lora_actmap_phi4.py
=============================
Phi4-specific LoRA transfer (3.8B→14B) using ActMap/R_l method.
Handles fused projections: qkv_proj, gate_up_proj.

Key differences from generic script:
  - Phi4 uses fused QKV (qkv_proj) instead of separate q/k/v
  - Phi4 uses fused gate+up (gate_up_proj) instead of separate gate/up
  - P_l uses first half of gate_up_proj.weight
  - Hungarian head matching splits fused QKV weights
  - All tensor ops on GPU (--device) in float32

Usage:
  CUDA_VISIBLE_DEVICES=0 python transfer_lora_actmap_phi4.py \
    --src_model models/phi4_3_8B_tofu \
    --tgt_model models/phi4_14B_tofu \
    --src_lora checkpoints/tofu_gagd/phi4_3_8B_tofu_forget05/adapter \
    --output_dir lora_transfer_output/phi4_3.8B_to_14B/forget05/a80_c5 \
    --r_l_dir tmp/act_align_phi4_3.8B_14B_tofu \
    --hungarian_heads --layer_mapping_mode cka_monotonic \
    --cka_path tmp/phi4_3.8B_14B_CKA.pt \
    --norm_cap 5 --alpha_override 80 --device cuda
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


# ===========================================================================
# Helpers (same as generic)
# ===========================================================================

def build_shard_index(model_dir: str) -> dict:
    index_file = Path(model_dir) / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            idx = json.load(f)
        return {k: str(Path(model_dir) / v) for k, v in idx["weight_map"].items()}
    single = Path(model_dir) / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())
        return {k: str(single) for k in keys}
    raise FileNotFoundError(f"No safetensors in {model_dir}")


def load_tensor(shard_index: dict, key: str, dtype=torch.float32):
    path = shard_index[key]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def get_model_config(model_dir: str) -> dict:
    with open(Path(model_dir) / "config.json") as f:
        return json.load(f)


def proportional_layer_mapping(n_src: int, n_tgt: int) -> list:
    return [max(0, min(n_src - 1, int(round(j * n_src / n_tgt)))) for j in range(n_tgt)]


def cka_monotonic_mapping(cka: np.ndarray, n_src: int, n_tgt: int) -> list:
    S = cka
    n, m = S.shape
    dp = np.full((n, m), -float("inf"), dtype=np.float64)
    path = np.zeros((n, m), dtype=np.int32)
    for j in range(m):
        dp[0, j] = S[0, j]
    for i in range(1, n):
        for j in range(i, m):
            best_v = -float("inf")
            best_k = -1
            for k in range(i - 1, j):
                v = dp[i - 1, k] + S[i, j]
                if v > best_v:
                    best_v = v
                    best_k = k
            dp[i, j] = best_v
            path[i, j] = best_k
    best_sum = -float("inf")
    last_j = m - 1
    for j in range(n - 1, m):
        if dp[n - 1, j] > best_sum:
            best_sum = dp[n - 1, j]
            last_j = j
    mapping_src_to_tgt = [0] * n
    for i in range(n - 1, -1, -1):
        mapping_src_to_tgt[i] = int(last_j)
        last_j = int(path[i, last_j])
    tgt_to_src = [0] * n_tgt
    for i, tgt in enumerate(mapping_src_to_tgt):
        tgt_to_src[tgt] = i
    last_src = 0
    for j in range(n_tgt):
        if tgt_to_src[j] == 0 and j > 0:
            tgt_to_src[j] = last_src
        last_src = tgt_to_src[j]
    return tgt_to_src


def get_unmapped_layers(mapping: list, n_src: int) -> set:
    used = set()
    unmapped = set()
    for j, i in enumerate(mapping):
        if i in used:
            unmapped.add(j)
        used.add(i)
    return unmapped


# ===========================================================================
# P_l computation
# ===========================================================================

def compute_P(W_g_tgt, R_l, W_g_src, rcond=1e-2, col_clip=3.0, device="cuda"):
    """P_l = W_g_tgt @ R_l.T @ pinv(W_g_src). All on GPU, float32."""
    W_g_tgt = W_g_tgt.float().to(device)
    R_l = R_l.float().to(device)
    W_g_src = W_g_src.float().to(device)

    A = W_g_tgt @ R_l.T  # (inter_tgt, hidden_src)
    W_pinv = torch.linalg.pinv(W_g_src, rcond=rcond)  # (hidden_src, inter_src)
    P_l = A @ W_pinv  # (inter_tgt, inter_src)

    col_norms = P_l.norm(dim=0, keepdim=True).clamp(min=1e-8)
    median_norm = col_norms.median()
    max_norm = (col_clip * median_norm).clamp(min=1.0)
    scale = (max_norm / col_norms).clamp(max=1.0)
    P_l = (P_l * scale).cpu()
    return P_l


# ===========================================================================
# Hungarian head matching for fused QKV
# ===========================================================================

def compute_head_similarity_fused(
    W_qkv_old, W_o_old, W_qkv_new, W_o_new,
    n_q_old, n_q_new, hd_old, hd_new, R_l
):
    """
    Compute QK-interaction head similarity for fused QKV.
    W_qkv: (q_out + k_out + v_out, hidden) fused.
    """
    n_kv_old = (W_qkv_old.shape[0] - n_q_old * hd_old) // (2 * hd_old)
    n_kv_new = (W_qkv_new.shape[0] - n_q_new * hd_new) // (2 * hd_new)
    d_old = W_qkv_old.shape[1]
    d_new = W_qkv_new.shape[1]

    # Split QKV
    q_end_old = n_q_old * hd_old
    kv_end_old = q_end_old + n_kv_old * hd_old
    W_q_old = W_qkv_old[:q_end_old]
    W_k_old = W_qkv_old[q_end_old:kv_end_old]
    # W_v_old = W_qkv_old[kv_end_old:]

    q_end_new = n_q_new * hd_new
    kv_end_new = q_end_new + n_kv_new * hd_new
    W_q_new = W_qkv_new[:q_end_new]
    W_k_new = W_qkv_new[q_end_new:kv_end_new]

    W_qo_h = W_q_old.reshape(n_q_old, hd_old, d_old)
    W_ko_h = W_k_old.reshape(n_kv_old, hd_old, d_old)
    W_qn_h = W_q_new.reshape(n_q_new, hd_new, d_new)
    W_kn_h = W_k_new.reshape(n_kv_new, hd_new, d_new)

    qk_old = []
    for h in range(n_q_old):
        kv_h = min(h * n_kv_old // n_q_old, n_kv_old - 1)
        qk = W_qo_h[h] @ W_ko_h[kv_h].T
        qk_old.append(qk.flatten())
    qk_old = torch.stack(qk_old)

    qk_new = []
    for h in range(n_q_new):
        kv_h = min(h * n_kv_new // n_q_new, n_kv_new - 1)
        qk = W_qn_h[h] @ W_kn_h[kv_h].T
        qk_new.append(qk.flatten())
    qk_new = torch.stack(qk_new)

    qk_old_n = F.normalize(qk_old.float(), dim=1)
    qk_new_n = F.normalize(qk_new.float(), dim=1)
    sim = qk_old_n @ qk_new_n.T
    return sim.cpu().numpy()


def hungarian_head_match(sim_matrix: np.ndarray):
    from scipy.optimize import linear_sum_assignment
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return dict(zip(row_ind, col_ind))


# ===========================================================================
# Resize util
# ===========================================================================

def resize_output_dim(tensor: torch.Tensor, new_dim: int) -> torch.Tensor:
    old_dim = tensor.shape[0]
    if old_dim == new_dim:
        return tensor.contiguous()
    t = tensor.T.unsqueeze(0).float()
    t = F.interpolate(t, size=new_dim, mode="linear", align_corners=False)
    return t.squeeze(0).T.contiguous()


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Phi4-specific LoRA transfer via ActMap")
    p.add_argument("--src_model", required=True)
    p.add_argument("--tgt_model", required=True)
    p.add_argument("--src_lora", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--r_l_dir", required=True)
    p.add_argument("--hungarian_heads", action="store_true", default=False)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--p_l_rcond", type=float, default=1e-2)
    p.add_argument("--p_l_col_clip", type=float, default=3.0)
    p.add_argument("--norm_cap", type=float, default=20.0)
    p.add_argument("--alpha_override", type=float, default=None)
    p.add_argument("--layer_mapping_mode", type=str, default="proportional",
                   choices=["proportional", "cka_monotonic"])
    p.add_argument("--cka_path", type=str, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    SAVE_DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else DTYPE
    dev = torch.device(args.device) if args.device != "cpu" else torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Configs ──
    src_cfg = get_model_config(args.src_model)
    tgt_cfg = get_model_config(args.tgt_model)
    N_SRC = src_cfg["num_hidden_layers"]
    N_TGT = tgt_cfg["num_hidden_layers"]
    D_SRC = src_cfg["hidden_size"]
    D_TGT = tgt_cfg["hidden_size"]
    INTER_SRC = src_cfg["intermediate_size"]
    INTER_TGT = tgt_cfg["intermediate_size"]
    N_Q_SRC = src_cfg["num_attention_heads"]
    N_Q_TGT = tgt_cfg["num_attention_heads"]
    N_KV_SRC = src_cfg.get("num_key_value_heads", N_Q_SRC)
    N_KV_TGT = tgt_cfg.get("num_key_value_heads", N_Q_TGT)
    HD_SRC = src_cfg.get("head_dim", D_SRC // N_Q_SRC)
    HD_TGT = tgt_cfg.get("head_dim", D_TGT // N_Q_TGT)

    Q_OUT_SRC = N_Q_SRC * HD_SRC
    Q_OUT_TGT = N_Q_TGT * HD_TGT
    KV_OUT_SRC = N_KV_SRC * HD_SRC
    KV_OUT_TGT = N_KV_TGT * HD_TGT
    QKV_OUT_SRC = Q_OUT_SRC + 2 * KV_OUT_SRC
    QKV_OUT_TGT = Q_OUT_TGT + 2 * KV_OUT_TGT

    print(f"Source: {N_SRC}L, h={D_SRC}, inter={INTER_SRC}, "
          f"Q={N_Q_SRC}, KV={N_KV_SRC}, hd={HD_SRC}")
    print(f"Target: {N_TGT}L, h={D_TGT}, inter={INTER_TGT}, "
          f"Q={N_Q_TGT}, KV={N_KV_TGT}, hd={HD_TGT}")

    # ── Layer mapping ──
    if args.layer_mapping_mode == "cka_monotonic" and args.cka_path and Path(args.cka_path).exists():
        cka = torch.load(args.cka_path, map_location="cpu").numpy()
        layer_mapping = cka_monotonic_mapping(cka, N_SRC, N_TGT)
        print(f"CKA monotonic mapping: {layer_mapping}")
    else:
        layer_mapping = proportional_layer_mapping(N_SRC, N_TGT)
    unmapped = get_unmapped_layers(layer_mapping, N_SRC)
    print(f"Layer mapping (tgt->src): {layer_mapping}")
    print(f"Unmapped target layers: {sorted(unmapped)}")

    # ── Load source LoRA ──
    print("Loading source LoRA...")
    lora_src = {}
    lora_path = Path(args.src_lora) / "adapter_model.safetensors"
    with safe_open(str(lora_path), framework="pt") as f:
        for k in f.keys():
            lora_src[k] = f.get_tensor(k).float()

    with open(Path(args.src_lora) / "adapter_config.json") as f:
        src_lora_cfg = json.load(f)
    rank = src_lora_cfg["r"]
    lora_alpha = args.alpha_override if args.alpha_override else src_lora_cfg["lora_alpha"]
    print(f"LoRA: rank={rank}, alpha={lora_alpha}")

    # ── Load R_l ──
    print("Loading R_l matrices...")
    R_l_dict = {}
    for i in range(N_SRC):
        rpath = Path(args.r_l_dir) / f"R_l_{i}.pt"
        if rpath.exists():
            R_l_dict[i] = torch.load(str(rpath), map_location="cpu").float()
        else:
            print(f"  WARNING: R_l_{i}.pt not found")

    # ── Shard indices ──
    idx_src = build_shard_index(args.src_model)
    idx_tgt = build_shard_index(args.tgt_model)

    # ── Hungarian head mappings ──
    head_mappings = {}
    if args.hungarian_heads:
        print("Computing Hungarian head mappings (fused QKV)...")
        for tgt_j in range(N_TGT):
            src_i = layer_mapping[tgt_j]
            if src_i not in R_l_dict:
                continue
            R_l = R_l_dict[src_i]

            try:
                W_qkv_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.qkv_proj.weight")
                W_o_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.o_proj.weight")
                W_qkv_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.qkv_proj.weight")
                W_o_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.o_proj.weight")

                sim = compute_head_similarity_fused(
                    W_qkv_old, W_o_old, W_qkv_new, W_o_new,
                    N_Q_SRC, N_Q_TGT, HD_SRC, HD_TGT, R_l)
                hmap = hungarian_head_match(sim)
                head_mappings[tgt_j] = hmap
            except Exception as e:
                print(f"  Head matching failed for L{tgt_j}: {e}")
        print(f"  Computed {len(head_mappings)} head mappings")

    # ── Process each target layer ──
    new_lora = {}
    TGT_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    for tgt_j in range(N_TGT):
        src_i = layer_mapping[tgt_j]
        if src_i not in R_l_dict:
            print(f"  SKIP L{tgt_j}: no R_l for L{src_i}")
            continue

        R_l = R_l_dict[src_i]

        # Compute P_l
        if INTER_SRC != INTER_TGT:
            # Phi4: use first half of gate_up_proj as gate_proj
            gate_up_src = load_tensor(idx_src, f"model.layers.{src_i}.mlp.gate_up_proj.weight")
            gate_up_tgt = load_tensor(idx_tgt, f"model.layers.{tgt_j}.mlp.gate_up_proj.weight")
            W_g_src = gate_up_src[:INTER_SRC, :]  # first half = gate
            W_g_tgt = gate_up_tgt[:INTER_TGT, :]
            P_l = compute_P(W_g_tgt, R_l, W_g_src,
                            rcond=args.p_l_rcond, col_clip=args.p_l_col_clip,
                            device=args.device)
        else:
            P_l = None

        print(f"Tgt L{tgt_j:2d} ← Src L{src_i}  "
              f"(R_l={tuple(R_l.shape)}, P_l={tuple(P_l.shape) if P_l is not None else 'None'})")

        # ── Helper closures on GPU ──
        def transform_A_hidden(A):
            return (A.float().to(dev) @ R_l.to(dev)).cpu().to(SAVE_DTYPE).contiguous()

        def transform_B_hidden(B):
            return (R_l.to(dev).T @ B.float().to(dev)).cpu().to(SAVE_DTYPE).contiguous()

        def transform_A_inter(A):
            if P_l is not None:
                return (A.float().to(dev) @ P_l.to(dev).T).cpu().to(SAVE_DTYPE).contiguous()
            return A.to(SAVE_DTYPE).contiguous()

        def transform_B_inter(B):
            if P_l is not None:
                return (P_l.to(dev) @ B.float().to(dev)).cpu().to(SAVE_DTYPE).contiguous()
            return B.to(SAVE_DTYPE).contiguous()

        src_pfx = f"base_model.model.model.layers.{src_i}"
        tgt_pfx = f"base_model.model.model.layers.{tgt_j}"

        # ═══════════════════════════════════════════════════════
        # qkv_proj (fused): split lora_B into Q/K/V, transform, reassemble
        # ═══════════════════════════════════════════════════════
        a_src = f"{src_pfx}.self_attn.qkv_proj.lora_A.weight"
        b_src = f"{src_pfx}.self_attn.qkv_proj.lora_B.weight"
        if a_src in lora_src and b_src in lora_src:
            A_qkv = lora_src[a_src]   # (rank, D_src)
            B_qkv = lora_src[b_src]   # (QKV_out_src, rank)

            # A: simple hidden transform (same as q_proj)
            A_tgt = transform_A_hidden(A_qkv)

            # B: split into Q, K, V parts
            B_q = B_qkv[:Q_OUT_SRC]                    # (Q_out_src, rank)
            B_k = B_qkv[Q_OUT_SRC:Q_OUT_SRC + KV_OUT_SRC]  # (KV_out_src, rank)
            B_v = B_qkv[Q_OUT_SRC + KV_OUT_SRC:]        # (KV_out_src, rank)

            # Hungarian head remapping for Q
            if args.hungarian_heads and tgt_j in head_mappings:
                hmap = head_mappings[tgt_j]
                B_q_h = B_q.float().reshape(N_Q_SRC, HD_SRC, rank)
                B_q_tgt = torch.zeros(N_Q_TGT, HD_TGT, rank, dtype=torch.float32)
                for old_h, new_h in hmap.items():
                    if old_h >= N_Q_SRC or new_h >= N_Q_TGT:
                        continue
                    if HD_SRC == HD_TGT:
                        B_q_tgt[new_h] = B_q_h[old_h]
                    else:
                        bt = B_q_h[old_h].T.unsqueeze(0)  # (1, rank, hd_src)
                        bt = F.interpolate(bt, size=HD_TGT, mode="linear", align_corners=False)
                        B_q_tgt[new_h] = bt.squeeze(0).T
                B_q_new = B_q_tgt.reshape(Q_OUT_TGT, rank)
            else:
                # Fallback: resize output dim proportionally
                B_q_new = resize_output_dim(B_q, Q_OUT_TGT)

            # KV: proportional resize
            B_k_new = resize_output_dim(B_k, KV_OUT_TGT)
            B_v_new = resize_output_dim(B_v, KV_OUT_TGT)

            # Reassemble fused B
            B_tgt = torch.cat([B_q_new, B_k_new, B_v_new], dim=0)

            a_tgt_key = f"{tgt_pfx}.self_attn.qkv_proj.lora_A.weight"
            b_tgt_key = f"{tgt_pfx}.self_attn.qkv_proj.lora_B.weight"

            # Norm matching
            dW_tgt = (B_tgt.float() @ A_tgt.float()).norm().item()
            dW_src = (lora_alpha / rank * (B_qkv.float() @ A_qkv.float())).norm().item()
            if dW_tgt > 1e-12 and dW_src > 1e-12:
                rescale = min(dW_src / dW_tgt, args.norm_cap) if tgt_j in unmapped else dW_src / dW_tgt
                B_tgt = (B_tgt.float() * rescale).to(SAVE_DTYPE).contiguous()
            else:
                B_tgt = B_tgt.to(SAVE_DTYPE).contiguous()

            new_lora[a_tgt_key] = A_tgt
            new_lora[b_tgt_key] = B_tgt

        # ═══════════════════════════════════════════════════════
        # o_proj: standard hidden↔hidden transform
        # ═══════════════════════════════════════════════════════
        a_src = f"{src_pfx}.self_attn.o_proj.lora_A.weight"
        b_src = f"{src_pfx}.self_attn.o_proj.lora_B.weight"
        if a_src in lora_src and b_src in lora_src:
            A_o = lora_src[a_src]
            B_o = lora_src[b_src]
            A_tgt = transform_A_hidden(A_o)
            B_tgt = transform_B_hidden(B_o)

            dW_tgt = (B_tgt.float() @ A_tgt.float()).norm().item()
            dW_src = (lora_alpha / rank * (B_o.float() @ A_o.float())).norm().item()
            if dW_tgt > 1e-12 and dW_src > 1e-12:
                rescale = min(dW_src / dW_tgt, args.norm_cap) if tgt_j in unmapped else dW_src / dW_tgt
                B_tgt = (B_tgt.float() * rescale).to(SAVE_DTYPE).contiguous()

            new_lora[f"{tgt_pfx}.self_attn.o_proj.lora_A.weight"] = A_tgt
            new_lora[f"{tgt_pfx}.self_attn.o_proj.lora_B.weight"] = B_tgt

        # ═══════════════════════════════════════════════════════
        # gate_up_proj (fused): split lora_B into gate/up, transform with P_l
        # ═══════════════════════════════════════════════════════
        a_src = f"{src_pfx}.mlp.gate_up_proj.lora_A.weight"
        b_src = f"{src_pfx}.mlp.gate_up_proj.lora_B.weight"
        if a_src in lora_src and b_src in lora_src:
            A_gu = lora_src[a_src]   # (rank, D_src)
            B_gu = lora_src[b_src]   # (2*inter_src, rank)

            # A: hidden transform
            A_tgt = transform_A_hidden(A_gu)

            # B: split into gate and up halves, apply P_l
            B_gate = B_gu[:INTER_SRC]   # (inter_src, rank)
            B_up   = B_gu[INTER_SRC:]   # (inter_src, rank)

            if P_l is not None:
                B_gate_tgt = (P_l.to(dev) @ B_gate.float().to(dev)).cpu().to(SAVE_DTYPE).contiguous()
                B_up_tgt   = (P_l.to(dev) @ B_up.float().to(dev)).cpu().to(SAVE_DTYPE).contiguous()
            else:
                B_gate_tgt = B_gate.to(SAVE_DTYPE).contiguous()
                B_up_tgt = B_up.to(SAVE_DTYPE).contiguous()

            B_tgt = torch.cat([B_gate_tgt, B_up_tgt], dim=0)

            dW_tgt = (B_tgt.float() @ A_tgt.float()).norm().item()
            dW_src = (lora_alpha / rank * (B_gu.float() @ A_gu.float())).norm().item()
            if dW_tgt > 1e-12 and dW_src > 1e-12:
                rescale = min(dW_src / dW_tgt, args.norm_cap) if tgt_j in unmapped else dW_src / dW_tgt
                B_tgt = (B_tgt.float() * rescale).to(SAVE_DTYPE).contiguous()

            new_lora[f"{tgt_pfx}.mlp.gate_up_proj.lora_A.weight"] = A_tgt
            new_lora[f"{tgt_pfx}.mlp.gate_up_proj.lora_B.weight"] = B_tgt

        # ═══════════════════════════════════════════════════════
        # down_proj: inter→hidden transform
        # ═══════════════════════════════════════════════════════
        a_src = f"{src_pfx}.mlp.down_proj.lora_A.weight"
        b_src = f"{src_pfx}.mlp.down_proj.lora_B.weight"
        if a_src in lora_src and b_src in lora_src:
            A_dn = lora_src[a_src]   # (rank, inter_src)
            B_dn = lora_src[b_src]   # (D_src, rank)

            A_tgt = transform_A_inter(A_dn)
            B_tgt = transform_B_hidden(B_dn)

            dW_tgt = (B_tgt.float() @ A_tgt.float()).norm().item()
            dW_src = (lora_alpha / rank * (B_dn.float() @ A_dn.float())).norm().item()
            if dW_tgt > 1e-12 and dW_src > 1e-12:
                rescale = min(dW_src / dW_tgt, args.norm_cap) if tgt_j in unmapped else dW_src / dW_tgt
                B_tgt = (B_tgt.float() * rescale).to(SAVE_DTYPE).contiguous()

            new_lora[f"{tgt_pfx}.mlp.down_proj.lora_A.weight"] = A_tgt
            new_lora[f"{tgt_pfx}.mlp.down_proj.lora_B.weight"] = B_tgt

        # Cleanup
        del R_l
        if P_l is not None:
            del P_l

    # ── Save ──
    print(f"\nSaving {len(new_lora)} tensors to {args.output_dir} ...")
    save_file(new_lora, os.path.join(args.output_dir, "adapter_model.safetensors"))

    tgt_mods = set()
    for k in new_lora.keys():
        parts = k.split(".")
        for idx, pt in enumerate(parts):
            if pt in ("lora_A", "lora_B"):
                tgt_mods.add(parts[idx - 1])
                break

    tgt_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": args.tgt_model,
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
        "target_modules": sorted(tgt_mods),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
