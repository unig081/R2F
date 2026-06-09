#!/usr/bin/env python3
"""
transfer_lora_actmap_generic.py
===============================
Generic ActMap v4 LoRA transfer: source model → target model (same family, different size).

Supports:
  - Multiple model families (llama, phi4, qwen3)
  - Proportional layer mapping (default) or CKA-based
  - Hungarian head matching (optional, --hungarian_heads)
  - head_dim change handling (e.g. llama 64→128)
  - KV head count change handling (e.g. phi4 8→10)
  - Per-module norm matching (v4 core)

Hyperparameters worth sweeping:
  --r_l_reg          Ridge for R_l computation (default 1e-3)
  --p_l_rcond        rcond for pinv in P_l computation (default 1e-2)
  --norm_cap         Norm rescale cap for unmapped layers (default 20.0)
  --p_l_col_clip     P_l column norm clip multiplier (default 3.0)
  --alpha_scale      Override lora_alpha scaling (default: keep same alpha/r ratio)

Usage:
  python transfer_lora_actmap_generic.py \
    --src_model models/llama_3_2_1B_instruct_tofu \
    --tgt_model models/llama_3_2_3B_instruct_tofu \
    --src_lora checkpoints/tofu_gagd/llama_3_2_1B_tofu_forget05/adapter \
    --output_dir lora_transfer_output/llama_1B_to_3B_forget05_actmap \
    --r_l_dir tmp/act_align_llama_1B_3B \
    --hungarian_heads
"""

import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_shard_index(model_dir: str) -> dict:
    """Return dict: tensor_name -> safetensors file path."""
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
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def load_tensor(shard_index: dict, key: str, dtype=torch.float32):
    """Load a single tensor from the sharded model."""
    path = shard_index[key]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def get_model_config(model_dir: str) -> dict:
    """Read model config.json."""
    with open(Path(model_dir) / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------

def proportional_layer_mapping(n_src: int, n_tgt: int) -> list:
    """Proportional layer mapping: target j -> nearest source i."""
    return [max(0, min(n_src - 1, int(round(j * n_src / n_tgt)))) for j in range(n_tgt)]


def cka_monotonic_mapping(cka: "np.ndarray", n_src: int, n_tgt: int) -> list:
    """CKA monotonic DP mapping (from collect_activations.py logic)."""
    import numpy as np
    S = cka  # (n_src, n_tgt)
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

    # Invert: tgt -> src
    tgt_to_src = [0] * n_tgt
    for i, tgt in enumerate(mapping_src_to_tgt):
        tgt_to_src[tgt] = i
    # Fill gaps
    last_src = 0
    for j in range(n_tgt):
        if tgt_to_src[j] == 0 and j > 0:
            tgt_to_src[j] = last_src
        last_src = tgt_to_src[j]
    return tgt_to_src


def cka_hungarian_mapping(cka: "np.ndarray", n_src: int, n_tgt: int) -> list:
    """CKA Hungarian 1-to-1 mapping."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    cost = -cka
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = [0] * n_src
    for r, c in zip(row_ind, col_ind):
        mapping[int(r)] = int(c)
    # Invert
    tgt_to_src = [0] * n_tgt
    for i, tgt in enumerate(mapping):
        tgt_to_src[tgt] = i
    last_src = 0
    for j in range(n_tgt):
        if tgt_to_src[j] == 0 and j > 0:
            tgt_to_src[j] = last_src
        last_src = tgt_to_src[j]
    return tgt_to_src


def get_unmapped_layers(mapping: list, n_src: int) -> set:
    """Find target layers that are unique (no other target maps to same source)."""
    src_usage = {}
    for j, i in enumerate(mapping):
        src_usage.setdefault(i, []).append(j)
    # Layers with >1 target mapping to the same source are "unmapped" in a sense
    unmapped = set()
    for i, js in src_usage.items():
        if len(js) > 1:
            # Mark all but the first as unmapped
            for j in js[1:]:
                unmapped.add(j)
    return unmapped


# ---------------------------------------------------------------------------
# Hungarian head matching (from LoRASuite logic)
# ---------------------------------------------------------------------------

def compute_head_similarity(W_q_old, W_k_old, W_v_old, W_o_old,
                            W_q_new, W_k_new, W_v_new, W_o_new,
                            n_heads_old, n_heads_new, head_dim_old, head_dim_new,
                            R_l):
    """
    Compute head-to-head similarity matrix based on QK + VO interaction matrices.
    Returns (n_heads_old, n_heads_new) similarity matrix.
    """
    # Reshape Q/K/V/O weights to per-head
    # Q: (n_heads * head_dim, hidden)
    d_old = W_q_old.shape[1]
    d_new = W_q_new.shape[1]

    W_qo_h = W_q_old.reshape(n_heads_old, head_dim_old, d_old)  # (H_o, hd_o, d_o)
    W_ko_h = W_k_old.reshape(-1, head_dim_old, d_old)           # num_kv may differ
    # For simplicity, handle KV by adjusting heads
    n_kv_old = W_ko_h.shape[0]
    n_kv_new = W_k_new.reshape(-1, head_dim_new, d_new).shape[0]

    W_qn_h = W_q_new.reshape(n_heads_new, head_dim_new, d_new)
    W_kn_h = W_k_new.reshape(-1, head_dim_new, d_new)

    # Build QK interaction per head: W_Q @ W_K^T
    # For GQA, expand KV heads to match Q heads
    qk_old = []
    for h in range(n_heads_old):
        kv_h = min(h * n_kv_old // n_heads_old, n_kv_old - 1)
        qk = W_qo_h[h] @ W_ko_h[kv_h].T  # (hd_o, hd_o)
        qk_old.append(qk.flatten())
    qk_old = torch.stack(qk_old)  # (n_heads_old, hd_o*hd_o)

    qk_new = []
    for h in range(n_heads_new):
        kv_h = min(h * n_kv_new // n_heads_new, n_kv_new - 1)
        qk = W_qn_h[h] @ W_kn_h[kv_h].T
        qk_new.append(qk.flatten())
    qk_new = torch.stack(qk_new)

    if qk_old.shape[1] != qk_new.shape[1]:
        common_dim = max(qk_old.shape[1], qk_new.shape[1])
        qk_old = F.interpolate(
            qk_old.unsqueeze(0), size=common_dim, mode="linear", align_corners=False
        ).squeeze(0)
        qk_new = F.interpolate(
            qk_new.unsqueeze(0), size=common_dim, mode="linear", align_corners=False
        ).squeeze(0)

    # Cosine similarity
    qk_old_n = F.normalize(qk_old.float(), dim=1)
    qk_new_n = F.normalize(qk_new.float(), dim=1)
    sim = qk_old_n @ qk_new_n.T  # (n_heads_old, n_heads_new)
    return sim.cpu().numpy()


def hungarian_head_match(sim_matrix: np.ndarray, n_heads_new: int):
    """Run Hungarian algorithm for 1-to-1 head matching."""
    from scipy.optimize import linear_sum_assignment
    # Maximize similarity = minimize -similarity
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return dict(zip(row_ind, col_ind))


# ---------------------------------------------------------------------------
# P_l computation
# ---------------------------------------------------------------------------

def compute_P(W_g_tgt, R_l, W_g_src, rcond=1e-2, col_clip=3.0, device="cuda"):
    """
    P_l = W_g_tgt @ R_l.T @ pinv(W_g_src).  All on GPU for speed.
    Memory: ~600MB for phi4 (17920×8192) in float32.  Fine on 48GB.
    """
    W_g_tgt = W_g_tgt.float().to(device)
    R_l = R_l.float().to(device)
    W_g_src = W_g_src.float().to(device)

    A = W_g_tgt @ R_l.T  # (inter_tgt, hidden_src)
    W_pinv = torch.linalg.pinv(W_g_src, rcond=rcond)  # (hidden_src, inter_src)
    P_l = A @ W_pinv  # (inter_tgt, inter_src)

    # Per-column norm clip
    col_norms = P_l.norm(dim=0, keepdim=True).clamp(min=1e-8)
    median_norm = col_norms.median()
    max_norm = (col_clip * median_norm).clamp(min=1.0)
    scale = (max_norm / col_norms).clamp(max=1.0)
    P_l = (P_l * scale).cpu()  # move back to CPU for saving
    return P_l


# ---------------------------------------------------------------------------
# Output-dim mapping for k/v when head_dim changes (e.g. llama 64→128)
# ---------------------------------------------------------------------------

def resize_output_dim(tensor: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Resize tensor along dim 0 via linear interpolation. tensor: (old_dim, rank)."""
    old_dim = tensor.shape[0]
    if old_dim == new_dim:
        return tensor.contiguous()
    t = tensor.T.unsqueeze(0).float()  # (1, rank, old_dim)
    t = F.interpolate(t, size=new_dim, mode="linear", align_corners=False)
    return t.squeeze(0).T.contiguous()


# ---------------------------------------------------------------------------
# Main transfer
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generic ActMap v4 LoRA Transfer")
    p.add_argument("--src_model", required=True)
    p.add_argument("--tgt_model", required=True)
    p.add_argument("--src_lora", required=True, help="Path to source LoRA adapter dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--r_l_dir", required=True, help="Dir with pre-computed R_l_{i}.pt files")
    p.add_argument("--hungarian_heads", action="store_true", default=False,
                   help="Enable Hungarian head matching for attention modules")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--p_l_rcond", type=float, default=1e-2)
    p.add_argument("--p_l_col_clip", type=float, default=3.0)
    p.add_argument("--norm_cap", type=float, default=20.0,
                   help="Max norm rescale for unmapped layers")
    p.add_argument("--alpha_override", type=float, default=None,
                   help="Override lora_alpha (default: keep same)")
    p.add_argument("--target_modules", type=str, default=None,
                   help="Comma-separated list, e.g. 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'")
    p.add_argument("--layer_mapping_mode", type=str, default="proportional",
                   choices=["proportional", "cka_monotonic", "cka_hungarian"],
                   help="Layer mapping strategy")
    p.add_argument("--cka_path", type=str, default=None,
                   help="Path to CKA matrix .pt (required for cka_* mapping modes)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    SAVE_DTYPE = torch.bfloat16 if args.dtype == "bfloat16" else DTYPE

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load configs ──
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

    KV_OUT_SRC = N_KV_SRC * HD_SRC
    KV_OUT_TGT = N_KV_TGT * HD_TGT
    Q_OUT_SRC = N_Q_SRC * HD_SRC
    Q_OUT_TGT = N_Q_TGT * HD_TGT

    print(f"Source: {N_SRC}L, hidden={D_SRC}, inter={INTER_SRC}, "
          f"Q_heads={N_Q_SRC}, KV_heads={N_KV_SRC}, head_dim={HD_SRC}")
    print(f"Target: {N_TGT}L, hidden={D_TGT}, inter={INTER_TGT}, "
          f"Q_heads={N_Q_TGT}, KV_heads={N_KV_TGT}, head_dim={HD_TGT}")

    # ── Target modules ──
    if args.target_modules:
        MODULES = [m.strip() for m in args.target_modules.split(",")]
    else:
        MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

    # ── Layer mapping ──
    if args.layer_mapping_mode == "cka_monotonic" or args.layer_mapping_mode == "cka_hungarian":
        if not args.cka_path or not Path(args.cka_path).exists():
            print(f"WARNING: CKA path not found, falling back to proportional")
            layer_mapping = proportional_layer_mapping(N_SRC, N_TGT)
        else:
            cka = torch.load(args.cka_path, map_location="cpu").numpy()
            if args.layer_mapping_mode == "cka_monotonic":
                layer_mapping = cka_monotonic_mapping(cka, N_SRC, N_TGT)
                print(f"CKA monotonic mapping: {layer_mapping}")
            else:
                layer_mapping = cka_hungarian_mapping(cka, N_SRC, N_TGT)
                print(f"CKA hungarian mapping: {layer_mapping}")
    else:
        layer_mapping = proportional_layer_mapping(N_SRC, N_TGT)
    unmapped = get_unmapped_layers(layer_mapping, N_SRC)
    print(f"Layer mapping (tgt->src): {layer_mapping}")
    print(f"Unmapped target layers: {sorted(unmapped)}")

    # ── Load source LoRA ──
    print("Loading source LoRA…")
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

    # ── Load R_l matrices ──
    print("Loading R_l matrices…")
    R_l_dict = {}
    for i in range(N_SRC):
        rpath = Path(args.r_l_dir) / f"R_l_{i}.pt"
        if rpath.exists():
            R_l_dict[i] = torch.load(str(rpath), map_location="cpu").float()
        else:
            print(f"  WARNING: R_l_{i}.pt not found")

    # ── Build shard indices for base model weights ──
    idx_src = build_shard_index(args.src_model)
    idx_tgt = build_shard_index(args.tgt_model)

    # ── Pre-compute Hungarian head mappings ──
    head_mappings = {}  # (tgt_layer, proj) -> dict{old_head: new_head}
    if args.hungarian_heads:
        print("Computing Hungarian head mappings…")
        for tgt_j in range(N_TGT):
            src_i = layer_mapping[tgt_j]
            if src_i not in R_l_dict:
                continue
            R_l = R_l_dict[src_i]

            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if proj not in MODULES:
                    continue
                try:
                    if proj == "q_proj":
                        W_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.q_proj.weight")
                        W_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.q_proj.weight")
                    elif proj == "k_proj":
                        W_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.k_proj.weight")
                        W_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.k_proj.weight")
                    elif proj == "v_proj":
                        W_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.v_proj.weight")
                        W_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.v_proj.weight")
                    elif proj == "o_proj":
                        W_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.o_proj.weight")
                        W_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.o_proj.weight")

                    # For simplicity, use Q weight for head matching
                    W_q_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.q_proj.weight")
                    W_k_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.k_proj.weight")
                    W_v_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.v_proj.weight")
                    W_o_old = load_tensor(idx_src, f"model.layers.{src_i}.self_attn.o_proj.weight")
                    W_q_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.q_proj.weight")
                    W_k_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.k_proj.weight")
                    W_v_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.v_proj.weight")
                    W_o_new = load_tensor(idx_tgt, f"model.layers.{tgt_j}.self_attn.o_proj.weight")

                    sim = compute_head_similarity(
                        W_q_old, W_k_old, W_v_old, W_o_old,
                        W_q_new, W_k_new, W_v_new, W_o_new,
                        N_Q_SRC, N_Q_TGT, HD_SRC, HD_TGT, R_l)
                    hmap = hungarian_head_match(sim, N_Q_TGT)
                    head_mappings[(tgt_j, proj)] = hmap
                except Exception as e:
                    print(f"  Head matching failed for L{tgt_j} {proj}: {e}")
        print(f"  Computed {len(head_mappings)} head mappings")

    # ── Process each target layer ──
    new_lora = {}

    for tgt_j in range(N_TGT):
        src_i = layer_mapping[tgt_j]
        if src_i not in R_l_dict:
            print(f"  SKIP target layer {tgt_j}: no R_l for source layer {src_i}")
            continue

        R_l = R_l_dict[src_i].to(args.device)

        # Compute P_l if needed (when intermediate sizes differ)
        if INTER_SRC != INTER_TGT:
            src_gate_key = f"model.layers.{src_i}.mlp.gate_proj.weight"
            tgt_gate_key = f"model.layers.{tgt_j}.mlp.gate_proj.weight"
            W_g_src = load_tensor(idx_src, src_gate_key)
            W_g_tgt = load_tensor(idx_tgt, tgt_gate_key)
            P_l = compute_P(W_g_tgt, R_l, W_g_src,
                            rcond=args.p_l_rcond, col_clip=args.p_l_col_clip,
                            device=args.device)
        else:
            P_l = None  # Same intermediate size, skip P_l

        print(f"Tgt L{tgt_j:2d} ← Src L{src_i}  "
              f"(R_l={tuple(R_l.shape)}, P_l={tuple(P_l.shape) if P_l is not None else 'None'})")

        # ── Helper closures (use GPU for speed) ──
        dev = torch.device(args.device) if args.device != "cpu" else torch.device("cpu")

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

        def copy_tensor(T):
            return T.to(SAVE_DTYPE).contiguous()

        # Key prefixes
        src_pfx = f"base_model.model.model.layers.{src_i}"
        tgt_pfx = f"base_model.model.model.layers.{tgt_j}"

        # Define per-module transform rules
        # Each: (src_suffix, tgt_suffix, transform_fn_for_A, transform_fn_for_B?, special_handling)
        MODULE_RULES = {}

        # q_proj: output dim = N_Q * head_dim
        if "q_proj" in MODULES:
            if Q_OUT_SRC == Q_OUT_TGT:
                MODULE_RULES["q_proj"] = (transform_A_hidden, transform_B_hidden)
            else:
                # Q output dim changed - use Hungarian or resize
                if args.hungarian_heads and (tgt_j, "q_proj") in head_mappings:
                    MODULE_RULES["q_proj"] = ("hungarian_q",)
                else:
                    # Use hidden-side transform; B side needs output dim handling
                    # q_proj output = hidden for standard architectures
                    if Q_OUT_SRC == D_SRC and Q_OUT_TGT == D_TGT:
                        MODULE_RULES["q_proj"] = (transform_A_hidden, transform_B_hidden)
                    else:
                        MODULE_RULES["q_proj"] = (transform_A_hidden, "resize_B_q")

        # k_proj: output dim = N_KV * head_dim
        if "k_proj" in MODULES:
            if KV_OUT_SRC == KV_OUT_TGT:
                MODULE_RULES["k_proj"] = (transform_A_hidden, copy_tensor)
            else:
                MODULE_RULES["k_proj"] = (transform_A_hidden, "resize_B_kv")

        # v_proj: output dim = N_KV * head_dim
        if "v_proj" in MODULES:
            if KV_OUT_SRC == KV_OUT_TGT:
                MODULE_RULES["v_proj"] = (transform_A_hidden, copy_tensor)
            else:
                MODULE_RULES["v_proj"] = (transform_A_hidden, "resize_B_kv")

        # o_proj: output dim = hidden
        if "o_proj" in MODULES:
            MODULE_RULES["o_proj"] = (transform_A_hidden, transform_B_hidden)

        # FFN modules
        if "gate_proj" in MODULES:
            MODULE_RULES["gate_proj"] = (transform_A_hidden, transform_B_inter)
        if "up_proj" in MODULES:
            MODULE_RULES["up_proj"] = (transform_A_hidden, transform_B_inter)
        if "down_proj" in MODULES:
            MODULE_RULES["down_proj"] = (transform_A_inter, transform_B_hidden)

        # ── Apply transforms ──
        for proj, rule in MODULE_RULES.items():
            a_src_key = f"{src_pfx}.self_attn.{proj}.lora_A.weight"
            b_src_key = f"{src_pfx}.self_attn.{proj}.lora_B.weight"
            # Try alternate prefix
            if proj in ("gate_proj", "up_proj", "down_proj"):
                a_src_key = f"{src_pfx}.mlp.{proj}.lora_A.weight"
                b_src_key = f"{src_pfx}.mlp.{proj}.lora_B.weight"
            a_tgt_key = f"{tgt_pfx}.self_attn.{proj}.lora_A.weight"
            b_tgt_key = f"{tgt_pfx}.self_attn.{proj}.lora_B.weight"
            if proj in ("gate_proj", "up_proj", "down_proj"):
                a_tgt_key = f"{tgt_pfx}.mlp.{proj}.lora_A.weight"
                b_tgt_key = f"{tgt_pfx}.mlp.{proj}.lora_B.weight"

            if a_src_key not in lora_src or b_src_key not in lora_src:
                continue

            A_src = lora_src[a_src_key]
            B_src = lora_src[b_src_key]

            if rule == ("hungarian_q",):
                # Hungarian head matching for Q: split by head, remap, reassemble
                A_tgt, B_tgt = _hungarian_q_transform(
                    A_src, B_src, head_mappings[(tgt_j, "q_proj")],
                    N_Q_SRC, N_Q_TGT, HD_SRC, HD_TGT, R_l)
            else:
                fn_A, fn_B = rule
                if fn_A == "resize_B_q":
                    # Only applicable if q output dim not equal hidden
                    A_tgt = transform_A_hidden(A_src)
                    B_tgt = resize_output_dim(B_src, Q_OUT_TGT)
                else:
                    A_tgt = fn_A(A_src)
                    if fn_B == "resize_B_kv":
                        B_tgt = resize_output_dim(B_src, KV_OUT_TGT)
                    else:
                        B_tgt = fn_B(B_src)

            # ── Per-module norm matching ──
            dW_tgt_norm = (B_tgt.float() @ A_tgt.float()).norm().item()
            dW_src_norm = (lora_alpha / rank * (B_src.float() @ A_src.float())).norm().item()

            if dW_tgt_norm > 1e-12 and dW_src_norm > 1e-12:
                rescale = dW_src_norm / dW_tgt_norm
                if tgt_j in unmapped:
                    rescale = min(rescale, args.norm_cap)
                B_tgt = (B_tgt.float() * rescale).to(SAVE_DTYPE).contiguous()

            new_lora[a_tgt_key] = A_tgt
            new_lora[b_tgt_key] = B_tgt

        # Cleanup
        del R_l
        if P_l is not None:
            del P_l

    # ── Save adapter ──
    print(f"\nSaving {len(new_lora)} tensors to {args.output_dir} …")
    # Duplicate target layers can intentionally reuse transformed source tensors.
    # safetensors rejects shared storage, so detach each entry before writing.
    new_lora = {k: v.clone().contiguous() for k, v in new_lora.items()}
    save_file(new_lora, os.path.join(args.output_dir, "adapter_model.safetensors"))

    # Collect target module names from output keys
    tgt_mods = set()
    for k in new_lora.keys():
        # k format: base_model.model.model.layers.{i}.{self_attn|mlp}.{proj}.lora_{A,B}.weight
        parts = k.split(".")
        # Find the projection name (the part before lora_A/lora_B)
        for idx, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                tgt_mods.add(parts[idx - 1])
                break

    tgt_cfg = {
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
        json.dump(tgt_cfg, f, indent=2)

    print("Done!")


def _hungarian_q_transform(A_src, B_src, head_map, n_q_src, n_q_tgt, hd_src, hd_tgt, R_l):
    """
    Hungarian-based Q transform: split lora_B by head, remap, handle head_dim change.
    A_src: (rank, hidden_src)
    B_src: (q_out_src, rank) = (n_q_src * hd_src, rank)
    """
    rank = A_src.shape[0]
    d_src = A_src.shape[1]
    d_tgt = R_l.shape[1]
    dev = R_l.device

    # Transform A via R_l (both on same device)
    A_tgt = (A_src.float().to(dev) @ R_l.float()).to(A_src.dtype)  # (rank, hidden_tgt)

    # Split B by head
    B_per_head_src = B_src.float().to(dev).reshape(n_q_src, hd_src, rank)  # (n_q, hd_src, rank)

    # Build new B
    B_tgt = torch.zeros(n_q_tgt, hd_tgt, rank, dtype=torch.float32, device=dev)

    for old_h, new_h in head_map.items():
        if old_h >= n_q_src or new_h >= n_q_tgt:
            continue
        # Map the old head's B to new head_dim via linear resize
        b_old_h = B_per_head_src[old_h]  # (hd_src, rank)
        if hd_src == hd_tgt:
            B_tgt[new_h] = b_old_h
        else:
            # Resize head_dim
            b_t = b_old_h.T.unsqueeze(0)  # (1, rank, hd_src)
            b_t = F.interpolate(b_t, size=hd_tgt, mode="linear", align_corners=False)
            B_tgt[new_h] = b_t.squeeze(0).T  # (hd_tgt, rank)

    # Handle unmatched new heads: zero-init
    B_tgt = B_tgt.reshape(n_q_tgt * hd_tgt, rank)
    return A_tgt.to(A_src.dtype).cpu(), B_tgt.to(B_src.dtype).cpu()


if __name__ == "__main__":
    main()
