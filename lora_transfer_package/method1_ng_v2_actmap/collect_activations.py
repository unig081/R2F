#!/usr/bin/env python3
"""
collect_activations.py
======================
Collect per-layer post-block hidden states from two LLMs on generic (non-task-specific)
text, then compute and save per-layer activation-space alignment matrices R_l.

  R_l : (old_hidden, new_hidden)  s.t.  H_old[i] @ R_l  ≈  H_new[layer_mapping[i]]

These replace the embedding-derived W_x in lora_adaption.py --act_align_path.

Usage
-----
python collect_activations.py \
    --old_model ./modelzoo/qwen3_1_7B/ \
    --new_model ./modelzoo/qwen3_8B/ \
    --data ./ft-training_set/sampled_100_math_10k.json \
    --n_samples 64 \
    --max_length 128 \
    --output_dir ./tmp/act_align_qwen3_math64/

Output
------
  output_dir/R_l_{i}.pt     (old_hidden, new_hidden)  per source layer i
  output_dir/meta.json      metadata / reconstruction errors
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_hidden_states(model_path: str, texts: list[str],
                          max_length: int = 128, batch_size: int = 2,
                          device: str = "cuda") -> tuple[list[torch.Tensor], int, int]:
    """
    Load model, run texts through it, and capture the post-block hidden state
    at every transformer layer.

    Returns
    -------
    layer_acts : list[Tensor(N_tokens, hidden)]  — one entry per layer, CPU float16
    n_layers   : int
    hidden_size: int
    """
    print(f"  Loading {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  {n_layers} layers, hidden={hidden_size}")

    # Storage: list of lists (per layer) of (batch, hidden) tensors
    layer_acts: list[list[tuple]] = [[] for _ in range(n_layers)]

    hooks = []
    # Store per-batch (attention_mask, hidden_states) tuples per layer
    layer_acts: list[list[tuple]] = [[] for _ in range(n_layers)]
    current_mask: list = [None]  # mutable container to share mask with hook

    def make_hook(idx: int):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = current_mask[0]  # (batch, seq)
            if mask is not None:
                # Mean-pool over valid (non-padding) tokens
                mask_f = mask.unsqueeze(-1).to(h.dtype)  # (batch, seq, 1)
                h_masked = h * mask_f
                pooled = h_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (batch, hidden)
            else:
                pooled = h.mean(dim=1)  # (batch, hidden)
            layer_acts[idx].append(pooled.detach().cpu().to(torch.float16))
        return hook

    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            current_mask[0] = enc.attention_mask
            with torch.no_grad():
                model(**enc)
            current_mask[0] = None
            done = min(start + batch_size, len(texts))
            print(f"  {done}/{len(texts)} samples processed …", end="\r")
        print()
    finally:
        for h in hooks:
            h.remove()

    del model
    torch.cuda.empty_cache()

    # Concatenate mean-pooled vectors: each entry is (batch, hidden) → cat → (N_samples, hidden)
    result: list[torch.Tensor] = []
    for acts in layer_acts:
        cat = torch.cat(acts, dim=0)   # (N_samples, hidden)
        result.append(cat)

    return result, n_layers, hidden_size


# ---------------------------------------------------------------------------
# Alignment matrix computation
# ---------------------------------------------------------------------------

def max_layer_similarity_mapping(A: np.ndarray) -> tuple[list[int], float]:
    """
    Monotonic DP layer mapping that maximizes total similarity.

    Supports A shaped either (n_old, n_new) or (n_new, n_old).
    Returns mapping old_layer -> new_layer.
    """
    n, m = A.shape
    transposed = False
    if n > m:
        A = A.T
        n, m = A.shape
        transposed = True

    dp = np.full((n, m), -float("inf"), dtype=np.float64)
    path = np.zeros((n, m), dtype=np.int32)

    for j in range(m):
        if j <= m - n:
            dp[0, j] = A[0, j]

    for i in range(1, n):
        for j in range(i, m):
            if j - i <= m - n:
                best_v = -float("inf")
                best_k = i - 1
                for k in range(i - 1, j):
                    v = dp[i - 1, k] + A[i, j]
                    if v > best_v:
                        best_v = v
                        best_k = k
                dp[i, j] = best_v
                path[i, j] = best_k

    best_sum = -float("inf")
    last_j = n - 1
    for j in range(n - 1, m):
        if dp[n - 1, j] > best_sum:
            best_sum = dp[n - 1, j]
            last_j = j

    mapping = [0] * n
    for i in range(n - 1, -1, -1):
        mapping[i] = int(last_j)
        last_j = int(path[i, last_j])

    if transposed:
        # If input was transposed, n still corresponds to old-layer count
        # because we always return old->new semantics.
        return mapping, float(best_sum)
    return mapping, float(best_sum)


def max_layer_similarity_mapping_hungarian(A: np.ndarray) -> tuple[list[int], float]:
    """
    One-to-one layer mapping via Hungarian algorithm.

    Supports A shaped either (n_old, n_new) or (n_new, n_old).
    Returns mapping old_layer -> new_layer.
    """
    n0, n1 = A.shape
    if n0 > n1:
        A = A.T
        n0, n1 = A.shape

    row_ind, col_ind = linear_sum_assignment(-A)
    mapping = [0] * n0
    for r, c in zip(row_ind, col_ind):
        mapping[int(r)] = int(c)
    score = float(A[row_ind, col_ind].sum())
    return mapping, score

def compute_alignment(H_old: torch.Tensor, H_new: torch.Tensor,
                      reg: float = 1e-3) -> tuple[torch.Tensor, float]:
    """
    Solve  H_old @ R  ≈  H_new  via ridge regression (normal equations).

    Ridge: R = (H^T H + λI)^{-1} H^T H_new
    where λ = reg * max_diagonal(H^T H).

    This is numerically stable even when N (samples) << d (hidden_dim),
    unlike SVD of H which can be degenerate.

    H_old : (N, old_hidden)  float16 or float32
    H_new : (N, new_hidden)  float16 or float32
    reg   : ridge regularisation relative to max eigenvalue (default: 1e-3)

    Returns R : (old_hidden, new_hidden),  err : float (relative Frobenius error)
    """
    Ho = H_old.float()   # (N, d_old)
    Hn = H_new.float()   # (N, d_new)

    # Replace any NaN/Inf that might appear in deep layers
    Ho = torch.nan_to_num(Ho, nan=0.0, posinf=1e4, neginf=-1e4)
    Hn = torch.nan_to_num(Hn, nan=0.0, posinf=1e4, neginf=-1e4)

    # lstsq is more numerically stable than normal-equations solve,
    # especially when N (64) << d (2048/4096).
    # driver='gelsd' uses divide-and-conquer SVD on the (N, d_old) matrix.
    result = torch.linalg.lstsq(Ho, Hn, rcond=reg, driver='gelsd')
    R = result.solution  # (d_old, d_new)

    recon_err = ((Ho @ R - Hn).norm() / (Hn.norm() + 1e-8)).item()
    return R, recon_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect activation-based alignment matrices")
    parser.add_argument("--old_model",  required=True,  help="Path to old (source) model")
    parser.add_argument("--new_model",  required=True,  help="Path to new (target) model")
    parser.add_argument("--data",       default="./ft-training_set/sampled_100_math_10k.json",
                        help="JSON file with generic (non-task) texts")
    parser.add_argument("--n_samples",  type=int, default=64,
                        help="Number of text samples to use (default: 64)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length per sample (default: 128)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for inference (default: 2)")
    parser.add_argument("--reg",        type=float, default=1e-3,
                        help="Regularisation coefficient for lstsq (default: 1e-3)")
    parser.add_argument("--layer_mapping_mode", type=str, default="fixed",
                        choices=["fixed", "cka_monotonic", "cka_hungarian"],
                        help="Layer mapping mode for R_l collection: fixed ratio or CKA-based mapping")
    parser.add_argument("--cka_path", type=str, default=None,
                        help="Path to CKA matrix .pt used when layer_mapping_mode is CKA-based")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save R_l_*.pt files and meta.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load generic text samples
    # -----------------------------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    def extract_text(item):
        for key in ("instruction", "question", "text", "input", "prompt"):
            if key in item and item[key]:
                return str(item[key]).strip()
        return str(list(item.values())[0]).strip()

    texts = [extract_text(item) for item in data[: args.n_samples]]
    n = len(texts)
    print(f"Using {n} samples from {args.data}")

    # -----------------------------------------------------------------------
    # Collect activations from old model
    # -----------------------------------------------------------------------
    print("\n=== OLD model ===")
    old_acts, old_n_layers, old_hidden = collect_hidden_states(
        args.old_model, texts, args.max_length, args.batch_size,
    )

    # -----------------------------------------------------------------------
    # Collect activations from new model
    # -----------------------------------------------------------------------
    print("\n=== NEW model ===")
    new_acts, new_n_layers, new_hidden = collect_hidden_states(
        args.new_model, texts, args.max_length, args.batch_size,
    )

    # -----------------------------------------------------------------------
    # Layer mapping  old[i] → new[layer_mapping[i]]
    # -----------------------------------------------------------------------
    cka_path_used = None
    cka_score = None
    if args.layer_mapping_mode == "fixed":
        layer_mapping = [int(i * new_n_layers / old_n_layers) for i in range(old_n_layers)]
    else:
        if args.cka_path is not None:
            cka_path_used = args.cka_path
        else:
            old_name = os.path.basename(os.path.normpath(args.old_model))
            new_name = os.path.basename(os.path.normpath(args.new_model))
            cka_path_used = os.path.join("./tmp", f"{old_name}_{new_name}_CKA.pt")

        if not os.path.exists(cka_path_used):
            raise FileNotFoundError(
                f"CKA matrix not found: {cka_path_used}. "
                f"Please pass --cka_path or precompute CKA first."
            )

        A = torch.load(cka_path_used, map_location="cpu").numpy()
        if args.layer_mapping_mode == "cka_monotonic":
            layer_mapping, cka_score = max_layer_similarity_mapping(A)
        else:
            layer_mapping, cka_score = max_layer_similarity_mapping_hungarian(A)

        if len(layer_mapping) != old_n_layers:
            raise ValueError(
                f"CKA mapping length mismatch: got {len(layer_mapping)}, expected {old_n_layers}"
            )
        if max(layer_mapping) >= new_n_layers:
            raise ValueError(
                f"CKA mapping index out of range: max={max(layer_mapping)}, new_n_layers={new_n_layers}"
            )

    print(f"\nLayer mapping mode: {args.layer_mapping_mode}")
    if cka_path_used is not None:
        print(f"CKA matrix path: {cka_path_used}")
        print(f"CKA mapping score: {cka_score:.6f}")
    print(f"Layer mapping ({old_n_layers} → {new_n_layers}): {layer_mapping}")

    # -----------------------------------------------------------------------
    # Compute R_l per mapped layer pair
    # -----------------------------------------------------------------------
    print("\n=== Computing alignment matrices ===")
    errors = []
    for i in range(old_n_layers):
        j = layer_mapping[i]
        Ho = old_acts[i]   # (N_tok, old_hidden)
        Hn = new_acts[j]   # (N_tok, new_hidden)

        R, err = compute_alignment(Ho, Hn, reg=args.reg)
        errors.append(err)
        print(f"  old[{i:2d}] → new[{j:2d}]  err={err:.4f}  R{tuple(R.shape)}")

        torch.save(R, os.path.join(args.output_dir, f"R_l_{i}.pt"))

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    meta = {
        "old_model":     args.old_model,
        "new_model":     args.new_model,
        "old_n_layers":  old_n_layers,
        "new_n_layers":  new_n_layers,
        "old_hidden":    old_hidden,
        "new_hidden":    new_hidden,
        "layer_mapping": layer_mapping,
        "n_samples":     n,
        "max_length":    args.max_length,
        "reg":           args.reg,
        "layer_mapping_mode": args.layer_mapping_mode,
        "cka_path":      cka_path_used,
        "cka_score":     cka_score,
        "errors":        [round(e, 4) for e in errors],
        "mean_error":    round(float(np.mean(errors)), 4),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved {old_n_layers} alignment matrices to '{args.output_dir}'")
    print(f"  Mean reconstruction error: {meta['mean_error']:.4f}")
    print(f"  Max  reconstruction error: {max(errors):.4f}")


if __name__ == "__main__":
    main()
