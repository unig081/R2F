#!/usr/bin/env python3
"""
compute_cka.py
==============
Compute layer-wise linear CKA similarity matrix between two LLMs.

For each pair (old_layer_i, new_layer_j) we compute linear CKA between
the mean-pooled hidden states collected by running texts through both models.

Result shape: (n_old_layers, n_new_layers), values in [0, 1].

Saved to: ./tmp/{old_name}_{new_name}_CKA.pt  (overwrites existing)

Usage
-----
python compute_cka.py \
    --old_model ./modelzoo/qwen3_1_7B/ \
    --new_model ./modelzoo/qwen3_8B/ \
    --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
    --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
    --n_samples 100 \
    --max_length 128 \
    --batch_size 4
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between X (N, p) and Y (N, q).

    Linear CKA = HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T,XX^T) * HSIC(YY^T,YY^T))

    Using the unbiased estimator from Kornblith et al. 2019 (simplified form):
        HSIC_1(K, L) = 1/(n-1)^2 * ||Y^T X||_F^2  (for centered X, Y)

    i.e. CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c, Y_c are column-centered (subtract mean over N).
    """
    X = X.float()
    Y = Y.float()
    # Center columns (subtract mean over samples)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # ||Y^T X||_F^2 = tr(X^T Y Y^T X) = ||Y^T X||_F^2
    XtY = X.T @ Y          # (p, q)
    num = (XtY ** 2).sum()

    XtX = X.T @ X          # (p, p)
    YtY = Y.T @ Y          # (q, q)
    denom = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum())

    if denom < 1e-10 or torch.isnan(denom) or torch.isnan(num):
        return 0.0
    return float(torch.clamp(num / denom, 0.0, 1.0).item())


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_hidden_states(model_path: str, texts: list,
                          max_length: int = 128, batch_size: int = 4,
                          device: str = "cuda"):
    """
    Load model, run texts through it, return mean-pooled hidden states per layer.
    Returns: list of Tensor(N_samples, hidden) for each layer, n_layers, hidden_size
    """
    print(f"  Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=False,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  {n_layers} layers, hidden={hidden_size}")

    layer_acts = [[] for _ in range(n_layers)]
    current_mask = [None]

    def make_hook(idx):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = current_mask[0]
            if mask is not None:
                mask_f = mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                pooled = h.mean(dim=1)
            layer_acts[idx].append(pooled.detach().cpu().float())
        return hook

    hooks = []
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
            print(f"  {min(start + batch_size, len(texts))}/{len(texts)} ...", end="\r")
        print()
    finally:
        for h in hooks:
            h.remove()

    del model
    torch.cuda.empty_cache()

    result = []
    for acts in layer_acts:
        cat = torch.cat(acts, dim=0).float()  # (N_samples, hidden)
        # Replace NaN/Inf that may arise from float16 overflow in deep layers
        cat = torch.nan_to_num(cat, nan=0.0, posinf=1e4, neginf=-1e4)
        result.append(cat)
    return result, n_layers, hidden_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model", required=True)
    parser.add_argument("--new_model", required=True)
    parser.add_argument("--forget_file", required=True)
    parser.add_argument("--retain_file", required=True)
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Total samples to use from forget+retain (50/50 split by default)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (default: ./tmp/{old}_{new}_CKA.pt)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load texts from forget + retain datasets
    # -----------------------------------------------------------------------
    def load_texts(path, n):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            # Try JSON array first, then JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                items = data[:n]
            else:
                # JSONL format
                items = []
                for line in f:
                    line = line.strip()
                    if line and len(items) < n:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        for item in items:
            for key in ("question", "instruction", "text", "input", "prompt"):
                if key in item and str(item[key]).strip():
                    texts.append(str(item[key]).strip())
                    break
            else:
                texts.append(str(list(item.values())[0]).strip())
        return texts

    half = args.n_samples // 2
    forget_texts = load_texts(args.forget_file, half)
    retain_texts = load_texts(args.retain_file, half)
    texts = forget_texts + retain_texts
    print(f"Total texts: {len(texts)} ({len(forget_texts)} forget + {len(retain_texts)} retain)")

    # -----------------------------------------------------------------------
    # Collect activations
    # -----------------------------------------------------------------------
    print("\n=== OLD model ===")
    old_acts, old_n_layers, old_hidden = collect_hidden_states(
        args.old_model, texts, args.max_length, args.batch_size)

    print("\n=== NEW model ===")
    new_acts, new_n_layers, new_hidden = collect_hidden_states(
        args.new_model, texts, args.max_length, args.batch_size)

    print(f"\nOld: {old_n_layers} layers x {old_hidden} hidden")
    print(f"New: {new_n_layers} layers x {new_hidden} hidden")

    # -----------------------------------------------------------------------
    # Compute full CKA matrix  (old_n_layers, new_n_layers)
    # -----------------------------------------------------------------------
    print(f"\n=== Computing CKA matrix ({old_n_layers} x {new_n_layers}) ===")
    cka_matrix = torch.zeros(old_n_layers, new_n_layers)
    for i in range(old_n_layers):
        for j in range(new_n_layers):
            cka_val = linear_cka(old_acts[i], new_acts[j])
            cka_matrix[i, j] = cka_val
        print(f"  old[{i:2d}] done, max new layer = {cka_matrix[i].argmax().item():2d} "
              f"(cka={cka_matrix[i].max().item():.4f})")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    old_name = os.path.normpath(args.old_model.rstrip('/\\')).split(os.sep)[-1]
    new_name = os.path.normpath(args.new_model.rstrip('/\\')).split(os.sep)[-1]
    out_path = args.output or f"./tmp/{old_name}_{new_name}_CKA.pt"
    torch.save(cka_matrix, out_path)
    print(f"\nSaved CKA matrix {tuple(cka_matrix.shape)} to {out_path}")

    # Print top mapping (argmax per old layer)
    argmax_map = cka_matrix.argmax(dim=1).tolist()
    print(f"Argmax layer mapping (CKA-best): {argmax_map}")

    # Print matrix stats
    print(f"CKA range: [{cka_matrix.min():.4f}, {cka_matrix.max():.4f}]")
    print(f"Diagonal (proportional): {[f'{cka_matrix[i, int(i*new_n_layers/old_n_layers)]:.3f}' for i in range(old_n_layers)]}")


if __name__ == "__main__":
    main()
