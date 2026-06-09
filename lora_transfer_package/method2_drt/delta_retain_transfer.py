"""
delta_retain_transfer.py

Zero-gradient LoRA transfer: 1.7B -> 8B with retain null-space constraint.

Key idea (different from functional_projection_transfer.py):
  - Instead of matching absolute activations, match the DELTA (forgetting direction)
    that 1.7B's LoRA produces on forget samples.
  - Simultaneously enforce near-zero effect on retain samples (null-space constraint).

Closed-form solution via kernel trick (N x N inversion, efficient):
  ΔW = Y @ (ridge * I_N + K)^{-1} @ X
  where X = vstack(X_forget, sqrt(lambda_retain) * X_retain)   [N x d_in]
        Y = hstack(delta_h_target, zeros)                       [d_out x N]
        K = X @ X.T                                             [N x N]

Then decompose ΔW -> rank-r LoRA (A, B) via SVD.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES  = ["gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delta-retain constrained LoRA transfer")
    p.add_argument("--old_model",        type=str, required=True,  help="Path to 1.7B base model")
    p.add_argument("--new_model",        type=str, required=True,  help="Path to 8B base model")
    p.add_argument("--old_lora_path",    type=str, required=True,  help="Path to 1.7B unlearn LoRA")
    p.add_argument("--forget_file",      type=str, required=True,  help="Forget probe JSON (with prompt_used)")
    p.add_argument("--retain_file",      type=str, required=True,  help="Retain probe JSON (with prompt_used)")
    p.add_argument("--output_dir",       type=str, required=True)
    p.add_argument("--max_forget",       type=int, default=59)
    p.add_argument("--max_retain",       type=int, default=50)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--tokens_per_sample",type=int, default=4,
                   help="Number of token positions to collect per sample (last-N tokens)")
    p.add_argument("--modules",          type=str, default="attn", choices=["attn","mlp","all"])
    p.add_argument("--ridge",            type=float, default=1e-2,
                   help="Ridge regularisation (added to kernel diagonal)")
    p.add_argument("--lambda_retain",    type=float, default=10.0,
                   help="Weight on retain null-space constraint (higher = more retention)")
    p.add_argument("--target_mapping",   type=str, default="resize",
                   choices=["resize", "linear_map"],
                   help="How to map old delta target to new output space")
    p.add_argument("--target_map_ridge", type=float, default=1e-2,
                   help="Ridge for linear old->new output mapping when target_mapping=linear_map")
    p.add_argument("--target_map_use_retain", action="store_true", default=False,
                   help="When target_mapping=linear_map, fit mapping on forget+retain outputs")
    p.add_argument("--target_map_retain_weight", type=float, default=1.0,
                   help="Retain sample weight in linear_map fit (sqrt-weight in concatenation)")
    p.add_argument("--retain_nullspace_rank", type=int, default=0,
                   help="If >0, project solved delta_W to nullspace of top-k retain input subspace")
    p.add_argument("--retain_nullspace_center", action="store_true", default=True,
                   help="Center retain inputs before computing retain subspace for nullspace projection")
    p.add_argument("--lora_r",           type=int, default=32,
                   help="Rank of output LoRA (top-r SVD truncation)")
    p.add_argument("--alpha_scale",      type=float, default=1.0,
                   help="Multiplicative scale on lora_alpha relative to source")
    p.add_argument("--new_layer_start",  type=int, default=0,
                   help="Inclusive start index of new-model layers to transform")
    p.add_argument("--new_layer_end",    type=int, default=-1,
                   help="Inclusive end index of new-model layers to transform (-1 means last layer)")
    p.add_argument("--norm_calibrate",   action="store_true", default=True,
                   help="Calibrate ΔW norm to match source relative perturbation")
    p.add_argument("--dtype",            type=str, default="bfloat16")
    p.add_argument("--device",           type=str, default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)


def load_prompts(path: str, max_n: int) -> List[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        prompts = [x["prompt_used"] for x in data if "prompt_used" in x]
    else:
        prompts = []
    return prompts[:max_n]


def get_layer_map(old_layers: int, new_layers: int) -> Dict[int, int]:
    """Map every new layer to nearest old layer."""
    m = {}
    for ni in range(new_layers):
        oi = int(round(ni * old_layers / new_layers))
        m[ni] = max(0, min(old_layers - 1, oi))
    return m


def module_full_name(layer: int, proj: str) -> str:
    if proj in MLP_MODULES:
        return f"model.layers.{layer}.mlp.{proj}"
    return f"model.layers.{layer}.self_attn.{proj}"


def get_module(model: torch.nn.Module, full_name: str) -> torch.nn.Module:
    cur = model
    for part in full_name.split("."):
        cur = getattr(cur, part)
    return cur


def read_lora_pair(weights: Dict, layer: int, proj: str):
    if proj in MLP_MODULES:
        base = f"base_model.model.model.layers.{layer}.mlp.{proj}"
    else:
        base = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
    for suffix in [("lora_A.weight", "lora_B.weight"),
                   ("lora_A.default.weight", "lora_B.default.weight")]:
        ka, kb = f"{base}.{suffix[0]}", f"{base}.{suffix[1]}"
        if ka in weights and kb in weights:
            return weights[ka].float(), weights[kb].float()
    return None, None


def collect_inputs(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    target_keys: List[Tuple[int, str]],
    max_len: int,
    tokens_per_sample: int,
    device: str,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Run prompts through model, collect input activations at each (layer, proj)."""
    records: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in target_keys}
    hooks = []

    for layer, proj in target_keys:
        mod = get_module(model, module_full_name(layer, proj))

        def make_hook(key, n_tok):
            def _hook(_m, inputs, _out):
                # inputs[0]: [B, T, D], take last n_tok tokens, first sample
                x = inputs[0][0, -n_tok:, :].detach().float().cpu()  # [n_tok, D]
                records[key].append(x)
            return _hook

        hooks.append(mod.register_forward_hook(make_hook((layer, proj), tokens_per_sample)))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    for h in hooks:
        h.remove()

    packed = {}
    for k, v in records.items():
        if v:
            packed[k] = torch.cat(v, dim=0)  # [N_samples*n_tok, D_in]
    return packed


def collect_outputs(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    target_keys: List[Tuple[int, str]],
    max_len: int,
    tokens_per_sample: int,
    device: str,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Run prompts through model, collect output activations at each (layer, proj)."""
    records: Dict[Tuple[int, str], List[torch.Tensor]] = {k: [] for k in target_keys}
    hooks = []

    for layer, proj in target_keys:
        mod = get_module(model, module_full_name(layer, proj))

        def make_hook(key, n_tok):
            def _hook(_m, _inputs, out):
                y = out
                if isinstance(y, tuple):
                    y = y[0]
                # y: [B, T, D], take last n_tok tokens, first sample
                y = y[0, -n_tok:, :].detach().float().cpu()  # [n_tok, D]
                records[key].append(y)
            return _hook

        hooks.append(mod.register_forward_hook(make_hook((layer, proj), tokens_per_sample)))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    for h in hooks:
        h.remove()

    packed = {}
    for k, v in records.items():
        if v:
            packed[k] = torch.cat(v, dim=0)  # [N_samples*n_tok, D_out]
    return packed


def resize_rows(y: torch.Tensor, d_new: int) -> torch.Tensor:
    """y: [D_old, N] -> [D_new, N] via linear interpolation."""
    if y.shape[0] == d_new:
        return y
    yt = y.T.unsqueeze(1)                                    # [N, 1, D_old]
    yt_r = F.interpolate(yt, size=d_new, mode="linear", align_corners=False)
    return yt_r.squeeze(1).T                                  # [D_new, N]


def map_delta_old_to_new_linear(
    delta_h_old: torch.Tensor,   # [D_old, N]
    old_out: torch.Tensor,       # [N, D_old]
    new_out: torch.Tensor,       # [N, D_new]
    ridge: float,
    old_out_retain: torch.Tensor = None,  # [Nr, D_old]
    new_out_retain: torch.Tensor = None,  # [Nr, D_new]
    retain_weight: float = 1.0,
) -> torch.Tensor:
    """Map old delta target to new output space via ridge linear map fitted on base outputs.

    M = O_new O_old^T (O_old O_old^T + ridge I)^-1
    delta_h_new = M delta_h_old
    """
    O_old = old_out.T.float()   # [D_old, Nf]
    O_new = new_out.T.float()   # [D_new, Nf]

    # Optional retain-aware mapping fit (LwF-style: constrain old-task behavior).
    if old_out_retain is not None and new_out_retain is not None and old_out_retain.numel() > 0:
        wr = max(float(retain_weight), 0.0) ** 0.5
        O_old_r = old_out_retain.T.float() * wr
        O_new_r = new_out_retain.T.float() * wr
        O_old = torch.cat([O_old, O_old_r], dim=1)
        O_new = torch.cat([O_new, O_new_r], dim=1)

    D_old = O_old.shape[0]

    A = O_old @ O_old.T + ridge * torch.eye(D_old, dtype=torch.float32)
    try:
        A_inv_Oold = torch.linalg.solve(A, O_old)           # [D_old, N]
    except Exception:
        A_inv_Oold = torch.linalg.lstsq(A, O_old).solution
    M = O_new @ A_inv_Oold.T                                 # [D_new, D_old]
    return M @ delta_h_old                                   # [D_new, N]


def solve_delta_retain_lora(
    delta_h_target: torch.Tensor, # [D_out_new, N_f] – mapped target delta on forget inputs
    x_forget:    torch.Tensor,   # [N_f, D_in_new]  – 8B module inputs for forget
    x_retain:    torch.Tensor,   # [N_r, D_in_new]  – 8B module inputs for retain
    lora_r:      int,
    ridge:       float,
    lambda_r:    float,
    w_new_norm:  float,          # Frobenius norm of 8B base weight (for calibration)
    w_old_norm:  float,          # Frobenius norm of 1.7B base weight
    delta_old_norm: float,       # Frobenius norm of 1.7B LoRA delta matrix
    norm_calibrate: bool,
    retain_nullspace_rank: int = 0,
    retain_nullspace_center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form solution:
        X = vstack(X_f, sqrt(λ)*X_r)    [N x d_in]
        Y = hstack(Δh_target, 0)         [d_out_new x N]
        K = X @ X.T                      [N x N]
        ΔW = Y @ (ridge*I + K)^{-1} @ X [d_out_new x d_in]
    Then SVD truncate to rank r.
    """
    d_out_new = delta_h_target.shape[0]
    dh_target = delta_h_target.float()  # [D_out_new, N_f]

    N_f = x_forget.shape[0]
    N_r = x_retain.shape[0]

    # 2. Stack inputs: forget rows plain, retain rows scaled by sqrt(lambda)
    x_retain_w = x_retain * (lambda_r ** 0.5)
    X = torch.cat([x_forget, x_retain_w], dim=0).float()  # [N_f+N_r, d_in]
    N = X.shape[0]

    # 3. Build target Y: [d_out_new, N] = [dh_target | zeros]
    Y = torch.cat([dh_target,
                   torch.zeros(d_out_new, N_r, dtype=torch.float32)], dim=1)  # [D_out_new, N]

    # 4. Kernel matrix K = X @ X.T  [N x N]
    K = X @ X.T

    # 5. Solve (ridge*I + K)^{-1} via Cholesky (rI+K is symmetric PD -> faster than LU)
    rI = ridge * torch.eye(N, dtype=torch.float32)
    A_mat = rI + K
    # ΔW = Y @ (rI + K)^{-1} @ X
    # cholesky_solve(X, L) returns (LL^T)^{-1} X  =  (rI+K)^{-1} X
    try:
        L = torch.linalg.cholesky(A_mat)
        coeff = torch.cholesky_solve(X, L)              # [N, d_in]
    except Exception:
        # Fallback to LU solve if Cholesky fails (e.g. near-singular)
        try:
            coeff = torch.linalg.solve(A_mat, X)
        except Exception:
            coeff = torch.linalg.lstsq(A_mat, X).solution

    delta_W = Y @ coeff                                  # [d_out_new, d_in]

    # 6A. GEOMETRY CONSTRAINT: Ensure learned direction aligns with target
    # Compute principal effect directions and force alignment
    geom_flipped = False
    if x_forget.shape[0] > 0 and dh_target.shape[1] > 0:
        try:
            # Compute effect directions
            mu_learned = (delta_W @ x_forget.T).mean(dim=1)      # [D_out_new]
            mu_target = dh_target.mean(dim=1)                    # [D_out_new]
            
            # Normalize
            norm_learned = torch.norm(mu_learned) + 1e-8
            norm_target = torch.norm(mu_target) + 1e-8
            u_learned = mu_learned / norm_learned
            u_target = mu_target / norm_target
            
            # Compute cosine similarity
            cosine_sim = (u_learned @ u_target).item()
            
            # If anti-aligned (cosine < 0.3), flip entire delta_W
            # This forces the learned weight to have same direction as target
            if cosine_sim < 0.3:
                delta_W = -delta_W
                geom_flipped = True
                
        except Exception:
            pass  # If constraint computation fails, keep original delta_W

    # 6. Optional: project delta_W onto nullspace of retain input subspace.
    # This enforces delta_W @ V ~= 0 for top-k right-singular directions V of X_retain.
    if retain_nullspace_rank > 0 and x_retain.shape[0] > 1 and x_retain.shape[1] > 1:
        xr = x_retain.float()
        if retain_nullspace_center:
            xr = xr - xr.mean(dim=0, keepdim=True)
        try:
            _, _, Vt_r = torch.linalg.svd(xr, full_matrices=False)
            k = min(retain_nullspace_rank, Vt_r.shape[0], Vt_r.shape[1])
            if k > 0:
                V = Vt_r[:k, :].T  # [d_in, k]
                # Right-side projection: ΔW <- ΔW (I - V V^T)
                delta_W = delta_W - (delta_W @ V) @ V.T
        except Exception:
            # Keep original delta_W if SVD/projection fails.
            pass

    # 7. Optional: calibrate norm to match source relative perturbation
    if norm_calibrate and w_new_norm > 1e-8 and w_old_norm > 1e-8:
        rel_old = delta_old_norm / (w_old_norm + 1e-8)
        cur_norm = float(torch.norm(delta_W, p="fro"))
        if cur_norm > 1e-8:
            target_norm = rel_old * w_new_norm
            delta_W = delta_W * (target_norm / cur_norm)

    # 8. SVD truncation -> LoRA
    try:
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
    except Exception:
        # Fallback: just skip this module
        raise RuntimeError("SVD failed")

    rank = min(lora_r, len(S))
    B = (U[:, :rank] * S[:rank].unsqueeze(0))  # [D_out_new, r]
    A = Vt[:rank, :]                            # [r, D_in_new]

    return A, B


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = to_dtype(args.dtype)

    forget_prompts = load_prompts(args.forget_file, args.max_forget)
    retain_prompts = load_prompts(args.retain_file, args.max_retain)
    if not forget_prompts:
        raise ValueError("No forget prompts found.")
    if not retain_prompts:
        raise ValueError("No retain prompts found.")
    print(f"Forget prompts: {len(forget_prompts)}, Retain prompts: {len(retain_prompts)}")

    # Load LoRA
    old_lora_w = load_file(os.path.join(args.old_lora_path, "adapter_model.safetensors"))
    with open(os.path.join(args.old_lora_path, "adapter_config.json"), "r", encoding="utf-8-sig") as f:
        old_cfg = json.load(f)
    lora_alpha_src = old_cfg.get("lora_alpha", 32)
    lora_r_src     = old_cfg.get("r", 32)
    lora_scale     = lora_alpha_src / lora_r_src

    modules = {"attn": ATTN_MODULES, "mlp": MLP_MODULES, "all": ATTN_MODULES + MLP_MODULES}[args.modules]
    lora_r = args.lora_r if args.lora_r > 0 else lora_r_src

    # Determine layer counts early
    old_cfg_m = AutoConfig.from_pretrained(args.old_model, trust_remote_code=True)
    new_cfg_m = AutoConfig.from_pretrained(args.new_model, trust_remote_code=True)
    old_nl = old_cfg_m.num_hidden_layers
    new_nl = new_cfg_m.num_hidden_layers
    layer_map = get_layer_map(old_nl, new_nl)
    layer_start = max(0, int(args.new_layer_start))
    layer_end = int(args.new_layer_end)
    if layer_end < 0:
        layer_end = new_nl - 1
    layer_end = min(new_nl - 1, layer_end)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: start={layer_start}, end={layer_end}, total={new_nl}")

    target_new_layers = [l for l in range(new_nl) if layer_start <= l <= layer_end]
    target_old_layers = sorted({layer_map[l] for l in target_new_layers})

    print(f"Layer map: {old_nl} (old) -> {new_nl} (new), {len(target_new_layers)} target layers")
    print(f"Target new layers: [{layer_start}, {layer_end}]")

    # -----------------------------------------------------------------------
    # Phase 1: Collect 1.7B+LoRA forget inputs & compute per-module delta
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading 1.7B model + LoRA to compute forget deltas...")
    tok_old = AutoTokenizer.from_pretrained(args.old_model, trust_remote_code=True)
    old_model = AutoModelForCausalLM.from_pretrained(
        args.old_model, torch_dtype=dtype, device_map=args.device, trust_remote_code=True
    )

    old_keys = [(l, p) for l in target_old_layers for p in modules]

    old_forget_inputs = collect_inputs(
        old_model, tok_old, forget_prompts, old_keys,
        args.max_input_length, args.tokens_per_sample, args.device
    )
    print(f"  Collected inputs for {len(old_forget_inputs)} (layer,proj) pairs from 1.7B")

    old_forget_outputs: Dict[Tuple[int, str], torch.Tensor] = {}
    old_retain_outputs: Dict[Tuple[int, str], torch.Tensor] = {}
    if args.target_mapping == "linear_map":
        old_forget_outputs = collect_outputs(
            old_model, tok_old, forget_prompts, old_keys,
            args.max_input_length, args.tokens_per_sample, args.device
        )
        print(f"  Collected outputs for {len(old_forget_outputs)} (layer,proj) pairs from 1.7B")
        if args.target_map_use_retain:
            old_retain_outputs = collect_outputs(
                old_model, tok_old, retain_prompts, old_keys,
                args.max_input_length, args.tokens_per_sample, args.device
            )
            print(f"  Collected retain outputs for {len(old_retain_outputs)} (layer,proj) pairs from 1.7B")

    # Compute delta = lora_scale * B @ A @ X  and base weight norms
    old_deltas:    Dict[Tuple[int, str], torch.Tensor] = {}
    old_w_norms:   Dict[Tuple[int, str], float]        = {}
    old_delta_norms: Dict[Tuple[int, str], float]      = {}

    sd_old = old_model.state_dict()
    for ol in target_old_layers:
        for proj in modules:
            A_s, B_s = read_lora_pair(old_lora_w, ol, proj)
            if A_s is None:
                continue
            key = (ol, proj)
            if key not in old_forget_inputs:
                continue
            x_old = old_forget_inputs[key]  # [N, D_in_old]
            # delta output: [D_out_old, N]
            delta_out = lora_scale * (B_s @ (A_s @ x_old.T))
            old_deltas[key] = delta_out

            # Weight norms
            if proj in MLP_MODULES:
                wk = f"model.layers.{ol}.mlp.{proj}.weight"
            else:
                wk = f"model.layers.{ol}.self_attn.{proj}.weight"
            if wk in sd_old:
                old_w_norms[key]     = float(torch.norm(sd_old[wk].float(), p="fro"))
            old_delta_norms[key] = float(torch.norm(lora_scale * (B_s @ A_s), p="fro"))

    print(f"  Deltas computed for {len(old_deltas)} modules")

    del old_model, sd_old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2: Collect 8B forget + retain inputs
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading 8B model...")
    tok_new = AutoTokenizer.from_pretrained(args.new_model, trust_remote_code=True)
    new_model = AutoModelForCausalLM.from_pretrained(
        args.new_model, torch_dtype=dtype, device_map=args.device, trust_remote_code=True
    )

    all_new_keys = [(l, p) for l in range(new_nl) for p in modules]

    print("[3/4] Collecting 8B forget activations...")
    new_forget_inputs = collect_inputs(
        new_model, tok_new, forget_prompts, all_new_keys,
        args.max_input_length, args.tokens_per_sample, args.device
    )
    print(f"  Forget: {len(new_forget_inputs)} module inputs collected")

    new_forget_outputs: Dict[Tuple[int, str], torch.Tensor] = {}
    new_retain_outputs: Dict[Tuple[int, str], torch.Tensor] = {}
    if args.target_mapping == "linear_map":
        new_forget_outputs = collect_outputs(
            new_model, tok_new, forget_prompts, all_new_keys,
            args.max_input_length, args.tokens_per_sample, args.device
        )
        print(f"  Forget: {len(new_forget_outputs)} module outputs collected")

    print("[3/4] Collecting 8B retain activations...")
    new_retain_inputs = collect_inputs(
        new_model, tok_new, retain_prompts, all_new_keys,
        args.max_input_length, args.tokens_per_sample, args.device
    )
    print(f"  Retain: {len(new_retain_inputs)} module inputs collected")
    if args.target_mapping == "linear_map" and args.target_map_use_retain:
        new_retain_outputs = collect_outputs(
            new_model, tok_new, retain_prompts, all_new_keys,
            args.max_input_length, args.tokens_per_sample, args.device
        )
        print(f"  Retain: {len(new_retain_outputs)} module outputs collected")

    # Grab 8B weight norms and output dims
    sd_new = new_model.state_dict()
    new_w_norms: Dict[Tuple[int, str], float] = {}
    new_out_dim: Dict[Tuple[int, str], int]   = {}
    for nl in range(new_nl):
        for proj in modules:
            if proj in MLP_MODULES:
                wk = f"model.layers.{nl}.mlp.{proj}.weight"
            else:
                wk = f"model.layers.{nl}.self_attn.{proj}.weight"
            if wk in sd_new:
                w = sd_new[wk].float()
                new_w_norms[(nl, proj)] = float(torch.norm(w, p="fro"))
                new_out_dim[(nl, proj)] = w.shape[0]

    del new_model, sd_new
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 3: Solve for LoRA per layer
    # -----------------------------------------------------------------------
    print("\n[4/4] Solving delta-retain constrained LoRA (closed-form)...")
    out_state: Dict[str, torch.Tensor] = {}
    solved = 0
    skipped = 0

    for new_layer in target_new_layers:
        old_layer = layer_map[new_layer]
        for proj in modules:
            key_o = (old_layer, proj)
            key_n = (new_layer, proj)

            if key_o not in old_deltas:
                skipped += 1
                continue
            if key_n not in new_forget_inputs or key_n not in new_retain_inputs:
                skipped += 1
                continue
            if key_n not in new_out_dim:
                skipped += 1
                continue

            delta_h_old = old_deltas[key_o]          # [D_out_old, N_f]
            x_forget    = new_forget_inputs[key_n]    # [N_f, D_in_new]
            x_retain    = new_retain_inputs[key_n]    # [N_r, D_in_new]
            d_out_new   = new_out_dim[key_n]

            if args.target_mapping == "linear_map":
                if key_o not in old_forget_outputs or key_n not in new_forget_outputs:
                    skipped += 1
                    continue
                old_r = None
                new_r = None
                if args.target_map_use_retain:
                    if key_o not in old_retain_outputs or key_n not in new_retain_outputs:
                        skipped += 1
                        continue
                    old_r = old_retain_outputs[key_o]
                    new_r = new_retain_outputs[key_n]
                delta_h_target = map_delta_old_to_new_linear(
                    delta_h_old,
                    old_forget_outputs[key_o],
                    new_forget_outputs[key_n],
                    args.target_map_ridge,
                    old_out_retain=old_r,
                    new_out_retain=new_r,
                    retain_weight=args.target_map_retain_weight,
                )
            else:
                delta_h_target = resize_rows(delta_h_old, d_out_new)

            try:
                A_t, B_t = solve_delta_retain_lora(
                    delta_h_target, x_forget, x_retain,
                    lora_r, args.ridge, args.lambda_retain,
                    w_new_norm     = new_w_norms.get(key_n, 1.0),
                    w_old_norm     = old_w_norms.get(key_o, 1.0),
                    delta_old_norm = old_delta_norms.get(key_o, 1.0),
                    norm_calibrate = args.norm_calibrate,
                    retain_nullspace_rank = args.retain_nullspace_rank,
                    retain_nullspace_center = args.retain_nullspace_center,
                )
            except Exception as e:
                print(f"  Skipped layer={new_layer} {proj}: {e}")
                skipped += 1
                continue

            if proj in MLP_MODULES:
                base = f"base_model.model.model.layers.{new_layer}.mlp.{proj}"
            else:
                base = f"base_model.model.model.layers.{new_layer}.self_attn.{proj}"

            out_state[f"{base}.lora_A.default.weight"] = A_t.to(dtype).contiguous()
            out_state[f"{base}.lora_B.default.weight"] = B_t.to(dtype).contiguous()
            solved += 1

    print(f"  Solved: {solved}, Skipped: {skipped}")

    # -----------------------------------------------------------------------
    # Save adapter
    # -----------------------------------------------------------------------
    save_file(out_state, os.path.join(args.output_dir, "adapter_model.safetensors"))

    lora_alpha_out = int(round(lora_alpha_src * args.alpha_scale))
    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": args.new_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": target_new_layers,
        "lora_alpha": lora_alpha_out,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }
    with open(os.path.join(args.output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved adapter to {args.output_dir}")
    print(f"  lora_alpha={lora_alpha_out}, lora_r={lora_r}, lambda_retain={args.lambda_retain}")


if __name__ == "__main__":
    main()
