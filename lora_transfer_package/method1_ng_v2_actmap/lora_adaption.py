
import argparse
import json
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from peft import load_peft_weights, LoraConfig
from safetensors.torch import save_file as safe_save_file
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed, 
) # noqa: F402
from scipy.optimize import linear_sum_assignment


def _get_parallel_and_null_bases(W0, energy_ratio=0.90):
    """
    Return parallel/null bases for row/column spaces using SVD energy threshold.
    """
    U0, S0, V0_t = torch.linalg.svd(W0, full_matrices=False)
    total_energy = torch.sum(S0**2)
    if total_energy <= 0:
        r0 = 0
    else:
        energy = torch.cumsum(S0**2, dim=0) / total_energy
        r0 = torch.searchsorted(energy, energy_ratio).item() + 1

    U_par = U0[:, :r0]
    U_null = U0[:, r0:]
    V_par = V0_t[:r0, :].T
    V_null = V0_t[r0:, :].T
    return U_par, U_null, V_par, V_null


def _project_with_bases(Delta_W, U_basis, V_basis):
    """
    Project Delta_W onto the bilinear subspace spanned by U_basis and V_basis.
    """
    if U_basis.numel() == 0 or V_basis.numel() == 0:
        return torch.zeros_like(Delta_W)
    return U_basis @ (U_basis.T @ Delta_W @ V_basis) @ V_basis.T


def prolora_filter(delta, W, energy_ratio=0.90, null_scale=1.0):
    """
    Filter delta w.r.t. W's subspace with minimal information loss:
      - Parallel component (bilinear-pp projection onto W's row+col space): kept fully
      - Residual (null + cross terms pn/np): attenuated by null_scale

    Key property: delta_par + null_scale*(delta - delta_par)
      - null_scale=1.0 ?????????no filtering (returns delta unchanged)
      - null_scale=0.0 ?????????only the parallel component
    Unlike the old two-function API, this preserves pn/np cross-terms (only attenuated),
    so par + null_scale*residual always reconstructs to delta when null_scale=1.0.
    """
    U_par, _, V_par, _ = _get_parallel_and_null_bases(W, energy_ratio)
    delta_par = _project_with_bases(delta, U_par, V_par)
    return delta_par + null_scale * (delta - delta_par)


def procrustes_L(W_old_head, W_new_head, W_x):
    """
    Solve for the optimal orthogonal left-transform L* via Procrustes:
        L* = argmin_Q ||W_new_head - Q @ W_old_head @ W_x||_F  s.t. Q^T Q = I

    Solution: L* = U @ Vt  where  U, _, Vt = svd(W_new_head @ (W_old_head @ W_x)^T)

    W_old_head: (head_size, old_hidden)
    W_new_head: (head_size, new_hidden)
    W_x:        (old_hidden, new_hidden)
    returns L*: (head_size, head_size)  ?????????orthogonal matrix
    """
    M = W_new_head @ (W_old_head @ W_x).T   # (head_size, head_size)
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt  # nearest orthogonal matrix


def apply_svd_transfer(old_delta, W_old, W_new, W_x):
    """
    SVD Procrustes Transfer (SVDT): transfer LoRA delta by aligning singular
    vectors directly in the new hidden space via Procrustes orthogonalization,
    then rescaling singular values by the norm ratio of the base weights.

    For Q/K/V heads: old_delta shape = (head_size, old_hidden)
    - Right singular vectors Vt_old (r, old_hidden) ?????????project to new hidden via W_x
    - Left singular vectors U_old  (head_size, r) ?????????head_size same in both models (64),
      so they are kept as-is (no alignment needed for head-dim space)
    - Singular values scaled by ||W_new||_F / ||W_old||_F

    Steps:
    1. SVD: old_delta = U_old @ diag(S_old) @ Vt_old
    2. Project right SVs: Vt_proj_raw = Vt_old @ W_x  (r, new_hidden)
    3. Procrustes: Vt_proj = Q s.t. Q ?????????Vt_proj_raw with orthonormal rows
                  ?????????via compact SVD: U_v @ Vt_v = svd(Vt_proj_raw)
                  ?????????Vt_proj = U_v @ Vt_v
    4. Scale SVs:  S_cal = S_old * (||W_new||_F / ||W_old||_F)
    5. Return:     U_old @ diag(S_cal) @ Vt_proj
    """
    U_old, S_old, Vt_old = torch.linalg.svd(old_delta, full_matrices=False)
    # Project right singular vectors into new hidden space
    Vt_proj_raw = Vt_old @ W_x          # (r, new_hidden)
    # Procrustes orthogonalization: find nearest orthogonal matrix
    U_v, _, Vt_v = torch.linalg.svd(Vt_proj_raw, full_matrices=False)
    Vt_proj = U_v @ Vt_v                # (r, new_hidden) with orthonormal rows
    # Scale singular values by weight norm ratio
    norm_ratio = torch.norm(W_new, 'fro') / (torch.norm(W_old, 'fro') + 1e-8)
    S_cal = S_old * norm_ratio
    return U_old @ torch.diag(S_cal) @ Vt_proj  # (head_size, new_hidden)


def apply_xform_with_prolora(old_delta, W_old, W_new, L, R,
                              use_prolora, mode, energy, null_scale,
                              scale_calibrate=False, spectral_cal=False,
                              polar_L=False, no_left_transform=False,
                              spectral_blend=1.0):
    """
    Apply bilinear map new_delta = L @ old_delta @ R with optional ProLoRA filtering
    and optional calibration (scale_calibrate or spectral_cal).

    scale_calibrate: global Frobenius-norm rescaling (||delta||/||W|| preserved)
    spectral_cal   : per-SV spectrum matching (strictly better than scale_calibrate
                     when xTransform distorts spectrum unevenly)
    polar_L        : replace L with its nearest orthogonal matrix (polar decomposition
                     L = U @ S @ Vt ?????????Q = U @ Vt). Removes head-space scale artifacts;
                     spectral_cal then handles the remaining scale correction.
    no_left_transform: use identity for L (pure W_x right-side transform only)
    spectral_blend : blend ratio [0,1] for spectral calibration SVs (1.0 = pure old SVs)
    """
    orig_old_delta = old_delta  # save before any ProLoRA modification
    if use_prolora and mode in ('source', 'both'):
        old_delta = prolora_filter(old_delta, W_old, energy, null_scale)
    if no_left_transform:
        # Use identity: new_delta = old_delta @ R
        new_delta = old_delta @ R
    else:
        if polar_L:
            U_L, _, Vt_L = torch.linalg.svd(L, full_matrices=False)
            L = U_L @ Vt_L  # nearest orthogonal matrix to L
        new_delta = L @ old_delta @ R
    if use_prolora and mode in ('target', 'both'):
        new_delta = prolora_filter(new_delta, W_new, energy, null_scale)

    if spectral_cal:
        new_delta = spectral_calibrate_delta(new_delta, orig_old_delta, W_new, W_old,
                                             blend=spectral_blend)
    elif scale_calibrate:
        old_rel = torch.norm(orig_old_delta, 'fro') / (torch.norm(W_old, 'fro') + 1e-8)
        new_rel = torch.norm(new_delta, 'fro') / (torch.norm(W_new, 'fro') + 1e-8)
        if new_rel > 1e-8:
            new_delta = new_delta * (old_rel / new_rel)
    return new_delta


def apply_svd_ffn_transfer(old_delta, W_old, W_new, W_x, side='up', rank=32, energy_ratio=0.95):
    """
    SVD-based FFN transfer that handles hidden_size AND intermediate_size dimension changes.

    For up_proj / gate_proj:  shape (intermediate, hidden)
      - right singular vectors span hidden space -> project via W_x
      - left singular vectors span intermediate space -> project via intermediate alignment
    For down_proj: shape (hidden, intermediate)
      - left singular vectors span hidden space -> project via W_x.T
      - right singular vectors span intermediate space -> project via intermediate alignment

    This avoids the W_new @ W_x_pinv @ W_old_pinv chain that causes norm explosion
    when intermediate_size doubles (6144 -> 12288).

    Energy is preserved by matching singular value spectra (scaled by W_new/W_old norm ratio).
    """
    # Move to GPU if available
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old_delta = old_delta.to(dev)
    W_old = W_old.to(dev)
    W_new = W_new.to(dev)
    W_x = W_x.to(dev)

    old_int, old_hid = W_old.shape if side in ('up', 'gate') else (W_old.shape[1], W_old.shape[0])
    new_int, new_hid = W_new.shape if side in ('up', 'gate') else (W_new.shape[1], W_new.shape[0])

    if side in ('up', 'gate'):
        # old_delta: (old_int, old_hid)
        # SVD
        r = min(old_delta.shape[0], old_delta.shape[1], max(rank, 64))
        try:
            U_old, S_old, V_old_svdlr = torch.svd_lowrank(old_delta.float(), q=r)
            Vt_old = V_old_svdlr.T  # svd_lowrank returns V not Vt
        except Exception:
            U_old, S_old, V_old = torch.linalg.svd(old_delta.float(), full_matrices=False)
            Vt_old = V_old  # linalg.svd returns Vt directly

        # Project right SVs (hidden) via W_x: Vt_old shape (r, old_hid) -> (r, new_hid)
        Vt_proj = Vt_old @ W_x     # (r, new_hid)
        # Orthogonalize rows
        U_v, _, Vt_v = torch.linalg.svd(Vt_proj, full_matrices=False)
        Vt_new = U_v @ Vt_v        # (r, new_hid)

        # Map old intermediate to new intermediate via nearest-subspace projection
        # Simple approach: tile/truncate U_old to new_int dimension
        if new_int >= old_int:
            # Expand: tile U_old rows to new_int
            scale = new_int // old_int
            rem = new_int - scale * old_int
            U_new_left = torch.cat([U_old.repeat(scale, 1)[:new_int, :],], dim=0)
            if U_new_left.shape[0] < new_int:
                U_new_left = torch.cat([U_new_left, U_old[:rem, :]], dim=0)
        else:
            U_new_left = U_old[:new_int, :]
        # Orthogonalize U_new_left
        U_nl, _, Vt_nl = torch.linalg.svd(U_new_left, full_matrices=False)
        U_new_left_orth = U_nl @ Vt_nl  # (new_int, r)

        # Scale SVs by norm ratio
        norm_ratio = W_new.norm('fro') / (W_old.norm('fro') + 1e-8)
        r_use = min(S_old.shape[0], U_new_left_orth.shape[1], Vt_new.shape[0])
        S_scaled = S_old[:r_use] * norm_ratio

        # Reconstruct: (new_int, r_use) @ diag(r_use) @ (r_use, new_hid)
        new_delta_out = U_new_left_orth[:, :r_use] @ torch.diag(S_scaled) @ Vt_new[:r_use, :]
        return new_delta_out.cpu()

    else:  # down_proj: (old_hid, old_int) -> (new_hid, new_int)
        r = min(old_delta.shape[0], old_delta.shape[1], max(rank, 64))
        try:
            U_old, S_old, V_old_svdlr = torch.svd_lowrank(old_delta.float(), q=r)
            Vt_old = V_old_svdlr.T  # svd_lowrank returns V not Vt
        except Exception:
            U_old, S_old, V_old = torch.linalg.svd(old_delta.float(), full_matrices=False)
            Vt_old = V_old  # linalg.svd returns Vt directly

        # Left SVs span hidden -> project via W_x
        # U_old: (old_hid, r), W_x: (old_hid, new_hid)
        U_proj = W_x.T @ U_old    # (new_hid, r)
        U_u, _, U_vt = torch.linalg.svd(U_proj, full_matrices=False)
        U_new = U_u @ U_vt        # (new_hid, r) orthonormal cols

        # Right SVs span intermediate -> expand similarly
        # Vt_old: (r, old_int) -> (r, new_int) by tiling
        if new_int >= old_int:
            scale = new_int // old_int
            rem = new_int - scale * old_int
            Vt_new_right = Vt_old.repeat(1, scale)[:, :new_int]
            if Vt_new_right.shape[1] < new_int:
                Vt_new_right = torch.cat([Vt_new_right, Vt_old[:, :rem]], dim=1)
        else:
            Vt_new_right = Vt_old[:, :new_int]
        # Orthogonalize rows of Vt_new_right
        Vt_u, _, Vt_vt = torch.linalg.svd(Vt_new_right, full_matrices=False)
        Vt_new_right_orth = Vt_u @ Vt_vt  # (r, new_int) orthonormal rows

        norm_ratio = W_new.norm('fro') / (W_old.norm('fro') + 1e-8)
        r_use = min(S_old.shape[0], U_new.shape[1], Vt_new_right_orth.shape[0])
        S_scaled = S_old[:r_use] * norm_ratio

        new_delta_out = U_new[:, :r_use] @ torch.diag(S_scaled) @ Vt_new_right_orth[:r_use, :]
        return new_delta_out.cpu()


def spectral_calibrate_delta(new_delta, old_delta, W_new, W_old, blend=1.0):
    """
    Per-singular-value spectral calibration:
    Replace singular values of new_delta (post-xTransform) with those of
    old_delta * (||W_new||_F / ||W_old||_F), keeping the singular DIRECTIONS
    of new_delta unchanged.

    blend=1.0: pure old SVs (original behavior, spectral_calibrate)
    blend=0.5: average of old and new SVs
    blend=0.0: keep new SVs unchanged (no calibration)
    """
    norm_ratio = torch.norm(W_new, 'fro') / (torch.norm(W_old, 'fro') + 1e-8)
    # Post-xTransform singular directions (keep these)
    U_new, S_new, Vt_new = torch.linalg.svd(new_delta, full_matrices=False)
    # Source spectrum (reference)
    S_old = torch.linalg.svdvals(old_delta)  # sorted descending
    S_cal = S_new.clone()
    # Effective rank of source delta (non-trivial singular values)
    eff_r = max(1, (S_old > 1e-5 * S_old[0]).sum().item())
    n = min(eff_r, S_new.shape[0])
    S_target = S_old[:n] * norm_ratio
    S_cal[:n] = blend * S_target + (1.0 - blend) * S_new[:n]
    return U_new @ torch.diag(S_cal) @ Vt_new


def compute_correct_L(W_old_head, W_new_head, W_x, reg=1e-4):
    """
    Compute the correct left-transform for Q/K/V-proj head migration.

    Derivation:
      We want  W_new_head @ f_new(h) ?????????W_old_head @ f_old(h)
      where    f_old(h) = h_old,  f_new(h) = W_x^T h_old  (approx)
      So       W_new_head @ W_x^T ?????????W_old_head
      =>       f: delta_new = L_correct^T @ delta_old @ W_x
      where    L_correct = (W_new_head @ W_x^T)^+ @ W_old_head  (via pinv)

    In practice:
      L_correct = pinv(W_new_head @ W_x^T) @ W_old_head
                = (W_x @ W_new_head^T)^+ @ W_old_head
      shape: (new_head_size, new_head_size) when composed with W_x right factor

    For numerical stability we compute it as a regularized least-squares:
      min_{L} ||W_old_head - L^T @ W_new_head @ W_x^T||_F
      => L^T = W_old_head @ (W_new_head @ W_x^T)^+
      => L   = ((W_new_head @ W_x^T)^+)^T @ W_old_head^T

    W_old_head: (head_size, old_hidden)
    W_new_head: (head_size, new_hidden)
    W_x:        (old_hidden, new_hidden)
    returns L:  (head_size, head_size)  ?????????used as L^T in new_delta = L^T @ delta_old @ W_x
    """
    # M = W_new_head @ W_x^T  shape (head_size, old_hidden)
    M = W_new_head @ W_x.T
    # Regularized pseudo-inverse via SVD
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    S_inv = S / (S ** 2 + reg * S[0] ** 2)
    M_pinv = Vt.T @ torch.diag(S_inv) @ U.T   # (old_hidden, head_size)
    # L^T = W_old_head @ M_pinv  (head_size, head_size)
    L_T = W_old_head @ M_pinv
    return L_T.T   # return L so caller does L^T @ delta @ W_x


def estimate_per_layer_wx(W_old_Q, W_new_Q, W_x_global, n_heads, head_size, reg=1e-3):
    """
    Estimate a per-layer hidden-space transformation W_x_layer from the
    weight matrices of the current layer rather than the global embedding matrix.

    Rationale: W_x was derived from embeddings (input space). Deep layers
    often operate in a rotated/scaled representation; the true "hidden-to-hidden"
    mapping at layer l may differ significantly from the embedding-level W_x.

    We minimize:
      ||W_new_Q - W_old_Q_mapped @ W_x_layer^T||_F
    where W_old_Q_mapped is W_old_Q projected to new hidden dim via global W_x,
    Solution (least-squares): W_x_layer = (W_new_Q^T @ W_old_Q_mapped^+)^T

    Falls back to W_x_global if solution is degenerate.

    W_old_Q: (old_n_heads*head_size, old_hidden)
    W_new_Q: (new_n_heads*head_size, new_hidden)
    W_x_global: (old_hidden, new_hidden)
    returns: (old_hidden, new_hidden)
    """
    # Use only the first min(old, new)*head_size rows for a stable estimate
    n_use = min(W_old_Q.shape[0], W_new_Q.shape[0])
    Wo = W_old_Q[:n_use, :]   # (n_use, old_hidden)
    Wn = W_new_Q[:n_use, :]   # (n_use, new_hidden)

    # Solve: Wo @ Wx_layer = Wn  =>  Wx_layer = Wo^+ @ Wn
    # Regularized via SVD
    U, S, Vt = torch.linalg.svd(Wo, full_matrices=False)
    threshold = reg * S[0]
    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
    Wo_pinv = Vt.T @ torch.diag(S_inv) @ U.T   # (old_hidden, n_use)
    W_x_layer = Wo_pinv @ Wn                     # (old_hidden, new_hidden)

    # Quality check: if reconstruction is poor, fall back to global
    recon = Wo @ W_x_layer
    err = (recon - Wn).norm() / (Wn.norm() + 1e-8)
    if err > 0.5:
        return W_x_global
    return W_x_layer


def apply_xform_nspt(old_delta, W_old, W_new, L, R, energy=0.90,
                     par_weight=1.0, null_weight=1.0, scale_calibrate=True):
    """
    Null-Space Preserved Transfer (NSPT): decompose old_delta into parallel and
    null-space components w.r.t. W_old, then weight each component separately
    before applying xTransform.

    Mathematical motivation:
      old_delta = delta_par + delta_null
      delta_par  lives in W_old's row/col space   ?????????xTransform is well-conditioned
      delta_null lives in W_old's null space       ?????????xTransform via pseudo-inverse
                                                      may distort this component

    Instead of projecting null into target null space (which is smaller and wastes signal),
    we weight each component separately:
      new_delta = L @ (par_weight * delta_par + null_weight * delta_null) @ R
                = par_weight * (L @ delta_par @ R) + null_weight * (L @ delta_null @ R)

    par_weight=1.0, null_weight=1.0 ?????????equivalent to scale_calibrate only (baseline)
    par_weight>null_weight ?????????trust parallel component more
    null_weight>par_weight ?????????amplify null component (compensate pseudo-inverse attenuation)

    scale_calibrate: normalize total energy to match original relative norm
    """
    # Decompose source delta into parallel and null components
    U_par_old, _, V_par_old, _ = _get_parallel_and_null_bases(W_old, energy)
    delta_par  = _project_with_bases(old_delta, U_par_old, V_par_old)
    delta_null = old_delta - delta_par

    # Apply xTransform to each component with separate weights
    new_delta = L @ (par_weight * delta_par + null_weight * delta_null) @ R

    if scale_calibrate:
        old_rel = torch.norm(old_delta, 'fro') / (torch.norm(W_old, 'fro') + 1e-8)
        new_rel = torch.norm(new_delta, 'fro') / (torch.norm(W_new, 'fro') + 1e-8)
        if new_rel > 1e-8:
            new_delta = new_delta * (old_rel / new_rel)
    return new_delta


old_rank_ffn = 32  # default, overwritten by main after reading adapter_config


def low_rank_decompose(matrix, rank, flag=False):
    """
    matrix: typically (hidden_size, hidden_size) or similar
    """
    device = matrix.device
    if matrix.device.type == "cpu" and torch.cuda.is_available():
        matrix = matrix.cuda()
        
    # Time/Space optimization: svd_lowrank instead of exact SVD for huge dimension reduction
    # It dramatically reduces RAM and VRAM usage.
    try:
        q_limit = min(matrix.shape[0], matrix.shape[1], max(rank * 2, 64))
        U, S, V = torch.svd_lowrank(matrix.float(), q=q_limit)
    except Exception:
        U, S, V = torch.svd(matrix.float())

    if not flag:
        Ur = U[:, :rank]
        Sr = torch.diag(torch.sqrt(S[:rank]))
        Vr = V[:, :rank]
        
        B = torch.matmul(Ur, Sr)
        A = torch.matmul(Sr, Vr.t())
        
        return B.to(device), A.to(device), S[:rank].to(device)
    else:
        Ur = U[:, :rank].contiguous()
        Sr = S[:rank].view(-1,1).contiguous()
        Vr = V[:, :rank].t().contiguous()
        
        return Ur.to(device), Sr.to(device), Vr.to(device)


def cosine_similarity(qk1, qk2):
    # Ensure the tensors are 1D
    # qk1 = qk1.view(-1)
    # qk2 = qk2.view(-1)
    
    # Compute the dot product
    dot_product = torch.sum(qk1 * qk2)
    
    # Compute the L2 norms
    norm_qk1 = torch.norm(qk1, p=2)
    norm_qk2 = torch.norm(qk2, p=2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_qk1 * norm_qk2)
    return cosine_sim



def repeat_kv(project_matrix: torch.Tensor, head_dim: int, n_rep: int) -> torch.Tensor:
    """
    The project_matrix go from (num_key_value_heads * head_dim, hidden_size) to (num_attention_heads * head_dim, hidden_size)
    """
    num_kv_heads, hidden_size = project_matrix.shape
    num_key_value_heads = num_kv_heads // head_dim

    if n_rep == 1:
        return project_matrix
    project_matrix = project_matrix.view(-1, head_dim, hidden_size)
    # project_matrix = project_matrix.repeat(n_rep, 1, 1)
    project_matrix = torch.repeat_interleave(project_matrix, dim=0, repeats=n_rep)
    return project_matrix.view(-1, hidden_size)

def embedding_transformation(tokenizer1, tokenizer2, embedding1, embedding2, flag=True):
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()

    # ??????????????????????????????token
    common_tokens = set(vocab1.keys()).intersection(set(vocab2.keys()))

    # ?????????????????????????????????????????????????????
    indices1 = [vocab1[token] for token in common_tokens]
    indices2 = [vocab2[token] for token in common_tokens]

    # ?????????????????????????????????????????????????????????????????????????????????????????????????????
    embedding_common1 = embedding1[indices1]  # (num_common_tokens, hidden_size1)
    embedding_common2 = embedding2[indices2]  # (num_common_tokens, hidden_size2)

    # ???????????????????????
    embedding_common1_pinv = torch.linalg.pinv(embedding_common1)  # (hidden_size1, num_common_tokens)

    # ????????????????????????????????????
    if flag:
        print("******Transform using matrix inverse.******")
        transformation_matrix = torch.matmul(embedding_common1_pinv, embedding_common2)  # (hidden_size1, hidden_size2)
    else:
        # ????????????????????????????????????????????????????????????????????????????????????????????????
        transformation_matrix, _ = torch.lstsq(embedding_common2, embedding_common1)


    return transformation_matrix


def max_layer_similarity_mapping(A):
    # ????????????????????????A?????????????????
    n, m = A.shape
    if n > m:
        A = A.T
        n, m = m, n
    
    # ??????????????????dp????????????????????????????????????????????????????????????????????????????????????
    dp = np.full((n, m), -float('inf'))
    # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    path = np.zeros((n, m), dtype=int)
    
    # ??????????????????dp?????????????????
    for j in range(m):
        if j <= m - n:
            dp[0][j] = A[0][j]
    
    # ????????????dp????????????
    for i in range(1, n):
        for j in range(i, m):  # j??????????????????????????????i?????????????????????????????????????????
            if j - i <= m - n:  # ????????????????????????????????????????????????m-n
                max_value = -float('inf')
                max_index = -1
                # ?????????????????????????????????????????????????????????????????
                for k in range(i-1, j):
                    if dp[i-1][k] + A[i][j] > max_value:
                        max_value = dp[i-1][k] + A[i][j]
                        max_index = k
                dp[i][j] = max_value
                path[i][j] = max_index
    
    # ?????????????????????????????????????????????????????
    max_sum = -float('inf')
    last_index = -1
    for j in range(n-1, m):
        if dp[n-1][j] > max_sum:
            max_sum = dp[n-1][j]
            last_index = j
    
    # ????????????path?????????????????????????????????????????????????????????????????
    mapping = [0] * n
    for i in range(n-1, -1, -1):
        mapping[i] = last_index
        last_index = path[i][last_index]
    
    return mapping, max_sum


def max_layer_similarity_mapping_hungarian(A):
    """
    One-to-one layer assignment via Hungarian algorithm.
    Returns mapping old_layer -> new_layer with unique targets.
    Supports A shaped either (n_old, n_new) or (n_new, n_old).
    """
    n0, n1 = A.shape
    transposed = False
    if n0 > n1:
        A = A.T
        n0, n1 = A.shape
        transposed = True

    # Hungarian minimizes cost, so use negative similarity.
    row_ind, col_ind = linear_sum_assignment(-A)
    mapping = [0] * n0
    for r, c in zip(row_ind, col_ind):
        mapping[r] = int(c)
    max_sum = float(A[row_ind, col_ind].sum())

    # Mapping semantics stay old->new regardless of original orientation.
    # If input was transposed, n0 already corresponds to old-layer count.
    return mapping, max_sum


def max_head_similarity_mapping(A):
    # ????????????????????????A?????????????????
    n, m = A.shape
    
    # ????????????n????????????????????????????????????m
    if n > m:
        raise ValueError("????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????attention head?????????????????????????????????????????????????????????????????????????????????????????????????????????")
    
    # ?????????????????????????????????????????????????????????????????????????????????????????????????????
    # ????????????linear_sum_assignment??????????????????????????????????????????????????????????????????????????????????????????A????????????
    cost_matrix = -A
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # ??????????????????????????????????????????????????????
    max_sum = A[row_ind, col_ind].sum()
    
    return col_ind, max_sum


def load_model(model_path, device_id) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = model_path

    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map=device_id,)
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype= torch.float16 if "Llama" in base_model else "auto",
            device_map=device_id,
            trust_remote_code=True,
        ) 
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    # Return layers
    num_layer = model.config.num_hidden_layers
    return tokenizer, model.state_dict(), num_layer

def tensor_inverse(A, type="left"):
    if type == "left":
        # A^T A
        AtA = torch.matmul(A.T, A)
        # (A^T A)^-1
        AtA_inv = torch.inverse(AtA)
        # (A^T A)^-1 A^T
        A_inv = torch.matmul(AtA_inv, A.T)
    elif type == "right":
        # A A^T
        AAt = torch.matmul(A, A.T)
        # (A A^T)^-1
        AAt_inv = torch.inverse(AAt)
        # A^T (A A^T)^-1
        A_inv = torch.matmul(A.T, AAt_inv)
    return A_inv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--new_model', type=str, required=True)
    parser.add_argument('--old_model', type=str, required=True)
    parser.add_argument('--old_lora_path', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--layer_method', type=str, default="xtransform")
    parser.add_argument('--attention_method', type=str, default="xtransform")
    parser.add_argument('--sim_metric', type=str, default="CKA")
    parser.add_argument('--qwen_layer_mapping_mode', type=str, default='fixed',
                        choices=['fixed', 'cka_monotonic', 'cka_hungarian'],
                        help="Layer mapping mode for qwen3_1_7B -> qwen3_8B: "
                             "fixed=int(i*36/28), cka_monotonic=DP monotonic CKA max, "
                             "cka_hungarian=one-to-one Hungarian on CKA.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_prolora', action='store_true', help="Use ProLoRA subspace projection to maintain zero-shot capability on base model weights")
    parser.add_argument('--prolora_energy', type=float, default=0.90, help="Energy ratio for parallel subspace in ProLoRA")
    parser.add_argument('--prolora_null_scale', type=float, default=1.0, help="Scaling factor for null-space (residual) component in ProLoRA (1.0=no filtering, 0.0=parallel only)")
    parser.add_argument('--prolora_mode', type=str, default='target', choices=['source', 'target', 'both'],
                        help="ProLoRA filtering side: 'source' filters old_delta before mapping, 'target' filters new_delta after mapping, 'both' applies both")
    parser.add_argument('--scale_calibrate', action='store_true',
                        help="Rescale new_delta to preserve relative Frobenius norm (||delta||/||W||) after xTransform")
    parser.add_argument('--use_nspt', action='store_true',
                        help="Use Null-Space Preserved Transfer: split delta into parallel/null parts, transfer separately")
    parser.add_argument('--nspt_energy', type=float, default=0.90,
                        help="Energy ratio for parallel subspace in NSPT (default: 0.90)")
    parser.add_argument('--nspt_par_weight', type=float, default=1.0,
                        help="Weight for parallel component in NSPT (default: 1.0)")
    parser.add_argument('--nspt_null_weight', type=float, default=1.0,
                        help="Weight for null-space component in NSPT (default: 1.0)")
    parser.add_argument('--spectral_calibrate', action='store_true',
                        help="Per-SV spectral calibration: match singular value spectrum of migrated delta "
                             "to source delta (normalized by model norms). Strictly better than "
                             "--scale_calibrate when xTransform distorts the spectrum unevenly.")
    parser.add_argument('--fill_unmatched_heads', action='store_true',
                        help="Fill target attention heads with no matched source head by reusing "
                             "the nearest-neighbor source head's xTransform. Fills the 12 zero-delta "
                             "heads in 1B->2B migration (24 src heads -> 36 tgt heads).")
    parser.add_argument('--new_rank', type=int, default=0,
                        help="Override LoRA rank for the saved migrated LoRA (0 = use source rank). "
                             "After per-head xTransform, new_delta_W_Q has rank up to 24*old_rank=768; "
                             "using a higher rank (e.g. 64, 128) preserves more head-level information "
                             "than the default old_rank=32 compression.")
    parser.add_argument('--finetuned_head_sim', action='store_true',
                        help="Use fine-tuned weights (W_old + delta_W) instead of pre-trained weights "
                             "for the old model's head similarity computation. This matches heads based "
                             "on their actual fine-tuned functional roles rather than pre-training behavior.")
    parser.add_argument('--polar_head', action='store_true',
                        help="Use polar decomposition of the per-head xTransform L matrix (L = U@S@Vt "
                             "?????????Q = U@Vt), making the left head-space transform purely orthogonal "
                             "(no scale distortion). Combined with spectral_calibrate for best effect.")
    parser.add_argument('--procrustes_L', action='store_true',
                        help="Use Procrustes-optimal orthogonal L for each head: "
                             "L* = argmin_Q ||W_new_head - Q W_old_head W_x||_F. "
                             "Solved via SVD: L* = U @ Vt of (W_new_head @ (W_old_head @ W_x)^T). "
                             "Strictly the best orthogonal approximation; does not require inversion "
                             "of W_x and is thus more numerically stable than correct_transform. "
                             "Use with --spectral_calibrate for norm correction.")
    parser.add_argument('--correct_transform', action='store_true',
                        help="Use the mathematically correct inverse transform for Q/K/V heads: "
                             "L_correct = pinv(W_new_head @ W_x^T) @ W_old_head, instead of the "
                             "approximate L = W_old_head @ W_x @ W_new_head^T. Addresses the hidden "
                             "assumption that W_x is orthogonal and L^T ?????????L^{-1}.")
    parser.add_argument('--per_layer_wx', action='store_true',
                        help="Estimate a per-layer hidden-space transformation W_x_layer from the "
                             "current layer's Q weight matrices (least-squares fit), rather than "
                             "using the global embedding-derived W_x for all layers. Addresses the "
                             "assumption that the input hidden representation transform is constant.")
    parser.add_argument('--per_layer_wx_reg', type=float, default=1e-3,
                        help="Regularization coefficient for per-layer Wx estimation (default: 1e-3). "
                             "Higher values fall back towards global Wx; lower values fit more locally.")
    parser.add_argument('--correct_transform_reg', type=float, default=1e-4,
                        help="Regularization for pinv in correct_transform (default: 1e-4).")
    parser.add_argument('--norm_correct', action='store_true',
                        help="Post-hoc per-layer norm correction: rescale migrated LoRA deltas so that "
                             "||delta_new||_F / ||W_new||_F == ||delta_old||_F / ||W_old||_F. "
                             "Addresses systematic over-scaling (ratio ~1.2-1.8x) introduced by "
                             "the bilinear xTransform when W_x is not norm-preserving. "
                             "Applied after spectral_calibrate if both are enabled (should be redundant "
                             "but fixes residual mismatch from ill-conditioned per-head transforms).")
    parser.add_argument('--no_left_transform', action='store_true',
                        help="Use identity for left (L) transform instead of head-alignment matrix. "
                             "new_delta = old_delta @ W_x (pure W_x transform), preserving the "
                             "original B-matrix output directions while adapting input directions. "
                             "Avoids potential distortion from head-space L rotation.")
    parser.add_argument('--fill_missing_layers', action='store_true',
                        help="After main loop, fill target layers with no mapped source layer by copying "
                             "LoRA weights from the nearest mapped target layer. Critical for 1.7B->8B "
                             "where 8/36 target layers are skipped by the linear layer_mapping.")
    parser.add_argument('--lora_alpha_scale', type=float, default=1.0,
                        help="Scale factor for lora_alpha: saved lora_alpha = save_rank * lora_alpha_scale. "
                             "Default 1.0 = alpha/r = 1.0 (no scaling). "
                             "Try 0.5 or 2.0 to adjust the effective LoRA magnitude.")
    parser.add_argument('--spectral_blend', type=float, default=1.0,
                        help="Blend ratio [0,1] for spectral_calibrate singular values: "
                             "blended_sv = blend * sv_old + (1-blend) * sv_new. "
                             "Default 1.0 = pure sv_old = spectral_calibrate behavior (backwards compat). "
                             "0.5 = average of old and new SVs. "
                             "Only used when --spectral_calibrate is enabled.")
    parser.add_argument('--spectral_blend_cka', action='store_true',
                        help="Use per-layer CKA similarity as spectral_blend for each layer. "
                             "Layers with high CKA (well-aligned) use blend???.0 (preserve source spectrum). "
                             "Layers with low CKA (poorly-aligned) use blend?????????.5 (allow transformed spectrum). "
                             "Overrides global --spectral_blend when enabled. "
                             "Suffix: _ckabl.")
    parser.add_argument('--svd_transfer', action='store_true',
                        help="SVD Procrustes Transfer: bypass bilinear xTransform entirely. "
                             "Decomposes old_delta via SVD, projects right singular vectors to "
                             "new hidden space via W_x, orthogonalizes via Procrustes, and "
                             "rescales singular values by ||W_new||/||W_old||. "
                             "Left singular vectors (head-dim space) are kept unchanged since "
                             "head_size is the same in both models (64-dim). "
                             "Cannot be combined with --spectral_calibrate (SVDT is self-calibrating).")
    parser.add_argument('--act_align_path', type=str, default=None,
                        help="Path to directory containing activation-based alignment matrices "
                             "R_l_{i}.pt (shape: old_hidden x new_hidden) produced by "
                             "collect_activations.py. When set, overrides the embedding-derived "
                             "W_x and per_layer_wx: uses the runtime-activation-aligned R_l for "
                             "each layer, which captures the true hidden-representation geometry "
                             "rather than only embedding-space alignment. ")
    parser.add_argument('--act_align_blend', type=float, default=1.0,
                        help="Blend ratio for activation-alignment matrices: "
                             "W_x_layer = blend * R_l + (1-blend) * W_x_global. "
                             "Default 1.0 = pure activation alignment. Lower values (e.g. 0.5) "
                             "regularize R_l toward global W_x to handle null-space directions "
                             "where N_samples << hidden_dim makes R_l underdetermined.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    A = torch.load(f"./tmp/{args.old_model.split('/')[-2]}_{args.new_model.split('/')[-2]}_{args.sim_metric}.pt").cpu().numpy()
    if args.old_model.split('/')[-2] == "qwen3_1_7B" and args.new_model.split('/')[-2] == "qwen3_8B":
        if args.qwen_layer_mapping_mode == 'fixed':
            layer_mapping = [int(i * 36 / 28) for i in range(28)]
        elif args.qwen_layer_mapping_mode == 'cka_monotonic':
            layer_mapping, _ = max_layer_similarity_mapping(A)
        else:  # cka_hungarian
            layer_mapping, _ = max_layer_similarity_mapping_hungarian(A)
    # The following layer_methods only work for the MiniCPM.
    elif args.layer_method == "xtransform":
        layer_mapping, _ = max_layer_similarity_mapping(A)
    elif args.layer_method == "first":
        layer_mapping = [i for i in range(A.shape[1])]
    elif args.layer_method == "medium":
        layer_mapping = [i for i in range(7, 47, 1)]
    elif args.layer_method == "last":
        layer_mapping = [i for i in range(A.shape[0]-A.shape[1], A.shape[0], 1)]

    # Compute per-layer CKA similarity for spectral_blend_cka feature
    # A[old_j, new_i] = CKA similarity when A has shape (n_old, n_new); transposed otherwise
    # A may be (n_old, n_new) or (n_new, n_old). We always want cka(old_i, new_j).
    if A.shape[0] <= A.shape[1]:  # (n_old, n_new)
        cka_per_layer = [float(A[i, layer_mapping[i]]) for i in range(len(layer_mapping))]
    else:  # (n_new, n_old)
        cka_per_layer = [float(A[layer_mapping[i], i]) for i in range(len(layer_mapping))]
    print(f"CKA per layer (new???old): {[f'{v:.3f}' for v in cka_per_layer]}")

    old_lora_weights = load_peft_weights(args.old_lora_path)
    with open(args.old_model + "config.json", "r") as f:
        data = json.load(f)
        if args.old_model.split('/')[-2] == "bloom-560m":
            old_hidden_size, old_hidden_layers = data['n_embed'], data['n_layer']
        else:
            old_hidden_size, old_hidden_layers = data['hidden_size'], data['num_hidden_layers']
        old_num_attention_heads = data['num_attention_heads']
        old_head_size = old_hidden_size // old_num_attention_heads
        if "MiniCPM" in args.old_model.split('/')[-2] or "qwen3" in args.old_model.lower():
            old_num_key_value_heads = data["num_key_value_heads"]
            old_num_rep = old_num_attention_heads // old_num_key_value_heads

    with open(args.new_model + "config.json", "r") as f:
        data = json.load(f)
        if args.new_model.split('/')[-2] == "bloomz-1b1":
            new_hidden_size, new_hidden_layers = data['n_embed'], data['n_layer']
        else:
            new_hidden_size, new_hidden_layers = data['hidden_size'], data['num_hidden_layers']
        new_num_attention_heads = data['num_attention_heads']
        new_head_size = new_hidden_size // new_num_attention_heads
        if args.new_model.split('/')[-2] == "Qwen2.5-3B" or "Meta-Llama-3-8B" in args.new_model.split('/')[-2] or "qwen3" in args.new_model.lower():
            new_num_key_value_heads = data["num_key_value_heads"]
            new_num_rep = new_num_attention_heads // new_num_key_value_heads
            
    with open(args.old_lora_path + "adapter_config.json", "r") as f:
        data = json.load(f)
        old_rank = data['r']
        old_rank_ffn = old_rank  # used in apply_svd_ffn_transfer
        target_modules = data['target_modules']
    save_rank = args.new_rank if args.new_rank > 0 else old_rank
    
    new_lora_weights = {}
        
    old_tokenizer, old_model_dic, _ = load_model(args.old_model, "cuda:0")
    new_tokenizer, new_model_dic, _ = load_model(args.new_model, "cuda:0")
    if (args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format" or 
         "Meta-Llama-3-8B" in args.new_model.split('/')[-2] or
         "qwen3" in args.old_model.lower() or "qwen3" in args.new_model.lower()
        ):    
        E1, E2 = old_model_dic["model.embed_tokens.weight"].to(device="cpu", dtype=torch.float32), new_model_dic["model.embed_tokens.weight"].to(device="cpu", dtype=torch.float32)
        W_x = embedding_transformation(old_tokenizer, new_tokenizer, E1, E2)
        del E1, E2
    elif args.old_model.split('/')[-2] == "bloom-560m":
        E1, E2 = old_model_dic["transformer.word_embeddings.weight"].to(device="cpu", dtype=torch.float32), new_model_dic["transformer.word_embeddings.weight"].to(device="cpu", dtype=torch.float32)
        W_x = embedding_transformation(old_tokenizer, new_tokenizer, E1, E2)
        del E1, E2
    elif args.old_model.split('/')[-2] == "pythia-1b" or args.new_model.split('/')[-2] == "Qwen2.5-3B":
        W_x = torch.eye(old_hidden_size).to(device="cpu", dtype=torch.float32)
    del old_tokenizer, new_tokenizer

    # Pre-compute polar decomposition of W_x once (shared by all O-proj calls).
    # polar(W_x.T) = polar_W_x.T, so we store polar_W_x and use its transpose for L.
    if args.polar_head:
        _U_Wx, _, _Vt_Wx = torch.linalg.svd(W_x, full_matrices=False)
        polar_W_x = _U_Wx @ _Vt_Wx   # shape (old_hidden, new_hidden), nearest orthogonal to W_x
        del _U_Wx, _Vt_Wx
    else:
        polar_W_x = W_x  # alias (no copy)

    # Load activation-based per-layer alignment matrices if provided
    act_align_matrices = None
    if args.act_align_path:
        import glob
        act_align_matrices = {}
        for fpath in glob.glob(os.path.join(args.act_align_path, 'R_l_*.pt')):
            idx = int(os.path.basename(fpath).replace('R_l_', '').replace('.pt', ''))
            act_align_matrices[idx] = torch.load(fpath, map_location='cpu').to(torch.float32)
        print(f'Loaded {len(act_align_matrices)} activation-alignment matrices from {args.act_align_path}')

    for i in range(len(layer_mapping)):
        if args.old_model.split('/')[-2] == "pythia-1b":
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")
            W_old_QKV = old_model_dic[f"gpt_neox.layers.{i}.attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_old_O = old_model_dic[f"gpt_neox.layers.{i}.attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            delta_W_QKV = old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_O = old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.dense.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.gpt_neox.layers.{i}.attention.dense.lora_A.weight"].to(device="cpu", dtype=torch.float32)
        
        elif args.old_model.split('/')[-2] == "bloom-560m":
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")
            W_old_QKV = old_model_dic[f"transformer.h.{i}.self_attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_old_O = old_model_dic[f"transformer.h.{i}.self_attention.dense.weight"].to(device="cpu", dtype=torch.float32)
            if "dense_h_to_4h" in target_modules:
                W_old_U = old_model_dic[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"].to(device="cpu", dtype=torch.float32)
                
            if "dense_4h_to_h" in target_modules:
                W_old_D = old_model_dic[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"].to(device="cpu", dtype=torch.float32)
            
            delta_W_QKV = old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.query_key_value.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.query_key_value.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_O = old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.dense.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.transformer.h.{i}.self_attention.dense.lora_A.weight"].to(device="cpu", dtype=torch.float32)
        
        elif args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format":
            print(f"Layer {layer_mapping[i]} is mapping to Layer {i}.")

            W_old_Q = old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            # https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/main/modeling_minicpm.py#L286
            W_old_K = repeat_kv(old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_V = repeat_kv(old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_O = old_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "up_proj" in target_modules:    
                W_old_U = old_model_dic[f"model.layers.{layer_mapping[i]}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "gate_proj" in target_modules:
                W_old_G = old_model_dic[f"model.layers.{layer_mapping[i]}.mlp.gate_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_old_D = old_model_dic[f"model.layers.{layer_mapping[i]}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

            if "adalora" in args.old_lora_path:
                print(f"Adalora Transformation")
                delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_E"].to(device="cpu", dtype=torch.float32))
                delta_W_K = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_E"].to(device="cpu", dtype=torch.float32)), old_head_size, old_num_rep)
                delta_W_V = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_E"].to(device="cpu", dtype=torch.float32)), old_head_size, old_num_rep)
                delta_W_O = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B"].to(device="cpu", dtype=torch.float32) @ (old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A"].to(device="cpu", dtype=torch.float32) * old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_E"].to(device="cpu", dtype=torch.float32))
            else:
                delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                delta_W_K = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
                delta_W_V = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
                delta_W_O = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)

        elif (args.old_model.split('/')[-2] == "Qwen1.5-1.8B") or ("Llama-2-7b" in args.old_model.split('/')[-2]) or ("qwen3" in args.old_model.lower()):
            print(f"Layer {i} is mapping to Layer {layer_mapping[i]}.")

            W_old_Q = old_model_dic[f"model.layers.{i}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_old_K = repeat_kv(old_model_dic[f"model.layers.{i}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_V = repeat_kv(old_model_dic[f"model.layers.{i}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            W_old_O = old_model_dic[f"model.layers.{i}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            
            delta_W_Q = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
            delta_W_K = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            delta_W_V = repeat_kv(old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32), old_head_size, old_num_rep)
            delta_W_O = old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)

            if "up_proj" in target_modules:    
                W_old_U = old_model_dic[f"model.layers.{i}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_old_D = old_model_dic[f"model.layers.{i}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)
        if args.new_model.split('/')[-2] == "pythia-1.4b":
            W_new_QKV = new_model_dic[f"gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"gpt_neox.layers.{layer_mapping[i]}.attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_QKV = torch.zeros(W_new_QKV.shape[0], W_new_QKV.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        
        elif args.new_model.split('/')[-2] == "bloomz-1b1":
            W_new_QKV = new_model_dic[f"transformer.h.{layer_mapping[i]}.self_attention.query_key_value.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"transformer.h.{layer_mapping[i]}.self_attention.dense.weight"].to(device="cpu", dtype=torch.float32)

            if "dense_h_to_4h" in target_modules:
                W_new_U = new_model_dic[f"transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.weight"].to(device="cpu", dtype=torch.float32)

            if "dense_4h_to_h" in target_modules:
                W_new_D = new_model_dic[f"transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_QKV = torch.zeros(W_new_QKV.shape[0], W_new_QKV.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        elif args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format":
            W_new_Q = new_model_dic[f"model.layers.{i}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_K = new_model_dic[f"model.layers.{i}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_V = new_model_dic[f"model.layers.{i}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_O = new_model_dic[f"model.layers.{i}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "up_proj" in target_modules:    
                W_new_U = new_model_dic[f"model.layers.{i}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "gate_proj" in target_modules:
                W_new_G = new_model_dic[f"model.layers.{i}.mlp.gate_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_new_D = new_model_dic[f"model.layers.{i}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_Q = torch.zeros(W_new_Q.shape[0], W_new_Q.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_K = torch.zeros(W_new_K.shape[0], W_new_K.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_V = torch.zeros(W_new_V.shape[0], W_new_V.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)
        elif args.new_model.split('/')[-2] == "Qwen2.5-3B" or  "Meta-Llama-3-8B" in args.new_model.split('/')[-2] or "qwen3" in args.new_model.lower():
            W_new_Q = new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.q_proj.weight"].to(device="cpu", dtype=torch.float32)
            W_new_K = repeat_kv(new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.k_proj.weight"].to(device="cpu", dtype=torch.float32), new_head_size, new_num_rep)
            W_new_V = repeat_kv(new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.v_proj.weight"].to(device="cpu", dtype=torch.float32), new_head_size, new_num_rep)
            W_new_O = new_model_dic[f"model.layers.{layer_mapping[i]}.self_attn.o_proj.weight"].to(device="cpu", dtype=torch.float32)

            new_delta_W_Q = torch.zeros(W_new_Q.shape[0], W_new_Q.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_K = torch.zeros(W_new_K.shape[0], W_new_K.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_V = torch.zeros(W_new_V.shape[0], W_new_V.shape[1]).to(device="cpu", dtype=torch.float32)
            new_delta_W_O = torch.zeros(W_new_O.shape[0], W_new_O.shape[1]).to(device="cpu", dtype=torch.float32)

            if "up_proj" in target_modules:    
                W_new_U = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
            if "down_proj" in target_modules:
                W_new_D = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)

        # Per-layer W_x estimation (MiniCPM-2B or Qwen3-8B when weights are available)
        qwen3_pair = (args.old_model.split('/')[-2] == "qwen3_1_7B"
                      and args.new_model.split('/')[-2] == "qwen3_8B")
        minicpm_pair = (args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format"
                        and args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format")
        if act_align_matrices is not None and i in act_align_matrices:
            # Activation-based alignment: use runtime hidden-state derived R_l
            # Blend with global W_x to regularize null-space directions (N << d)
            R_l = act_align_matrices[i]
            blend = args.act_align_blend
            if blend < 1.0:
                # W_x shape: (old_hidden, new_hidden), same as R_l
                W_x_layer = blend * R_l + (1.0 - blend) * W_x
            else:
                W_x_layer = R_l
            print(f'  Layer {i}: act_align R_l (blend={blend:.2f}) shape={tuple(W_x_layer.shape)}')
        elif args.per_layer_wx and (minicpm_pair or qwen3_pair):
            W_x_layer = estimate_per_layer_wx(
                W_old_Q, W_new_Q, W_x,
                old_num_attention_heads, old_head_size,
                reg=args.per_layer_wx_reg)
        else:
            W_x_layer = W_x
        if args.polar_head:
            _U_l, _, _Vt_l = torch.linalg.svd(W_x_layer, full_matrices=False)
            polar_W_x_layer = _U_l @ _Vt_l
            del _U_l, _Vt_l
        else:
            polar_W_x_layer = W_x_layer

        # Per-layer spectral blend: use CKA similarity if enabled
        if args.spectral_blend_cka and i < len(cka_per_layer):
            layer_spectral_blend = cka_per_layer[i]
        else:
            layer_spectral_blend = args.spectral_blend

        # ????????????????????????????????????????????????head similarity
        # Optimized: precompute per-head vectors once, then batch cosine similarity
        if args.old_model.split('/')[-2] == "pythia-1b" or args.old_model.split('/')[-2] == "bloom-560m":
            # --- pythia/bloom: O(n_old * n_new) naive loop (shapes differ) ---
            head_sim_matrix = np.zeros((old_num_attention_heads, new_num_attention_heads))
            for a in range(old_num_attention_heads):
                for b in range(new_num_attention_heads):
                    Q1, K1, V1 = W_old_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :], W_old_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :], W_old_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :]
                    Q2, K2, V2 = W_new_QKV[(b*new_head_size*3):(b*new_head_size*3+new_head_size), :], W_new_QKV[(b*new_head_size*3+new_head_size):(b*new_head_size*3+2*new_head_size), :], W_new_QKV[(b*new_head_size*3+2*new_head_size):((b+1)*new_head_size*3), :]
                    O1, O2 = W_old_O[:, a*old_head_size:(a+1)*old_head_size], W_new_O[:, b*new_head_size:(b+1)*new_head_size]
                    qk1, qk2 = W_x.T @ Q1.T @ K1 @ W_x, Q2.T @ K2
                    vo1, vo2 = W_x.T @ V1.T @ O1.T @ W_x, V2.T @ O2.T
                    vec1 = torch.cat([qk1.view(-1), vo1.view(-1)], dim=0)
                    vec2 = torch.cat([qk2.view(-1), vo2.view(-1)], dim=0)
                    head_sim_matrix[a, b] = cosine_similarity(vec1, vec2).item()
        else:
            # --- MiniCPM/Llama: precompute per-head qk/vo in SOURCE space, then cosine ---
            # Key insight: qk1[a] and vo1[a] only depend on old-head a (not b).
            # OLD code recomputed them 864 times (24????36 pairs). Now compute only 24+36 times.
            # Speeds up 29???? by avoiding redundant (2304, 2304) matrix multiplications.
            old_qkvo = []  # list of (2*new_h????, ) normalized vectors
            for a in range(old_num_attention_heads):
                if args.finetuned_head_sim:
                    Q1 = (W_old_Q + delta_W_Q)[a*old_head_size:(a+1)*old_head_size, :]
                    K1 = (W_old_K + delta_W_K)[a*old_head_size:(a+1)*old_head_size, :]
                    V1 = (W_old_V + delta_W_V)[a*old_head_size:(a+1)*old_head_size, :]
                    O1 = (W_old_O + delta_W_O)[:, a*old_head_size:(a+1)*old_head_size]
                else:
                    Q1 = W_old_Q[a*old_head_size:(a+1)*old_head_size, :]
                    K1 = W_old_K[a*old_head_size:(a+1)*old_head_size, :]
                    V1 = W_old_V[a*old_head_size:(a+1)*old_head_size, :]
                    O1 = W_old_O[:, a*old_head_size:(a+1)*old_head_size]
                qk1 = W_x.T @ Q1.T @ K1 @ W_x
                vo1 = W_x.T @ V1.T @ O1.T @ W_x
                vec1 = torch.cat([qk1.reshape(-1), vo1.reshape(-1)])
                old_qkvo.append(vec1 / (vec1.norm() + 1e-8))

            new_qkvo = []  # list of (2*new_h????, ) normalized vectors
            for b in range(new_num_attention_heads):
                Q2 = W_new_Q[b*new_head_size:(b+1)*new_head_size, :]
                K2 = W_new_K[b*new_head_size:(b+1)*new_head_size, :]
                V2 = W_new_V[b*new_head_size:(b+1)*new_head_size, :]
                O2 = W_new_O[:, b*new_head_size:(b+1)*new_head_size]
                qk2 = Q2.T @ K2
                vo2 = V2.T @ O2.T
                vec2 = torch.cat([qk2.reshape(-1), vo2.reshape(-1)])
                new_qkvo.append(vec2 / (vec2.norm() + 1e-8))

            head_sim_matrix = np.zeros((old_num_attention_heads, new_num_attention_heads))
            for a in range(old_num_attention_heads):
                for b in range(new_num_attention_heads):
                    head_sim_matrix[a, b] = torch.dot(old_qkvo[a], new_qkvo[b]).item()

        head_mapping, _ = max_head_similarity_mapping(head_sim_matrix)
        for a in range(old_num_attention_heads):
            # if args.old_model.split('/')[-2] == "pythia-1b":
            #     W_x = W_old[a*old_head_size*3:(a+1)*old_head_size*3, :] @ W_new[head_mapping[a]*new_head_size*3:(head_mapping[a]+1)*new_head_size*3, :].T
            #     new_delta_W[head_mapping[a]*new_head_size*3:(head_mapping[a]+1)*new_head_size*3, :] = (delta_W[a*old_head_size*3:(a+1)*old_head_size*3, :].T @ W_x).T
            if args.old_model.split('/')[-2] == "pythia-1b" or args.new_model.split('/')[-2] == "bloomz-1b1":
                W_Q_x = W_old_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3):(head_mapping[a]*new_head_size*3+new_head_size), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3):(head_mapping[a]*new_head_size*3+new_head_size), :] = W_Q_x.T @ delta_W_QKV[(a*old_head_size*3):(a*old_head_size*3+old_head_size), :] @ W_x

                W_K_x = W_old_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3+new_head_size):(head_mapping[a]*new_head_size*3+2*new_head_size), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3+new_head_size):(head_mapping[a]*new_head_size*3+2*new_head_size), :] = W_K_x.T @ delta_W_QKV[(a*old_head_size*3+old_head_size):(a*old_head_size*3+2*old_head_size), :] @ W_x

                W_V_x = W_old_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :] @ W_x @ W_new_QKV[(head_mapping[a]*new_head_size*3+2*new_head_size):((head_mapping[a]+1)*new_head_size*3), :].T
                new_delta_W_QKV[(head_mapping[a]*new_head_size*3+2*new_head_size):((head_mapping[a]+1)*new_head_size*3), :] = W_V_x.T @ delta_W_QKV[(a*old_head_size*3+2*old_head_size):((a+1)*old_head_size*3), :] @ W_x

                W_O_x = W_old_O[:, (a*old_head_size):((a+1)*old_head_size)].T @ W_x @ W_new_O[:, (head_mapping[a]*new_head_size):((head_mapping[a]+1)*new_head_size)]
                new_delta_W_O[:, (head_mapping[a]*new_head_size):((head_mapping[a]+1)*new_head_size)] = W_x.T @ delta_W_O[:, a*old_head_size:(a+1)*old_head_size] @ W_O_x
            
            else:
                if args.attention_method == "xtransform":
                    old_start = a * old_head_size
                    old_end = (a + 1) * old_head_size
                    new_start = head_mapping[a] * new_head_size
                    new_end = (head_mapping[a] + 1) * new_head_size

                    W_Q_x = W_old_Q[old_start:old_end, :] @ W_x_layer @ W_new_Q[new_start:new_end, :].T
                    old_delta_q = delta_W_Q[old_start:old_end, :]
                    if args.svd_transfer:
                        new_delta_W_Q[new_start:new_end, :] = apply_svd_transfer(
                            old_delta_q,
                            W_old_Q[old_start:old_end, :], W_new_Q[new_start:new_end, :],
                            W_x_layer,
                        )
                    elif args.procrustes_L:
                        L_Q_T = procrustes_L(
                            W_old_Q[old_start:old_end, :], W_new_Q[new_start:new_end, :],
                            W_x_layer).T
                    elif args.correct_transform:
                        L_Q_T = compute_correct_L(
                            W_old_Q[old_start:old_end, :], W_new_Q[new_start:new_end, :],
                            W_x_layer, reg=args.correct_transform_reg).T
                    else:
                        L_Q_T = W_Q_x.T
                    if not args.svd_transfer:
                        if args.use_nspt:
                            new_delta_W_Q[new_start:new_end, :] = apply_xform_nspt(
                                old_delta_q, W_old_Q[old_start:old_end, :], W_new_Q[new_start:new_end, :],
                                L_Q_T, W_x_layer,
                                energy=args.nspt_energy,
                                par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                                scale_calibrate=args.scale_calibrate,
                            )
                        else:
                            new_delta_W_Q[new_start:new_end, :] = apply_xform_with_prolora(
                                old_delta_q, W_old_Q[old_start:old_end, :], W_new_Q[new_start:new_end, :],
                                L_Q_T, W_x_layer,
                                args.use_prolora, args.prolora_mode,
                                args.prolora_energy, args.prolora_null_scale, args.scale_calibrate,
                                args.spectral_calibrate, args.polar_head,
                                no_left_transform=args.no_left_transform,
                                spectral_blend=layer_spectral_blend,
                            )

                    W_K_x = W_old_K[old_start:old_end, :] @ W_x_layer @ W_new_K[new_start:new_end, :].T
                    old_delta_k = delta_W_K[old_start:old_end, :]
                    if args.svd_transfer:
                        new_delta_W_K[new_start:new_end, :] = apply_svd_transfer(
                            old_delta_k,
                            W_old_K[old_start:old_end, :], W_new_K[new_start:new_end, :],
                            W_x_layer,
                        )
                    elif args.procrustes_L:
                        L_K_T = procrustes_L(
                            W_old_K[old_start:old_end, :], W_new_K[new_start:new_end, :],
                            W_x_layer).T
                    elif args.correct_transform:
                        L_K_T = compute_correct_L(
                            W_old_K[old_start:old_end, :], W_new_K[new_start:new_end, :],
                            W_x_layer, reg=args.correct_transform_reg).T
                    else:
                        L_K_T = W_K_x.T
                    if not args.svd_transfer:
                        if args.use_nspt:
                            new_delta_W_K[new_start:new_end, :] = apply_xform_nspt(
                                old_delta_k, W_old_K[old_start:old_end, :], W_new_K[new_start:new_end, :],
                                L_K_T, W_x_layer,
                                energy=args.nspt_energy,
                                par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                                scale_calibrate=args.scale_calibrate,
                            )
                        else:
                            new_delta_W_K[new_start:new_end, :] = apply_xform_with_prolora(
                                old_delta_k, W_old_K[old_start:old_end, :], W_new_K[new_start:new_end, :],
                                L_K_T, W_x_layer,
                                args.use_prolora, args.prolora_mode,
                                args.prolora_energy, args.prolora_null_scale, args.scale_calibrate,
                                args.spectral_calibrate, args.polar_head,
                                no_left_transform=args.no_left_transform,
                                spectral_blend=layer_spectral_blend,
                            )

                    W_V_x = W_old_V[old_start:old_end, :] @ W_x_layer @ W_new_V[new_start:new_end, :].T
                    old_delta_v = delta_W_V[old_start:old_end, :]
                    if args.svd_transfer:
                        new_delta_W_V[new_start:new_end, :] = apply_svd_transfer(
                            old_delta_v,
                            W_old_V[old_start:old_end, :], W_new_V[new_start:new_end, :],
                            W_x_layer,
                        )
                    elif args.procrustes_L:
                        L_V_T = procrustes_L(
                            W_old_V[old_start:old_end, :], W_new_V[new_start:new_end, :],
                            W_x_layer).T
                    elif args.correct_transform:
                        L_V_T = compute_correct_L(
                            W_old_V[old_start:old_end, :], W_new_V[new_start:new_end, :],
                            W_x_layer, reg=args.correct_transform_reg).T
                    else:
                        L_V_T = W_V_x.T
                    if not args.svd_transfer:
                        if args.use_nspt:
                            new_delta_W_V[new_start:new_end, :] = apply_xform_nspt(
                                old_delta_v, W_old_V[old_start:old_end, :], W_new_V[new_start:new_end, :],
                                L_V_T, W_x_layer,
                                energy=args.nspt_energy,
                                par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                                scale_calibrate=args.scale_calibrate,
                            )
                        else:
                            new_delta_W_V[new_start:new_end, :] = apply_xform_with_prolora(
                                old_delta_v, W_old_V[old_start:old_end, :], W_new_V[new_start:new_end, :],
                                L_V_T, W_x_layer,
                                args.use_prolora, args.prolora_mode,
                                args.prolora_energy, args.prolora_null_scale, args.scale_calibrate,
                                args.spectral_calibrate, args.polar_head,
                                no_left_transform=args.no_left_transform,
                                spectral_blend=layer_spectral_blend,
                            )

                    W_O_x = W_old_O[:, old_start:old_end].T @ W_x_layer @ W_new_O[:, new_start:new_end]
                    old_delta_o = delta_W_O[:, old_start:old_end]
                    if args.svd_transfer:
                        # O-proj: old_delta_o shape (old_hidden_slice, head_size)
                        # Transpose: "right SVs" become old left SVs ?????need mapping via W_x_layer
                        # W_x_layer: (old_hidden, new_hidden) maps old_hidden ?????new_hidden
                        new_delta_W_O[:, new_start:new_end] = apply_svd_transfer(
                            old_delta_o.T,
                            W_old_O[:, old_start:old_end].T, W_new_O[:, new_start:new_end].T,
                            W_x_layer,
                        ).T
                    elif args.procrustes_L:
                        W_old_O_head = W_old_O[:, old_start:old_end].T  # (head_size, old_hidden)
                        W_new_O_head = W_new_O[:, new_start:new_end].T  # (head_size, new_hidden)
                        L_O_T = procrustes_L(W_old_O_head, W_new_O_head, W_x_layer).T
                        L_O_left = polar_W_x_layer.T
                    elif args.correct_transform:
                        # O-proj: left space is output (hidden), right space is head dim
                        # For O: old is [hidden, head_size], new is [hidden, head_size]
                        # L is along hidden-dim, R is along head-dim
                        # correct_L for O: using W_old_O column heads as "W_old_head", W_new_O column heads as "W_new_head"
                        # W_x_layer maps old_hidden -> new_hidden; for O-proj cols: need W_x_layer.T for reverse
                        W_old_O_head = W_old_O[:, old_start:old_end].T  # (head_size, old_hidden)
                        W_new_O_head = W_new_O[:, new_start:new_end].T  # (head_size, new_hidden)
                        L_O = compute_correct_L(W_old_O_head, W_new_O_head, W_x_layer,
                                                reg=args.correct_transform_reg)
                        L_O_T = L_O.T   # right multiplier for O is head-dim
                        L_O_left = polar_W_x_layer.T  # left multiplier (hidden space)
                    else:
                        L_O_left = polar_W_x_layer.T
                        L_O_T = W_O_x
                    if not args.svd_transfer:
                        if args.use_nspt:
                            new_delta_W_O[:, new_start:new_end] = apply_xform_nspt(
                                old_delta_o, W_old_O[:, old_start:old_end], W_new_O[:, new_start:new_end],
                                L_O_left, L_O_T,
                                energy=args.nspt_energy,
                                par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                                scale_calibrate=args.scale_calibrate,
                            )
                        else:
                            new_delta_W_O[:, new_start:new_end] = apply_xform_with_prolora(
                                old_delta_o, W_old_O[:, old_start:old_end], W_new_O[:, new_start:new_end],
                                L_O_left, L_O_T,
                                args.use_prolora, args.prolora_mode,
                                args.prolora_energy, args.prolora_null_scale, args.scale_calibrate,
                                args.spectral_calibrate, False,  # polar already applied via polar_W_x_layer
                                no_left_transform=False,  # O-proj needs hidden-space transform
                                spectral_blend=layer_spectral_blend,
                            )
                else:
                    new_delta_W_Q = W_x.T @ delta_W_Q @ W_x
                    new_delta_W_K = W_x.T @ delta_W_K @ W_x
                    new_delta_W_V = W_x.T @ delta_W_V @ W_x
                    new_delta_W_O = W_x.T @ delta_W_O @ W_x

        # Fill unmatched target heads (heads with zero LoRA due to 24?????6 mismatch)
        if (args.fill_unmatched_heads
                and args.attention_method == "xtransform"
                and args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format"
                and args.old_model.split('/')[-2] == "MiniCPM-S-1B-sft-llama-format"):
            matched_set = set(int(x) for x in head_mapping)
            for b in range(new_num_attention_heads):
                if b in matched_set:
                    continue
                # Nearest unmatched source head by cosine similarity
                best_a = int(np.argmax(head_sim_matrix[:, b]))
                o_s = best_a * old_head_size
                o_e = (best_a + 1) * old_head_size
                n_s = b * new_head_size
                n_e = (b + 1) * new_head_size

                sc = args.spectral_calibrate
                sc_g = args.scale_calibrate

                # Q
                W_Q_x_f = W_old_Q[o_s:o_e, :] @ W_x_layer @ W_new_Q[n_s:n_e, :].T
                dq_f = delta_W_Q[o_s:o_e, :]
                if args.correct_transform:
                    L_Q_T_f = compute_correct_L(
                        W_old_Q[o_s:o_e, :], W_new_Q[n_s:n_e, :],
                        W_x_layer, reg=args.correct_transform_reg).T
                else:
                    L_Q_T_f = W_Q_x_f.T
                if args.use_nspt:
                    new_delta_W_Q[n_s:n_e, :] = apply_xform_nspt(
                        dq_f, W_old_Q[o_s:o_e, :], W_new_Q[n_s:n_e, :],
                        L_Q_T_f, W_x_layer, energy=args.nspt_energy,
                        par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                        scale_calibrate=sc_g)
                else:
                    new_delta_W_Q[n_s:n_e, :] = apply_xform_with_prolora(
                        dq_f, W_old_Q[o_s:o_e, :], W_new_Q[n_s:n_e, :],
                        L_Q_T_f, W_x_layer,
                        args.use_prolora, args.prolora_mode,
                        args.prolora_energy, args.prolora_null_scale, sc_g, sc, args.polar_head,
                        no_left_transform=args.no_left_transform,
                        spectral_blend=layer_spectral_blend)
                # K
                W_K_x_f = W_old_K[o_s:o_e, :] @ W_x_layer @ W_new_K[n_s:n_e, :].T
                dk_f = delta_W_K[o_s:o_e, :]
                if args.correct_transform:
                    L_K_T_f = compute_correct_L(
                        W_old_K[o_s:o_e, :], W_new_K[n_s:n_e, :],
                        W_x_layer, reg=args.correct_transform_reg).T
                else:
                    L_K_T_f = W_K_x_f.T
                if args.use_nspt:
                    new_delta_W_K[n_s:n_e, :] = apply_xform_nspt(
                        dk_f, W_old_K[o_s:o_e, :], W_new_K[n_s:n_e, :],
                        L_K_T_f, W_x_layer, energy=args.nspt_energy,
                        par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                        scale_calibrate=sc_g)
                else:
                    new_delta_W_K[n_s:n_e, :] = apply_xform_with_prolora(
                        dk_f, W_old_K[o_s:o_e, :], W_new_K[n_s:n_e, :],
                        L_K_T_f, W_x_layer,
                        args.use_prolora, args.prolora_mode,
                        args.prolora_energy, args.prolora_null_scale, sc_g, sc, args.polar_head,
                        no_left_transform=args.no_left_transform,
                        spectral_blend=layer_spectral_blend)
                # V
                W_V_x_f = W_old_V[o_s:o_e, :] @ W_x_layer @ W_new_V[n_s:n_e, :].T
                dv_f = delta_W_V[o_s:o_e, :]
                if args.correct_transform:
                    L_V_T_f = compute_correct_L(
                        W_old_V[o_s:o_e, :], W_new_V[n_s:n_e, :],
                        W_x_layer, reg=args.correct_transform_reg).T
                else:
                    L_V_T_f = W_V_x_f.T
                if args.use_nspt:
                    new_delta_W_V[n_s:n_e, :] = apply_xform_nspt(
                        dv_f, W_old_V[o_s:o_e, :], W_new_V[n_s:n_e, :],
                        L_V_T_f, W_x_layer, energy=args.nspt_energy,
                        par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                        scale_calibrate=sc_g)
                else:
                    new_delta_W_V[n_s:n_e, :] = apply_xform_with_prolora(
                        dv_f, W_old_V[o_s:o_e, :], W_new_V[n_s:n_e, :],
                        L_V_T_f, W_x_layer,
                        args.use_prolora, args.prolora_mode,
                        args.prolora_energy, args.prolora_null_scale, sc_g, sc, args.polar_head,
                        no_left_transform=args.no_left_transform,
                        spectral_blend=layer_spectral_blend)
                # O (transposed layout: [hidden, num_heads*head_size])
                W_O_x_f = W_old_O[:, o_s:o_e].T @ W_x_layer @ W_new_O[:, n_s:n_e]
                do_f = delta_W_O[:, o_s:o_e]
                if args.correct_transform:
                    W_old_O_head_f = W_old_O[:, o_s:o_e].T
                    W_new_O_head_f = W_new_O[:, n_s:n_e].T
                    L_O_f = compute_correct_L(W_old_O_head_f, W_new_O_head_f, W_x_layer,
                                              reg=args.correct_transform_reg)
                    L_O_left_f = polar_W_x_layer.T
                    L_O_T_f = L_O_f.T
                else:
                    L_O_left_f = polar_W_x_layer.T
                    L_O_T_f = W_O_x_f
                if args.use_nspt:
                    new_delta_W_O[:, n_s:n_e] = apply_xform_nspt(
                        do_f, W_old_O[:, o_s:o_e], W_new_O[:, n_s:n_e],
                        L_O_left_f, L_O_T_f, energy=args.nspt_energy,
                        par_weight=args.nspt_par_weight, null_weight=args.nspt_null_weight,
                        scale_calibrate=sc_g)
                else:
                    new_delta_W_O[:, n_s:n_e] = apply_xform_with_prolora(
                        do_f, W_old_O[:, o_s:o_e], W_new_O[:, n_s:n_e],
                        L_O_left_f, L_O_T_f,
                        args.use_prolora, args.prolora_mode,
                        args.prolora_energy, args.prolora_null_scale, sc_g, sc, False,  # polar already applied via polar_W_x_layer
                        no_left_transform=False,  # O-proj needs hidden-space transform
                        spectral_blend=layer_spectral_blend)

        if args.new_model.split('/')[-2] == "pythia-1.4b":
            B, A, _ = low_rank_decompose(new_delta_W_QKV, old_rank)
            
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.query_key_value.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, old_rank)
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.dense.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.gpt_neox.layers.{layer_mapping[i]}.attention.dense.lora_A.weight"] = A

        elif args.new_model.split('/')[-2] == "bloomz-1b1":
            B, A, _ = low_rank_decompose(new_delta_W_QKV, old_rank)
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.query_key_value.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.query_key_value.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, old_rank)
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.dense.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.self_attention.dense.lora_A.weight"] = A

            if "dense_h_to_4h" in target_modules:
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.lora_B.weight"] = W_new_U @ torch.linalg.pinv(W_x) @ torch.linalg.pinv(W_old_U) @ old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_h_to_4h.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_h_to_4h.lora_A.weight"] = old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_h_to_4h.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ W_x
            if "dense_4h_to_h" in target_modules:
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.lora_B.weight"] = torch.linalg.pinv(W_x) @ old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_4h_to_h.lora_B.weight"].to(device="cpu", dtype=torch.float32) 
                new_lora_weights[f"base_model.model.transformer.h.{layer_mapping[i]}.mlp.dense_4h_to_h.lora_A.weight"] = old_lora_weights[f"base_model.model.transformer.h.{i}.mlp.dense_4h_to_h.lora_A.weight"].to(device="cpu", dtype=torch.float32) @ torch.linalg.pinv(W_old_D) @ W_x @ W_new_D

        elif args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format":
            
            if args.use_prolora:
                print(f"Applying ProLoRA source decomposition and target recomposition for Layer {i} (MiniCPM-2B)")

            B, A, E = low_rank_decompose(new_delta_W_Q, save_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_K, save_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_V, save_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"] = A

            B, A, E = low_rank_decompose(new_delta_W_O, save_rank, "adalora" in args.old_lora_path)
            if "adalora" in args.old_lora_path:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_E"] = E
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A"] = A
            else:
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight"] = B
                new_lora_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight"] = A

            # FFN transfer for MiniCPM: use apply_svd_ffn_transfer (handles non-integer
            # intermediate ratio 3840->5760=1.5x via SVD+tiling) with per-layer W_x.
            # Fixes: (1) gate_proj was completely missing; (2) old code used global W_x
            # instead of W_x_layer; (3) pinv chain was numerically unstable.
            if "up_proj" in target_modules:
                old_delta_W_up = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_up = apply_svd_ffn_transfer(old_delta_W_up, W_old_U, W_new_U, W_x_layer, side='up', rank=save_rank)
                if args.spectral_calibrate:
                    new_delta_W_up = spectral_calibrate_delta(new_delta_W_up, old_delta_W_up, W_new_U, W_old_U, blend=layer_spectral_blend)
                elif args.scale_calibrate:
                    old_rel = torch.norm(old_delta_W_up, 'fro') / (torch.norm(W_old_U, 'fro') + 1e-8)
                    new_rel = torch.norm(new_delta_W_up, 'fro') / (torch.norm(W_new_U, 'fro') + 1e-8)
                    if new_rel > 1e-8:
                        new_delta_W_up = new_delta_W_up * (old_rel / new_rel)
                B_up, A_up, _ = low_rank_decompose(new_delta_W_up, old_rank)
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"] = B_up
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"] = A_up

            if "gate_proj" in target_modules:
                old_delta_W_gate = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.gate_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.gate_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_gate = apply_svd_ffn_transfer(old_delta_W_gate, W_old_G, W_new_G, W_x_layer, side='gate', rank=save_rank)
                if args.spectral_calibrate:
                    new_delta_W_gate = spectral_calibrate_delta(new_delta_W_gate, old_delta_W_gate, W_new_G, W_old_G, blend=layer_spectral_blend)
                elif args.scale_calibrate:
                    old_rel = torch.norm(old_delta_W_gate, 'fro') / (torch.norm(W_old_G, 'fro') + 1e-8)
                    new_rel = torch.norm(new_delta_W_gate, 'fro') / (torch.norm(W_new_G, 'fro') + 1e-8)
                    if new_rel > 1e-8:
                        new_delta_W_gate = new_delta_W_gate * (old_rel / new_rel)
                B_gate, A_gate, _ = low_rank_decompose(new_delta_W_gate, old_rank)
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight"] = B_gate
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight"] = A_gate

            if "down_proj" in target_modules:
                old_delta_W_down = old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_down = apply_svd_ffn_transfer(old_delta_W_down, W_old_D, W_new_D, W_x_layer, side='down', rank=save_rank)
                if args.spectral_calibrate:
                    new_delta_W_down = spectral_calibrate_delta(new_delta_W_down, old_delta_W_down, W_new_D, W_old_D, blend=layer_spectral_blend)
                elif args.scale_calibrate:
                    old_rel = torch.norm(old_delta_W_down, 'fro') / (torch.norm(W_old_D, 'fro') + 1e-8)
                    new_rel = torch.norm(new_delta_W_down, 'fro') / (torch.norm(W_new_D, 'fro') + 1e-8)
                    if new_rel > 1e-8:
                        new_delta_W_down = new_delta_W_down * (old_rel / new_rel)
                B_down, A_down, _ = low_rank_decompose(new_delta_W_down, old_rank)
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] = B_down
                new_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] = A_down
        
        elif args.new_model.split('/')[-2] == "Qwen2.5-3B" or  "Meta-Llama-3-8B" in args.new_model.split('/')[-2] or "qwen3" in args.new_model.lower():
            B, A, _ = low_rank_decompose(new_delta_W_Q, save_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.q_proj.lora_A.weight"] = A

            new_delta_W_K = new_delta_W_K.view(new_num_key_value_heads, new_num_rep, new_head_size, new_hidden_size).mean(dim=1) 
            B, A, _ = low_rank_decompose(new_delta_W_K.view(new_num_key_value_heads*new_head_size, new_hidden_size), save_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.k_proj.lora_A.weight"] = A

            new_delta_W_V = new_delta_W_V.view(new_num_key_value_heads, new_num_rep, new_head_size, new_hidden_size).mean(dim=1) 
            B, A, _ = low_rank_decompose(new_delta_W_V.view(new_num_key_value_heads*new_head_size, new_hidden_size), save_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.v_proj.lora_A.weight"] = A

            B, A, _ = low_rank_decompose(new_delta_W_O, save_rank)
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_B.weight"] = B
            new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.self_attn.o_proj.lora_A.weight"] = A

            if "up_proj" in target_modules:    
                old_delta_W_up = old_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                W_new_U = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
                W_old_U = old_model_dic[f"model.layers.{i}.mlp.up_proj.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_up = apply_svd_ffn_transfer(old_delta_W_up, W_old_U, W_new_U, W_x, side='up', rank=save_rank)
                B_up, A_up, _ = low_rank_decompose(new_delta_W_up, save_rank)
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_B.weight"] = B_up 
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.up_proj.lora_A.weight"] = A_up
            if "down_proj" in target_modules:
                old_delta_W_down = old_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                W_new_D = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)
                W_old_D = old_model_dic[f"model.layers.{i}.mlp.down_proj.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_down = apply_svd_ffn_transfer(old_delta_W_down, W_old_D, W_new_D, W_x, side='down', rank=save_rank)
                B_down, A_down, _ = low_rank_decompose(new_delta_W_down, save_rank)
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_B.weight"] = B_down
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.down_proj.lora_A.weight"] = A_down
            if "gate_proj" in target_modules:
                old_delta_W_gate = old_lora_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight"].to(device="cpu", dtype=torch.float32) @ old_lora_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight"].to(device="cpu", dtype=torch.float32)
                W_new_G = new_model_dic[f"model.layers.{layer_mapping[i]}.mlp.gate_proj.weight"].to(device="cpu", dtype=torch.float32)
                W_old_G = old_model_dic[f"model.layers.{i}.mlp.gate_proj.weight"].to(device="cpu", dtype=torch.float32)
                new_delta_W_gate = apply_svd_ffn_transfer(old_delta_W_gate, W_old_G, W_new_G, W_x, side='gate', rank=save_rank)
                B_gate, A_gate, _ = low_rank_decompose(new_delta_W_gate, save_rank)
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.gate_proj.lora_B.weight"] = B_gate
                new_lora_weights[f"base_model.model.model.layers.{layer_mapping[i]}.mlp.gate_proj.lora_A.weight"] = A_gate
        


    # ---- Fill missing target layers (nearest-neighbor copy) ----
    # For qwen3_1_7B -> qwen3_8B, layer_mapping covers only 28/36 target layers.
    # Missing target layers: {4,8,13,17,22,26,31,35}.
    # We copy LoRA weights from the nearest covered target layer.
    if args.fill_missing_layers:
        mapped_targets = set(layer_mapping)
        all_targets = set(range(new_hidden_layers))
        missing_targets = sorted(all_targets - mapped_targets)
        if missing_targets:
            print(f"Filling {len(missing_targets)} missing target layers: {missing_targets}")
            attn_projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            ffn_projs  = ['gate_proj', 'up_proj', 'down_proj']
            for miss_t in missing_targets:
                nearest = min(mapped_targets, key=lambda t: abs(t - miss_t))
                for proj in attn_projs:
                    for ab in ['lora_A', 'lora_B']:
                        src_key = f"base_model.model.model.layers.{nearest}.self_attn.{proj}.{ab}.weight"
                        tgt_key = f"base_model.model.model.layers.{miss_t}.self_attn.{proj}.{ab}.weight"
                        if src_key in new_lora_weights:
                            new_lora_weights[tgt_key] = new_lora_weights[src_key].clone()
                for proj in ffn_projs:
                    for ab in ['lora_A', 'lora_B']:
                        src_key = f"base_model.model.model.layers.{nearest}.mlp.{proj}.{ab}.weight"
                        tgt_key = f"base_model.model.model.layers.{miss_t}.mlp.{proj}.{ab}.weight"
                        if src_key in new_lora_weights:
                            new_lora_weights[tgt_key] = new_lora_weights[src_key].clone()
            print("Missing layer fill complete.")

    suffix = ""
    if args.use_nspt:
        suffix += f"_nspt_e{args.nspt_energy}_p{args.nspt_par_weight}_n{args.nspt_null_weight}"
    elif args.use_prolora:
        suffix += f"_prolora_{args.prolora_mode}_e{args.prolora_energy}_ns{args.prolora_null_scale}"
    if args.spectral_calibrate:
        suffix += "_spcal"
    if args.scale_calibrate:
        suffix += "_sc"
    if args.fill_unmatched_heads:
        suffix += "_fill"
    if args.fill_missing_layers:
        suffix += "_filllayers"
    if args.old_model.split('/')[-2] == "qwen3_1_7B" and args.new_model.split('/')[-2] == "qwen3_8B":
        if args.qwen_layer_mapping_mode != 'fixed':
            suffix += f"_qmap{args.qwen_layer_mapping_mode}"
    if args.correct_transform:
        suffix += "_ctxf"
    if args.per_layer_wx:
        suffix += "_plwx"
    if args.act_align_path:
        suffix += "_actalign"
    if args.new_rank > 0 and args.new_rank != old_rank:
        suffix += f"_r{args.new_rank}"
    if args.finetuned_head_sim:
        suffix += "_ftsim"
    if args.polar_head:
        suffix += "_polar"
    if args.layer_method != "xtransform":
        suffix += f"_lm{args.layer_method}"
    if args.norm_correct:
        suffix += "_nc"
    if args.no_left_transform:
        suffix += "_nlt"
    if args.svd_transfer:
        suffix += "_svdt"
    if args.procrustes_L:
        suffix += "_procL"
    if args.lora_alpha_scale != 1.0:
        scale_str = f"{args.lora_alpha_scale:.2f}".replace('.', 'p')
        suffix += f"_as{scale_str}"
    if args.spectral_calibrate and args.spectral_blend != 1.0:
        blend_str = f"{args.spectral_blend:.2f}".replace('.', 'p')
        suffix += f"_sb{blend_str}"
    if args.spectral_blend_cka:
        suffix += "_ckabl"
    if not suffix:
        suffix = "_base"
    new_lora_path = "./trained_models/xTransform/" + args.new_model.split('/')[-2] + f"_{args.old_lora_path.split('/')[-2]}{suffix}/"
        
    if not os.path.exists(new_lora_path):
        os.makedirs(new_lora_path)

    # Post-hoc per-layer norm correction: rescale each migrated LoRA delta so that
    # ||delta_new||_F / ||W_new||_F == ||delta_old||_F / ||W_old||_F.
    # Only applied for MiniCPM-2B (the model pair we've analyzed the over-scaling for).
    if args.norm_correct and args.new_model.split('/')[-2] == "MiniCPM-2B-sft-fp32-llama-format":
        print("Applying post-hoc norm correction...")
        new_model_weights = torch.load(
            args.new_model + "pytorch_model.bin", map_location="cpu"
        )
        old_model_weights = torch.load(
            args.old_model + "pytorch_model.bin", map_location="cpu"
        )
        modules_to_correct = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        for new_layer_idx in range(len(layer_mapping)):
            old_layer_idx = layer_mapping[new_layer_idx]
            for mod in modules_to_correct:
                b_key = f"base_model.model.model.layers.{new_layer_idx}.self_attn.{mod}.lora_B.weight"
                a_key = f"base_model.model.model.layers.{new_layer_idx}.self_attn.{mod}.lora_A.weight"
                b_key_old = f"base_model.model.model.layers.{old_layer_idx}.self_attn.{mod}.lora_B.weight"
                a_key_old = f"base_model.model.model.layers.{old_layer_idx}.self_attn.{mod}.lora_A.weight"
                if b_key not in new_lora_weights or a_key not in new_lora_weights:
                    continue
                if b_key_old not in old_lora_weights or a_key_old not in old_lora_weights:
                    continue
                # Reconstruct deltas ?????use CPU for all norm computations
                B_new = new_lora_weights[b_key].float().cpu()
                A_new = new_lora_weights[a_key].float().cpu()
                B_old = old_lora_weights[b_key_old].float().cpu()
                A_old = old_lora_weights[a_key_old].float().cpu()
                delta_new = B_new @ A_new
                delta_old = B_old @ A_old
                # Look up W_new and W_old base weights
                w_new_key = f"model.layers.{new_layer_idx}.self_attn.{mod}.weight"
                w_old_key = f"model.layers.{old_layer_idx}.self_attn.{mod}.weight"
                if w_new_key not in new_model_weights or w_old_key not in old_model_weights:
                    continue
                W_new_base = new_model_weights[w_new_key].float()
                W_old_base = old_model_weights[w_old_key].float()
                # Compute target relative norm
                rel_old = delta_old.norm('fro') / (W_old_base.norm('fro') + 1e-8)
                rel_new = delta_new.norm('fro') / (W_new_base.norm('fro') + 1e-8)
                if rel_new > 1e-8:
                    scale = rel_old / rel_new
                    # Only rescale B to preserve A's structure
                    # Keep same device and dtype as the stored tensor
                    orig = new_lora_weights[b_key]
                    new_lora_weights[b_key] = (B_new * scale).to(device=orig.device, dtype=orig.dtype)
        print("Norm correction applied.")

    peft_config = LoraConfig(
        r=save_rank,
        lora_alpha=save_rank * args.lora_alpha_scale,   # effective scale = alpha/r = lora_alpha_scale
        lora_dropout=data['lora_dropout'],
        bias="none",
        target_modules=data['target_modules'],
        task_type="CAUSAL_LM",
    )
    peft_config.base_model_name_or_path = args.new_model
    peft_config.inference_mode = True
    peft_config.save_pretrained(new_lora_path, auto_mapping_dict=None)
    safe_save_file(
        new_lora_weights,
        os.path.join(new_lora_path, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


