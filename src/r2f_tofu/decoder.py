from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .module_keys import DEFAULT_TARGET_MODULES
from .utils import sine_cosine_depth


class GradientDecoder(nn.Module):
    def __init__(
        self,
        rank: int,
        num_modules: int = len(DEFAULT_TARGET_MODULES),
        num_layers: int = 1,
        module_embedding_dim: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.05,
        use_projection_residual: bool = True,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.num_modules = int(num_modules)
        self.num_layers = int(num_layers)
        self.module_embedding_dim = int(module_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_projection_residual = bool(use_projection_residual)
        self.module_embedding = nn.Embedding(num_modules, module_embedding_dim)
        self.projection_feature_dim = 4
        input_dim = self.rank * 8 + 8 + 3 + module_embedding_dim + self.projection_feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 128)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 128), 1),
        )

    def build_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        a = batch["A_col"].float()
        b = batch["B_row"].float()
        da = batch["dA_col"].float()
        db = batch["dB_row"].float()
        if a.shape[-1] != self.rank:
            raise ValueError(f"Expected rank {self.rank}, got {a.shape[-1]}")

        interactions = [
            a * db,
            b * da,
            a * b,
            da * db,
        ]
        stats = torch.stack(
            [
                rms(a),
                rms(b),
                rms(da),
                rms(db),
                torch.sum(a * da, dim=-1),
                torch.sum(b * db, dim=-1),
                torch.sum(a * db, dim=-1),
                torch.sum(b * da, dim=-1),
            ],
            dim=-1,
        )
        layer_idx = batch["layer_idx"].long()
        num_layers_tensor = batch.get("num_layers")
        if num_layers_tensor is not None and torch.is_tensor(num_layers_tensor):
            num_layers = int(num_layers_tensor.max().item())
        else:
            num_layers = self.num_layers
        rel, depth_sin, depth_cos = sine_cosine_depth(layer_idx, num_layers)
        depth = torch.stack([rel, depth_sin, depth_cos], dim=-1).to(a.device)
        module_emb = self.module_embedding(batch["module_id"].long().to(a.device))
        projection = torch.stack(
            [
                batch["pinv_dB_norm"].float(),
                batch["pinv_dA_norm"].float(),
                batch["pinv_mean_norm"].float(),
                batch["pinv_diff_norm"].float(),
            ],
            dim=-1,
        ).to(a.device)
        return torch.cat([a, b, da, db, *interactions, stats, depth, module_emb, projection], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        residual = self.net(self.build_features(batch)).squeeze(-1)
        if not self.use_projection_residual:
            return residual
        return batch["pinv_mean_norm"].float().to(residual.device) + residual

    def config_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "num_modules": self.num_modules,
            "num_layers": self.num_layers,
            "module_embedding_dim": self.module_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "feature_version": "projection_v2",
            "projection_feature_dim": self.projection_feature_dim,
            "use_projection_residual": self.use_projection_residual,
        }


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.float().pow(2), dim=-1) + 1e-12)


def decoder_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    sign_loss_weight: float = 0.01,
    target_clip: float | None = None,
    sign_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_norm = target_norm.float()
    pred_norm = pred_norm.float()
    if target_clip is not None and float(target_clip) > 0:
        target_for_loss = target_norm.clamp(-float(target_clip), float(target_clip))
    else:
        target_for_loss = target_norm
    huber = nn.functional.huber_loss(pred_norm, target_for_loss, delta=1.0)
    strong_mask = target_norm.abs() >= float(sign_threshold)
    if strong_mask.any():
        sign_loss = nn.functional.softplus(-pred_norm[strong_mask] * target_for_loss[strong_mask]).mean()
    else:
        sign_loss = pred_norm.sum() * 0.0
    loss = huber + sign_loss_weight * sign_loss
    with torch.no_grad():
        sign_acc = (torch.sign(pred_norm) == torch.sign(target_norm)).float().mean()
        if strong_mask.any():
            sign_acc_strong = (
                torch.sign(pred_norm[strong_mask]) == torch.sign(target_norm[strong_mask])
            ).float().mean()
        else:
            sign_acc_strong = torch.tensor(0.0, device=pred_norm.device)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "huber": float(huber.detach().cpu()),
        "sign_loss": float(sign_loss.detach().cpu()),
        "sign_acc": float(sign_acc.detach().cpu()),
        "sign_acc_strong": float(sign_acc_strong.detach().cpu()),
        "strong_frac": float(strong_mask.float().mean().detach().cpu()),
    }
