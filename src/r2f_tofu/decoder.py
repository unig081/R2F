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
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.num_modules = int(num_modules)
        self.num_layers = int(num_layers)
        self.module_embedding_dim = int(module_embedding_dim)
        self.module_embedding = nn.Embedding(num_modules, module_embedding_dim)
        input_dim = self.rank * 8 + 8 + 3 + module_embedding_dim
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
        return torch.cat([a, b, da, db, *interactions, stats, depth, module_emb], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(self.build_features(batch)).squeeze(-1)

    def config_dict(self) -> dict[str, Any]:
        first_linear = next(m for m in self.net if isinstance(m, nn.Linear))
        return {
            "rank": self.rank,
            "num_modules": self.num_modules,
            "num_layers": self.num_layers,
            "module_embedding_dim": self.module_embedding_dim,
            "hidden_dim": first_linear.out_features,
        }


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.float().pow(2), dim=-1) + 1e-12)


def decoder_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    sign_loss_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_norm = target_norm.float()
    huber = nn.functional.huber_loss(pred_norm.float(), target_norm, delta=1.0)
    sign_loss = nn.functional.softplus(-pred_norm.float() * target_norm).mean()
    loss = huber + sign_loss_weight * sign_loss
    with torch.no_grad():
        sign_acc = (torch.sign(pred_norm) == torch.sign(target_norm)).float().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "huber": float(huber.detach().cpu()),
        "sign_loss": float(sign_loss.detach().cpu()),
        "sign_acc": float(sign_acc.detach().cpu()),
    }
