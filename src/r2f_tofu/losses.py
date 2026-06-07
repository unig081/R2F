from __future__ import annotations

import torch


def ce_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**batch)
    return outputs.loss


def ga_gd_loss(
    model: torch.nn.Module,
    batch: dict[str, dict[str, torch.Tensor]],
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    forget_ce = ce_loss(model, batch["forget"])
    retain_ce = ce_loss(model, batch["retain"])
    loss = gamma * (-forget_ce) + alpha * retain_ce
    return loss, {
        "loss": float(loss.detach().cpu()),
        "forget_ce": float(forget_ce.detach().cpu()),
        "retain_ce": float(retain_ce.detach().cpu()),
    }
