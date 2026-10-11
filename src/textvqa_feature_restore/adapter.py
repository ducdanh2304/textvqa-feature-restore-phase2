from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualFeatureRestorer(nn.Module):
    def __init__(self, hidden_size: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bottleneck_dim, self.hidden_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, received {feature.shape[-1]}")
        return feature + self.net(feature)


def masked_feature_loss(
    restored: torch.Tensor,
    clean: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if restored.shape != clean.shape or restored.ndim != 3:
        raise ValueError(f"Feature shapes must be equal [B,T,H], got {restored.shape} and {clean.shape}")
    if valid_tokens.shape != restored.shape[:2] or valid_tokens.dtype != torch.bool:
        raise ValueError("valid_tokens must be a boolean [B,T] mask")
    if not valid_tokens.any():
        raise ValueError("A batch cannot contain zero valid tokens")
    mask = valid_tokens.unsqueeze(-1)
    squared = (restored.float() - clean.float()).pow(2)
    mse = (squared * mask).sum() / (mask.sum() * restored.shape[-1])
    cosine_per_token = F.cosine_similarity(restored.float(), clean.float(), dim=-1)
    cosine = 1.0 - cosine_per_token.masked_select(valid_tokens).mean()
    total = mse + 0.1 * cosine
    return total, mse, cosine


def adapter_config() -> dict[str, object]:
    return {
        "class_name": "ResidualFeatureRestorer",
        "hidden_size": 2048,
        "bottleneck_dim": 256,
        "dropout": 0.0,
        "residual_addition": True,
        "final_linear_zero_initialized": True,
        "dtype": "float32",
    }

