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

    def delta(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, received {feature.shape[-1]}")
        return self.net(feature)


class GatedFeatureRestorer(nn.Module):
    """Residual feature restoration with one learned gate per visual token."""

    def __init__(self, hidden_size: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.restorer = ResidualFeatureRestorer(hidden_size, bottleneck_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, feature: torch.Tensor, *, return_gate: bool = False):
        delta = self.restorer.delta(feature)
        gate = torch.sigmoid(self.gate(feature))
        restored = feature + gate * delta
        return (restored, gate) if return_gate else restored


class QuestionGatedFeatureRestorer(nn.Module):
    """Per-token gate conditioned on a frozen question embedding."""

    def __init__(self, hidden_size: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.restorer = ResidualFeatureRestorer(hidden_size, bottleneck_dim)
        self.visual = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck_dim))
        self.question = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck_dim))
        self.gate = nn.Sequential(nn.GELU(), nn.Linear(bottleneck_dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, feature: torch.Tensor, question: torch.Tensor, *, return_gate: bool = False):
        squeezed = feature.ndim == 2
        if squeezed:
            feature = feature.unsqueeze(0)
        if question.ndim == 2:
            question = question[:, None, :]
        if question.ndim != 3 or question.shape[0] != feature.shape[0] or question.shape[-1] != self.hidden_size:
            raise ValueError(f"Question feature must be [B,H] or [B,1,H], got {question.shape}")
        delta = self.restorer.delta(feature)
        logits = self.gate(self.visual(feature) + self.question(question))
        gate = torch.sigmoid(logits)
        restored = feature + gate * delta
        if squeezed:
            restored, gate = restored.squeeze(0), gate.squeeze(0)
        return (restored, gate) if return_gate else restored


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

