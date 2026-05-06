"""Late-fusion baseline.

Each modality has its own projection + pool + prediction head. Per-modality
predictions are combined via a learnable softmax-weighted sum. Trained
jointly (single forward pass over the batch); the legacy preprint used
the same approach.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention_pooling import AttentionPool
from src.models.baselines._common import uniform_interactions
from src.models.explanation_heads import XMoFEOutput
from src.models.prediction_heads import PredictionHead
from src.models.projections import ModalityProjection


class _UnimodalBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        dropout: float,
        task: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.proj = ModalityProjection(input_dim, shared_dim, dropout)
        self.pool = AttentionPool(shared_dim)
        self.head = PredictionHead(shared_dim, task=task, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.proj(x)
        z, alpha = self.pool(Z, length)
        return self.head(z), z, alpha


class LateFusionModel(nn.Module):
    """``ŷ = Σ_m softmax(w)_m · ŷ_m`` over per-modality predictions."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        shared_dim: int = 256,
        dropout: float = 0.2,
        task: str = "regression",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.task = task
        self.shared_dim = shared_dim

        self.text_branch = _UnimodalBranch(text_dim, shared_dim, dropout, task, num_classes)
        self.audio_branch = _UnimodalBranch(audio_dim, shared_dim, dropout, task, num_classes)
        self.visual_branch = _UnimodalBranch(visual_dim, shared_dim, dropout, task, num_classes)

        # Learned scalar weights — softmax to keep them as a valid distribution.
        self.fusion_logits = nn.Parameter(torch.zeros(3))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        task: str,
        num_classes: int = 1,
    ) -> "LateFusionModel":
        return cls(
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            dropout=config.get("dropout", 0.2),
            task=task, num_classes=num_classes,
        )

    def _combine(self, preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.fusion_logits, dim=-1)         # (3,)
        # Both regression (B,) and classification (B, C) work via broadcasting.
        stacked = torch.stack(preds, dim=0)                     # (3, B[, C])
        # weights: (3,) → (3, 1[, 1])
        for _ in range(stacked.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (weights * stacked).sum(dim=0)                   # (B[, C])

    def forward(
        self,
        text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor,
        text_length: torch.Tensor, audio_length: torch.Tensor, visual_length: torch.Tensor,
        quality: torch.Tensor | None = None,  # noqa: ARG002
        return_intermediates: bool = True,
    ) -> XMoFEOutput:
        pred_t, z_t, alpha_t = self.text_branch(text, text_length)
        pred_a, z_a, alpha_a = self.audio_branch(audio, audio_length)
        pred_v, z_v, alpha_v = self.visual_branch(visual, visual_length)

        prediction = self._combine((pred_t, pred_a, pred_v))

        # Reliability output here is *not* a placeholder — late-fusion's
        # learned softmax weights are exactly what each modality contributes
        # to the final prediction. Useful for ablation tables but the X-MoFE
        # paper should not present these as faithful explanations.
        weights = F.softmax(self.fusion_logits, dim=-1).unsqueeze(0).expand(text.size(0), -1)

        return XMoFEOutput(
            prediction=prediction,
            reliability=weights,
            interactions=uniform_interactions(text.size(0), num_interactions=1, device=text.device),
            temporal_attention={"text": alpha_t, "audio": alpha_a, "visual": alpha_v},
            interaction_names=("late_weighted_sum",),
            pooled_modalities=(
                {"text": z_t, "audio": z_a, "visual": z_v} if return_intermediates else None
            ),
        )
