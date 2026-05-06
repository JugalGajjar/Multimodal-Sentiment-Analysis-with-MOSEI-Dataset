"""Unimodal baseline — single-modality projection + pool + prediction head.

One class, ``modality`` argument selects which input to consume. The other
modalities are still passed in by the dataloader; we just ignore them.
Reliability output one-hot encodes the active modality so downstream
ablation tables read cleanly.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPool
from src.models.baselines._common import uniform_interactions, zeros_attention
from src.models.explanation_heads import XMoFEOutput
from src.models.prediction_heads import PredictionHead
from src.models.projections import ModalityProjection

VALID_MODALITIES = ("text", "audio", "visual")


class UnimodalModel(nn.Module):
    """``ŷ = Head(AttnPool(Linear_m(x_m)))`` for a single chosen modality ``m``."""

    def __init__(
        self,
        modality: str,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        shared_dim: int = 256,
        dropout: float = 0.2,
        task: str = "regression",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if modality not in VALID_MODALITIES:
            raise ValueError(f"modality must be one of {VALID_MODALITIES}; got {modality!r}")
        self.modality = modality
        self.task = task

        input_dim = {"text": text_dim, "audio": audio_dim, "visual": visual_dim}[modality]
        self.proj = ModalityProjection(input_dim, shared_dim, dropout)
        self.pool = AttentionPool(shared_dim)
        self.head = PredictionHead(shared_dim, task=task, num_classes=num_classes, dropout=dropout)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        task: str,
        num_classes: int = 1,
        modality: str = "text",
    ) -> "UnimodalModel":
        return cls(
            modality=modality,
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            dropout=config.get("dropout", 0.2),
            task=task, num_classes=num_classes,
        )

    def _select(
        self,
        text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor,
        text_length: torch.Tensor, audio_length: torch.Tensor, visual_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return {
            "text": (text, text_length),
            "audio": (audio, audio_length),
            "visual": (visual, visual_length),
        }[self.modality]

    def forward(
        self,
        text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor,
        text_length: torch.Tensor, audio_length: torch.Tensor, visual_length: torch.Tensor,
        quality: torch.Tensor | None = None,  # noqa: ARG002
        return_intermediates: bool = True,
    ) -> XMoFEOutput:
        x, lengths = self._select(text, audio, visual, text_length, audio_length, visual_length)
        Z = self.proj(x)
        z, alpha = self.pool(Z, lengths)
        prediction = self.head(z)

        b = text.size(0)
        device = text.device
        # One-hot reliability — text=0, audio=1, visual=2.
        modality_idx = VALID_MODALITIES.index(self.modality)
        rel = torch.zeros(b, 3, device=device)
        rel[:, modality_idx] = 1.0

        # Real attention only for the chosen modality; zeros for the others.
        temporal = {
            "text": zeros_attention(b, text.size(1), device=device),
            "audio": zeros_attention(b, audio.size(1), device=device),
            "visual": zeros_attention(b, visual.size(1), device=device),
        }
        temporal[self.modality] = alpha

        return XMoFEOutput(
            prediction=prediction,
            reliability=rel,
            interactions=uniform_interactions(b, num_interactions=1, device=device),
            temporal_attention=temporal,
            interaction_names=(f"unimodal_{self.modality}",),
            pooled_modalities={self.modality: z} if return_intermediates else None,
            fused=z if return_intermediates else None,
        )
