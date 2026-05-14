"""Auxiliary per-modality prediction heads for Self-MM-style multitask training.

Three small MLPs (one per modality) that map each modality's pooled
representation ``z_m`` to a task prediction. The X-MoFE forward path
optionally runs them in parallel with the main fusion head; the
composite loss adds an auxiliary-unimodal-task term that flows gradient
back through each modality's projection + pooling separately.

This is the structural piece of Lever 4. The composite loss
(``src/losses/xmofe_loss.py``) adds the auxiliary-unimodal task term
behind weight ``epsilon``: when ``unimodal_labels`` is present in the
batch (CH-SIMS), each modality is supervised by its own column; when
absent (MOSEI/MELD), the multimodal target ``y_M`` is reused as the
Self-MM ``y_m^s = y_M`` ablation baseline. Centroid-based
momentum-updated pseudo-labels (Self-MM Algorithm 1) can be layered on
later via a trainer-level hook that overwrites ``unimodal_labels`` per
batch; the loss term itself stays the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.prediction_heads import PredictionHead


class UnimodalHeads(nn.Module):
    """Three task-specific heads, one per modality (text/audio/visual).

    The output of ``forward`` is a dict keyed by modality so downstream
    losses and reporting can address each modality independently.

    Args:
        shared_dim: per-modality pooled representation dim ``d`` (matches
            the X-MoFE shared dim).
        task: ``"regression"`` or ``"classification"``; same as the main head.
        num_classes: prediction-head output count (ignored for regression).
        dropout: dropout rate inside each head's MLP.
    """

    MODALITIES = ("text", "audio", "visual")

    def __init__(
        self,
        shared_dim: int,
        task: str,
        num_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.heads = nn.ModuleDict({
            m: PredictionHead(shared_dim, task=task, num_classes=num_classes, dropout=dropout)
            for m in self.MODALITIES
        })

    def forward(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "text": self.heads["text"](z_text),
            "audio": self.heads["audio"](z_audio),
            "visual": self.heads["visual"](z_visual),
        }
