"""Typed container for X-MoFE outputs.

Bundles the prediction with the three explanation outputs (modality,
temporal, interaction) plus the intermediate representations downstream
analyses might want. Trainers use ``output.prediction`` for the loss;
explainability eval scripts pull the explanation fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass
class XMoFEOutput:
    """Outputs from a single ``XMoFE.forward`` call."""

    prediction: torch.Tensor
    """``(B,)`` for regression or ``(B, C)`` logits for classification."""

    reliability: torch.Tensor
    """``(B, 3)`` softmax over [text, audio, visual] — the modality-level explanation."""

    interactions: torch.Tensor
    """``(B, K)`` softmax over interaction blocks; K = 3 (no tri-modal) or 4."""

    temporal_attention: Mapping[str, torch.Tensor]
    """Per-modality attention pooling weights — the temporal explanation.

    Keys are ``"text"``, ``"audio"``, ``"visual"``; each value is ``(B, L_m)``
    where ``L_m`` is the cached max length for that modality.
    """

    interaction_names: tuple[str, ...] = field(default_factory=tuple)
    """Names of the K interaction columns (e.g. ``("text_audio", "text_visual", "audio_visual", "trimodal")``)."""

    pooled_modalities: Mapping[str, torch.Tensor] | None = None
    """Optional ``(B, D)`` pooled embedding per modality, useful for analysis/probing."""

    interaction_vectors: Mapping[str, torch.Tensor] | None = None
    """Optional ``(B, D)`` per-interaction pooled embedding, parallel to ``interactions``."""

    fused: torch.Tensor | None = None
    """Optional ``(B, D)`` fused representation passed to the prediction head."""

    unimodal_predictions: Mapping[str, torch.Tensor] | None = None
    """Optional per-modality task predictions from Self-MM-style auxiliary heads.

    Keys are ``"text"``, ``"audio"``, ``"visual"``; each value has the same
    shape as ``prediction``. Populated only when the model was built with
    ``use_unimodal_heads=True`` (Lever-4); ``None`` otherwise.
    """

    def detach(self) -> "XMoFEOutput":
        """Return a copy with every tensor detached and on the same device."""
        return XMoFEOutput(
            prediction=self.prediction.detach(),
            reliability=self.reliability.detach(),
            interactions=self.interactions.detach(),
            temporal_attention={k: v.detach() for k, v in self.temporal_attention.items()},
            interaction_names=self.interaction_names,
            pooled_modalities=(
                {k: v.detach() for k, v in self.pooled_modalities.items()}
                if self.pooled_modalities is not None else None
            ),
            interaction_vectors=(
                {k: v.detach() for k, v in self.interaction_vectors.items()}
                if self.interaction_vectors is not None else None
            ),
            fused=self.fused.detach() if self.fused is not None else None,
            unimodal_predictions=(
                {k: v.detach() for k, v in self.unimodal_predictions.items()}
                if self.unimodal_predictions is not None else None
            ),
        )
