"""Standard hybrid-fusion baseline.

Same architecture as :class:`src.models.xmofe.XMoFE` minus the reliability
estimator and the interaction contribution estimator: pool each modality,
compute the same pairwise + tri-modal cross-modal interactions, then
*concatenate everything* into the fusion MLP without any learned weights.
This isolates X-MoFE's explainability mechanism — same fusion components,
no XAI losses, no exposed explanation distributions.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPool
from src.models.baselines._common import uniform_interactions, uniform_reliability
from src.models.explanation_heads import XMoFEOutput
from src.models.fusion_layers import HybridFusion
from src.models.interaction import CrossModalInteraction, TriModalInteraction
from src.models.prediction_heads import PredictionHead
from src.models.projections import ModalityProjection


class HybridFusionModel(nn.Module):
    """Pool + cross-modal interactions concatenated into a fusion MLP."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        shared_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.2,
        use_trimodal: bool = True,
        task: str = "regression",
        num_classes: int = 1,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.use_trimodal = use_trimodal
        self.task = task

        self.proj_text = ModalityProjection(text_dim, shared_dim, dropout)
        self.proj_audio = ModalityProjection(audio_dim, shared_dim, dropout)
        self.proj_visual = ModalityProjection(visual_dim, shared_dim, dropout)

        self.pool_text = AttentionPool(shared_dim)
        self.pool_audio = AttentionPool(shared_dim)
        self.pool_visual = AttentionPool(shared_dim)

        self.cross_ta = CrossModalInteraction(shared_dim, attention_heads, dropout)
        self.cross_tv = CrossModalInteraction(shared_dim, attention_heads, dropout)
        self.cross_av = CrossModalInteraction(shared_dim, attention_heads, dropout)
        self.cross_tav: TriModalInteraction | None = (
            TriModalInteraction(shared_dim, attention_heads, dropout) if use_trimodal else None
        )

        # Concat 3 pooled + (3 or 4) interaction vectors → project to shared_dim → fuse.
        n_components = 3 + (4 if use_trimodal else 3)
        self.combine = nn.Sequential(
            nn.Linear(n_components * shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Reuse the X-MoFE FFN-with-residual fusion layer for parity.
        self.fusion = HybridFusion(shared_dim, dropout=dropout, ffn_multiplier=ffn_multiplier)
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
    ) -> "HybridFusionModel":
        inter = config.get("interaction") or {}
        return cls(
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            attention_heads=config.get("attention_heads", 4),
            dropout=config.get("dropout", 0.2),
            use_trimodal=inter.get("use_trimodal", True),
            ffn_multiplier=config.get("ffn_multiplier", 4),
            task=task, num_classes=num_classes,
        )

    def forward(
        self,
        text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor,
        text_length: torch.Tensor, audio_length: torch.Tensor, visual_length: torch.Tensor,
        quality: torch.Tensor | None = None,  # noqa: ARG002
        return_intermediates: bool = True,
    ) -> XMoFEOutput:
        Z_t = self.proj_text(text)
        Z_a = self.proj_audio(audio)
        Z_v = self.proj_visual(visual)

        z_t, alpha_t = self.pool_text(Z_t, text_length)
        z_a, alpha_a = self.pool_audio(Z_a, audio_length)
        z_v, alpha_v = self.pool_visual(Z_v, visual_length)

        c_ta = self.cross_ta(Z_t, Z_a, text_length, audio_length)
        c_tv = self.cross_tv(Z_t, Z_v, text_length, visual_length)
        c_av = self.cross_av(Z_a, Z_v, audio_length, visual_length)
        if self.cross_tav is not None:
            c_tav = self.cross_tav(Z_t, Z_a, Z_v, text_length, audio_length, visual_length)
            cross_vecs: tuple[torch.Tensor, ...] = (c_ta, c_tv, c_av, c_tav)
        else:
            cross_vecs = (c_ta, c_tv, c_av)

        # Equal-weight combine: just concatenate everything and project down.
        # No reliability or interaction-contribution scoring.
        combined = torch.cat([z_t, z_a, z_v, *cross_vecs], dim=-1)
        u = self.combine(combined)
        # We split the FFN-with-residual call into u + 0 so the same
        # HybridFusion module signature works without inventing a separate
        # interaction stream here.
        h_fused = self.fusion(u, torch.zeros_like(u))
        prediction = self.head(h_fused)

        b = text.size(0)
        return XMoFEOutput(
            prediction=prediction,
            reliability=uniform_reliability(b, device=text.device),
            interactions=uniform_interactions(b, num_interactions=len(cross_vecs), device=text.device),
            temporal_attention={"text": alpha_t, "audio": alpha_a, "visual": alpha_v},
            interaction_names=(
                ("text_audio", "text_visual", "audio_visual", "trimodal")
                if self.use_trimodal
                else ("text_audio", "text_visual", "audio_visual")
            ),
            pooled_modalities=(
                {"text": z_t, "audio": z_a, "visual": z_v} if return_intermediates else None
            ),
            interaction_vectors=(
                dict(zip(
                    ("text_audio", "text_visual", "audio_visual", "trimodal")
                    if self.use_trimodal
                    else ("text_audio", "text_visual", "audio_visual"),
                    cross_vecs,
                )) if return_intermediates else None
            ),
            fused=h_fused if return_intermediates else None,
        )
