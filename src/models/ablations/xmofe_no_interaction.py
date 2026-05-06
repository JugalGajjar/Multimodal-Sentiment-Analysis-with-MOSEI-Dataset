"""``XMoFENoInteraction`` — X-MoFE with the cross-modal interaction block removed.

Drops the pairwise cross-attention blocks (``c_TA``, ``c_TV``, ``c_AV``),
the optional tri-modal block, and the interaction-contribution estimator.
Fusion sees only the unimodal evidence stream: ``h = LayerNorm(u)`` with
``i = 0``. Spec §19.1's "w/o interaction block" row.

Tests **H3** (spec §4): does explicit interaction modeling carry
predictive weight beyond reliability-weighted unimodal fusion?
"""

from __future__ import annotations

from typing import Any, Mapping

from src.models.xmofe import XMoFE


class XMoFENoInteraction(XMoFE):
    """X-MoFE with the cross-modal interaction block ablated to zero."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_interaction_block"] = False
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        task: str,
        num_classes: int = 1,
    ) -> "XMoFENoInteraction":
        rel = config.get("reliability") or {}
        return cls(
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            attention_heads=config.get("attention_heads", 4),
            dropout=config.get("dropout", 0.2),
            task=task, num_classes=num_classes,
            num_quality_features_per_modality=(
                rel.get("num_quality_features", 0) if rel.get("use_quality_features", False) else 0
            ),
            ffn_multiplier=config.get("ffn_multiplier", 4),
            reliability_mlp_hidden=rel.get("mlp_hidden", 256),
        )
