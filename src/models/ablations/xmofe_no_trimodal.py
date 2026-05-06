"""``XMoFENoTrimodal`` — X-MoFE without the tri-modal interaction block.

Drops only ``c_TAV``; the three pairwise blocks (``c_TA``, ``c_TV``,
``c_AV``) and the interaction-contribution estimator (now over 3 columns)
remain intact. Spec §19.1's "w/o tri-modal interaction" row.

Tests whether the optional tri-modal interaction adds predictive or
explanatory value beyond the pairwise interactions, motivating its
inclusion in the default architecture.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.models.xmofe import XMoFE


class XMoFENoTrimodal(XMoFE):
    """X-MoFE with ``c_TAV`` ablated; pairwise interactions retained."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_trimodal"] = False
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
    ) -> "XMoFENoTrimodal":
        rel = config.get("reliability") or {}
        inter = config.get("interaction") or {}
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
            interaction_mlp_hidden=inter.get("mlp_hidden", 256),
        )
