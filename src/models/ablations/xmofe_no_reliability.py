"""``XMoFENoReliability`` — X-MoFE with the modality reliability gate removed.

Replaces the learned ``MLP_R`` reliability estimator with a fixed uniform
``r = [1/3, 1/3, 1/3]``, so the unimodal evidence reduces to a plain mean
``u = (z_T + z_A + z_V) / 3``. Spec §19.1's "w/o reliability gate" row.

Tests **H1** (spec §4): does sample-specific reliability estimation
actually beat uniform weighting? If this ablation matches X-MoFE on
predictive metrics, the reliability gate isn't doing useful work.

Note: any composite-loss term that operates on ``r`` (reliability
supervision, faithfulness alignment, entropy regularization) becomes a
constant on the (now non-trainable) uniform distribution and won't
contribute gradients. Run alongside ``loss_no_reliability_supervision``
or just rely on the task loss for cleanest interpretation.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.models.xmofe import XMoFE


class XMoFENoReliability(XMoFE):
    """X-MoFE with the reliability gate ablated to fixed uniform 1/3."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_reliability_gate"] = False
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
    ) -> "XMoFENoReliability":
        rel = config.get("reliability") or {}
        inter = config.get("interaction") or {}
        return cls(
            text_dim=text_dim, audio_dim=audio_dim, visual_dim=visual_dim,
            shared_dim=config.get("shared_dim", 256),
            attention_heads=config.get("attention_heads", 4),
            dropout=config.get("dropout", 0.2),
            use_trimodal=inter.get("use_trimodal", True),
            task=task, num_classes=num_classes,
            num_quality_features_per_modality=(
                rel.get("num_quality_features", 0) if rel.get("use_quality_features", False) else 0
            ),
            ffn_multiplier=config.get("ffn_multiplier", 4),
            interaction_mlp_hidden=inter.get("mlp_hidden", 256),
        )
