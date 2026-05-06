"""Model factory dispatching across X-MoFE and the controlled-fusion baselines.

The trainer calls ``build_model(variant, config, ...)`` and gets back an
``nn.Module`` whose ``forward`` returns an :class:`XMoFEOutput`. All
variants share the same input/output signature, so the rest of the
training pipeline is variant-agnostic.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from src.models.baselines import (
    EarlyFusionModel,
    HybridFusionModel,
    LateFusionModel,
    UnimodalModel,
)
from src.models.xmofe import XMoFE

VARIANTS = ("xmofe", "early_fusion", "late_fusion", "hybrid_fusion", "unimodal")


def build_model(
    variant: str,
    config: Mapping[str, Any],
    text_dim: int,
    audio_dim: int,
    visual_dim: int,
    task: str,
    num_classes: int = 1,
    modality: str | None = None,
) -> nn.Module:
    """Instantiate the requested variant.

    Args:
        variant: one of ``VARIANTS``.
        config: parsed ``configs/models/xmofe.yaml`` (or similar). The
            baselines share architectural knobs (shared_dim, dropout, etc.)
            with X-MoFE, so a single config flows everywhere.
        text_dim, audio_dim, visual_dim: per-modality input dims.
        task, num_classes: prediction head specs.
        modality: required when ``variant == "unimodal"``.
    """
    if variant == "xmofe":
        return XMoFE.from_config(config, text_dim, audio_dim, visual_dim, task, num_classes)
    if variant == "early_fusion":
        return EarlyFusionModel.from_config(config, text_dim, audio_dim, visual_dim, task, num_classes)
    if variant == "late_fusion":
        return LateFusionModel.from_config(config, text_dim, audio_dim, visual_dim, task, num_classes)
    if variant == "hybrid_fusion":
        return HybridFusionModel.from_config(config, text_dim, audio_dim, visual_dim, task, num_classes)
    if variant == "unimodal":
        if modality is None:
            raise ValueError("variant='unimodal' requires --modality {text,audio,visual}")
        return UnimodalModel.from_config(
            config, text_dim, audio_dim, visual_dim, task, num_classes, modality=modality,
        )
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
