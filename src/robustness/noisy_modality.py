"""Noisy-modality evaluation conditions (spec §16.2).

Per-modality severity dial:

    * text:   token-vector dropout probability
    * audio:  Gaussian noise standard deviation
    * visual: patch/frame-vector dropout probability

Default ``medium`` severity is the spec's "compute-aware starting point".
Each call optionally accepts a ``torch.Generator`` so the trainer/evaluator
can produce deterministic noise patterns when reproducibility matters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.robustness.audio_corruptions import gaussian_noise as _audio_noise
from src.robustness.text_corruptions import token_dropout as _text_dropout
from src.robustness.visual_corruptions import patch_dropout as _visual_dropout

# Severity → per-modality strength.
SEVERITY_LEVELS: dict[str, dict[str, float]] = {
    "low":    {"text": 0.05, "audio": 0.05, "visual": 0.05},
    "medium": {"text": 0.15, "audio": 0.10, "visual": 0.15},
    "high":   {"text": 0.30, "audio": 0.25, "visual": 0.30},
}

VALID_MODALITIES = ("text", "audio", "visual")


def apply_noise(
    batch: Mapping[str, Any],
    modality: str,
    severity: str = "medium",
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Apply the configured corruption to the named modality at ``severity``."""
    if modality not in VALID_MODALITIES:
        raise ValueError(f"modality must be one of {VALID_MODALITIES}; got {modality!r}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {list(SEVERITY_LEVELS)}; got {severity!r}")
    strength = SEVERITY_LEVELS[severity][modality]

    out: dict[str, Any] = dict(batch)
    if modality == "text":
        out["text"] = _text_dropout(batch["text"], strength, generator=generator)
    elif modality == "audio":
        out["audio"] = _audio_noise(batch["audio"], strength, generator=generator)
    else:  # visual
        out["visual"] = _visual_dropout(batch["visual"], strength, generator=generator)
    return out
