"""Stability loss penalizing explanation drift under label-preserving perturbations.

Generates a perturbed copy of the batch (token dropout for text, Gaussian
noise for audio, patch dropout for visual) and runs an extra forward pass.
The L2 distance between the clean and perturbed reliability + interaction
distributions is the stability penalty. We deliberately backprop through
both the clean and perturbed forwards — the model should be invariant to
small perturbations, and the gradient should reach both sides.

Spec §14.4.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.models.explanation_heads import XMoFEOutput


class StabilityLoss(nn.Module):
    """``L_stability = ||E_modality(x) - E_modality(x')||₂ + ||E_interaction(x) - E_interaction(x')||₂``."""

    def __init__(
        self,
        text_token_dropout: float = 0.1,
        audio_noise_std: float = 0.1,
        visual_patch_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_token_dropout = float(text_token_dropout)
        self.audio_noise_std = float(audio_noise_std)
        self.visual_patch_dropout = float(visual_patch_dropout)

    @staticmethod
    def _token_dropout(features: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0:
            return features
        # mask: (B, L, 1) — drops the same fraction of tokens per sample
        mask = (torch.rand(features.size(0), features.size(1), 1, device=features.device) > p).to(features.dtype)
        return features * mask

    @staticmethod
    def _gaussian_noise(features: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return features
        return features + std * torch.randn_like(features)

    def _perturb(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        if "text" in batch:
            out["text"] = self._token_dropout(batch["text"], self.text_token_dropout)
        if "audio" in batch:
            out["audio"] = self._gaussian_noise(batch["audio"], self.audio_noise_std)
        if "visual" in batch:
            out["visual"] = self._token_dropout(batch["visual"], self.visual_patch_dropout)
        return out

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        clean_output: XMoFEOutput,
    ) -> torch.Tensor:
        perturbed_batch = self._perturb(batch)
        perturbed_inputs = {
            k: v for k, v in perturbed_batch.items()
            if k in {
                "text", "audio", "visual",
                "text_length", "audio_length", "visual_length",
                "quality",
            }
        }
        perturbed_output = model(**perturbed_inputs)

        l2_modality = (clean_output.reliability - perturbed_output.reliability).norm(dim=-1).mean()
        l2_interaction = (clean_output.interactions - perturbed_output.interactions).norm(dim=-1).mean()
        return l2_modality + l2_interaction
