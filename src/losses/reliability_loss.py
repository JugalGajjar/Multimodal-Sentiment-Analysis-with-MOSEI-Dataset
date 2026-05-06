"""Reliability-supervision loss using CH-SIMS unimodal annotations.

CH-SIMS provides per-clip ``y_T``, ``y_A``, ``y_V`` alongside the multimodal
``y_M``. We turn the per-modality agreement with ``y_M`` into a target
reliability distribution ``r*`` and pull the model's predicted ``r`` toward
it via KL divergence. Absent unimodal labels (MOSEI / MELD), the loss is
simply not computed by the composite — there is no zeroed-out fallback here.

Spec §14.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityLoss(nn.Module):
    """``L_reliability = KL(r* || r)`` where ``r* = softmax(-|y_m - y_M|/τ)``.

    Args:
        similarity_temperature: τ in the softmax. Lower τ → sharper target.
    """

    def __init__(self, similarity_temperature: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.similarity_temperature = float(similarity_temperature)
        self.eps = eps

    def forward(
        self,
        reliability: torch.Tensor,
        unimodal_labels: torch.Tensor,
        multimodal_label: torch.Tensor,
    ) -> torch.Tensor:
        # reliability: (B, 3); unimodal_labels: (B, 3); multimodal_label: (B,)
        deltas = (unimodal_labels - multimodal_label.unsqueeze(-1)).abs()
        scores = -deltas / self.similarity_temperature
        target = F.softmax(scores, dim=-1)                      # (B, 3) — r*
        # F.kl_div computes Σ target * (log target - log_input)
        log_r = (reliability + self.eps).log()
        return F.kl_div(log_r, target, reduction="batchmean")
