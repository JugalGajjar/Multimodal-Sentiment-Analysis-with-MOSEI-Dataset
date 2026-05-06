"""Faithfulness loss aligning reliability scores with prediction sensitivity.

If the model claims modality m matters (high ``r_m``), removing modality m
should change the prediction. We compute three additional model forward
passes — each with one modality "removed" by setting its length to zero so
the architecture's missing-token logic kicks in — and use the resulting
prediction sensitivities as a target distribution ``s`` over modalities.
The KL between that observed distribution and the model's claimed ``r`` is
the faithfulness penalty.

Spec §14.3.

This loss runs three extra forwards per batch — the dominant compute cost
in the composite. The trainer can amortize by computing it less frequently
than every step if needed; that's a Phase-5 concern.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.explanation_heads import XMoFEOutput

MODALITIES = ("text", "audio", "visual")


class FaithfulnessLoss(nn.Module):
    """``L_faithfulness = KL(s || r)`` with ``s = softmax([Δ_T, Δ_A, Δ_V])``.

    Args:
        task: ``"regression"`` or ``"classification"`` — selects the
            sensitivity metric ``D``.
        classification_metric: how to score sensitivity for classification —
            ``"kl"`` (KL between predicted distributions),
            ``"tvd"`` (total variation),
            ``"prob_drop"`` (drop in probability of clean's argmax class).
        detach_target: if True, gradient flows only through ``r`` (not ``s``).
            This focuses learning on aligning reliability with observed
            sensitivities rather than reshaping the ablated forwards.
    """

    def __init__(
        self,
        task: str,
        classification_metric: str = "kl",
        detach_target: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
        if classification_metric not in {"kl", "tvd", "prob_drop"}:
            raise ValueError(f"unknown classification_metric: {classification_metric}")
        self.task = task
        self.classification_metric = classification_metric
        self.detach_target = detach_target
        self.eps = eps

    def _sensitivity(
        self,
        pred_clean: torch.Tensor,
        pred_modified: torch.Tensor,
    ) -> torch.Tensor:
        if self.task == "regression":
            return (pred_clean - pred_modified).abs()           # (B,)

        p_clean = F.softmax(pred_clean, dim=-1)
        p_mod = F.softmax(pred_modified, dim=-1)
        if self.classification_metric == "kl":
            return (
                p_clean
                * ((p_clean + self.eps).log() - (p_mod + self.eps).log())
            ).sum(dim=-1)
        if self.classification_metric == "tvd":
            return 0.5 * (p_clean - p_mod).abs().sum(dim=-1)
        # prob_drop
        argmax = p_clean.argmax(dim=-1, keepdim=True)
        drop = p_clean.gather(-1, argmax) - p_mod.gather(-1, argmax)
        return drop.squeeze(-1).clamp(min=0.0)

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        clean_output: XMoFEOutput,
    ) -> torch.Tensor:
        deltas: list[torch.Tensor] = []
        for modality in MODALITIES:
            length_key = f"{modality}_length"
            if length_key not in batch:
                raise KeyError(f"batch missing key {length_key!r}")
            modified_batch = {**batch, length_key: torch.zeros_like(batch[length_key])}
            # Drop label/unimodal_labels: model.forward doesn't accept them
            modified_inputs = {
                k: v for k, v in modified_batch.items()
                if k in {
                    "text", "audio", "visual",
                    "text_length", "audio_length", "visual_length",
                    "quality",
                }
            }
            modified_pred = model(**modified_inputs).prediction
            deltas.append(self._sensitivity(clean_output.prediction, modified_pred))

        deltas_stacked = torch.stack(deltas, dim=-1)            # (B, 3)
        target = F.softmax(deltas_stacked, dim=-1)              # (B, 3) — s
        if self.detach_target:
            target = target.detach()
        log_r = (clean_output.reliability + self.eps).log()
        return F.kl_div(log_r, target, reduction="batchmean")
