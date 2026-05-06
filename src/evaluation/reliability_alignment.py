"""Reliability-alignment evaluations (spec §17.1 + §17.6).

Two flavors:

* :func:`modality_faithfulness` — applies to every dataset. Computes
  observed prediction sensitivity to each modality being removed, turns it
  into a per-sample distribution ``s = softmax([Δ_T, Δ_A, Δ_V])``, and
  compares to the model's claimed reliability ``r``. A faithful model has
  high Spearman(r, s), low KL(s ‖ r), and high top-1 agreement.

* :func:`chsims_reliability_alignment` — CH-SIMS only. Uses the dataset's
  unimodal annotations ``y_T, y_A, y_V`` to build an *external* target
  distribution ``r* = softmax(-|y_m - y_M|)``, then compares ``r`` to ``r*``.
  Auto-skips datasets without unimodal labels (returns ``None``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.evaluation.explanation_metrics import (
    kl_divergence,
    prediction_sensitivity,
    spearman_correlation,
    top1_agreement,
)

MODEL_INPUT_KEYS = (
    "text", "audio", "visual",
    "text_length", "audio_length", "visual_length",
    "quality",
)


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}


@torch.no_grad()
def modality_faithfulness(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    task: str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Compare model reliability ``r`` to observed prediction sensitivity ``s``."""
    was_training = model.training
    model.eval()

    all_r: list[torch.Tensor] = []
    all_s: list[torch.Tensor] = []

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = _model_inputs(batch)
            clean = model(**inputs)
            deltas: list[torch.Tensor] = []
            for modality in ("text", "audio", "visual"):
                ablated_inputs = {**inputs, f"{modality}_length": torch.zeros_like(batch[f"{modality}_length"])}
                ablated = model(**ablated_inputs).prediction
                deltas.append(prediction_sensitivity(clean.prediction, ablated, task))
            stacked = torch.stack(deltas, dim=-1)            # (B, 3)
            s = F.softmax(stacked, dim=-1)
            all_r.append(clean.reliability.detach().cpu())
            all_s.append(s.detach().cpu())
    finally:
        if was_training:
            model.train()

    if not all_r:
        return {"n_samples": 0}

    r = torch.cat(all_r, dim=0)
    s = torch.cat(all_s, dim=0)

    # Spearman over the flattened (sample, modality) cross-product captures
    # whether high-r positions tend to be high-s positions across the whole
    # eval set; per-sample Spearman wouldn't be useful with only 3 dims.
    return {
        "n_samples": int(r.shape[0]),
        "spearman": spearman_correlation(r, s),
        "kl_s_to_r_mean": float(kl_divergence(s, r).mean()),
        "top1_agreement": top1_agreement(r, s),
        "reliability_mean": [float(r[:, i].mean()) for i in range(r.shape[1])],
        "sensitivity_mean": [float(s[:, i].mean()) for i in range(s.shape[1])],
    }


@torch.no_grad()
def chsims_reliability_alignment(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    similarity_temperature: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, Any] | None:
    """Compare model reliability to CH-SIMS unimodal-label-derived target.

    Returns ``None`` if the dataloader has no unimodal labels (e.g. MELD,
    MOSEI).
    """
    was_training = model.training
    model.eval()

    all_r: list[torch.Tensor] = []
    all_r_star: list[torch.Tensor] = []

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            unimodal = batch.get("unimodal_labels")
            if unimodal is None:
                continue
            multimodal = batch["label"].to(unimodal.dtype)
            deltas = (unimodal - multimodal.unsqueeze(-1)).abs()
            scores = -deltas / similarity_temperature
            r_star = F.softmax(scores, dim=-1)            # (B, 3)
            inputs = _model_inputs(batch)
            r = model(**inputs).reliability
            all_r.append(r.detach().cpu())
            all_r_star.append(r_star.detach().cpu())
    finally:
        if was_training:
            model.train()

    if not all_r:
        return None

    r = torch.cat(all_r, dim=0)
    r_star = torch.cat(all_r_star, dim=0)

    return {
        "n_samples": int(r.shape[0]),
        "spearman": spearman_correlation(r, r_star),
        "kl_rstar_to_r_mean": float(kl_divergence(r_star, r).mean()),
        "top1_agreement": top1_agreement(r, r_star),
        "reliability_mean": [float(r[:, i].mean()) for i in range(r.shape[1])],
        "rstar_mean": [float(r_star[:, i].mean()) for i in range(r_star.shape[1])],
    }
