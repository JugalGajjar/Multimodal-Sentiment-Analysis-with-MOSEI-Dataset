"""Temporal deletion and insertion tests for X-MoFE explanations.

For each modality and each sample, we use the model's per-position
attention weights ``α_m ∈ R^{L_m}`` as the explanation. The deletion
test progressively zeroes out the top-k attended positions and measures
how much the prediction changes — a faithful explanation should produce a
large drop. The insertion test starts from an all-zero (modality-only)
baseline and progressively inserts the top-k attended positions back in,
measuring how quickly the prediction recovers — a faithful explanation
should recover quickly (low residual sensitivity).

Convention used here:

* **Deletion AULC** larger ⇒ better — top-k positions matter more.
* **Insertion AULC** smaller ⇒ better — top-k features are sufficient.

``k_fraction`` is interpreted **relative to each sample's valid length**
(not the cache's padded ``L_max``), so a 20-token clip isn't asked to
"remove top-50% of 1568 positions". This costs a small Python loop per
batch but keeps the metric meaningful for variable-length inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from src.evaluation.explanation_metrics import prediction_sensitivity, trapezoid_auc

DEFAULT_K_FRACTIONS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
MODEL_INPUT_KEYS = (
    "text", "audio", "visual",
    "text_length", "audio_length", "visual_length",
    "quality",
)


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}


def _per_sample_top_k_indices(
    attention: torch.Tensor,
    lengths: torch.Tensor,
    k_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each sample, get the indices of the top ``k_fraction × length`` positions.

    Returns:
        rows: ``(N,)`` flat row indices into the (B, L) tensor.
        cols: ``(N,)`` flat column indices.
    """
    b, l = attention.shape
    rows_list: list[torch.Tensor] = []
    cols_list: list[torch.Tensor] = []
    for i in range(b):
        valid_len = int(lengths[i].item())
        if valid_len <= 0:
            continue
        k = max(1, int(round(valid_len * k_fraction)))
        k = min(k, valid_len)
        # rank only over valid positions
        valid_attn = attention[i, :valid_len]
        _, top_idx = torch.topk(valid_attn, k)
        rows_list.append(torch.full((k,), i, dtype=torch.long, device=attention.device))
        cols_list.append(top_idx.to(torch.long))
    if not rows_list:
        return (
            torch.zeros(0, dtype=torch.long, device=attention.device),
            torch.zeros(0, dtype=torch.long, device=attention.device),
        )
    return torch.cat(rows_list), torch.cat(cols_list)


def mask_top_k_positions(
    features: torch.Tensor,
    attention: torch.Tensor,
    lengths: torch.Tensor,
    k_fraction: float,
) -> torch.Tensor:
    """Zero out the top-k_fraction valid positions of ``features``."""
    if k_fraction <= 0:
        return features
    rows, cols = _per_sample_top_k_indices(attention, lengths, k_fraction)
    mask = torch.ones(features.shape[:2], device=features.device, dtype=features.dtype)
    if rows.numel() > 0:
        mask[rows, cols] = 0.0
    return features * mask.unsqueeze(-1)


def keep_only_top_k_positions(
    features: torch.Tensor,
    attention: torch.Tensor,
    lengths: torch.Tensor,
    k_fraction: float,
) -> torch.Tensor:
    """Keep only the top-k_fraction valid positions of ``features`` (zero elsewhere)."""
    rows, cols = _per_sample_top_k_indices(attention, lengths, k_fraction)
    mask = torch.zeros(features.shape[:2], device=features.device, dtype=features.dtype)
    if rows.numel() > 0:
        mask[rows, cols] = 1.0
    return features * mask.unsqueeze(-1)


def _modified_inputs(
    batch: Mapping[str, Any],
    modality: str,
    new_features: torch.Tensor,
) -> dict[str, Any]:
    out = _model_inputs(batch)
    out[modality] = new_features
    return out


def _ensure_xmofe_inputs(modality: str) -> str:
    if modality not in ("text", "audio", "visual"):
        raise ValueError(f"modality must be text/audio/visual; got {modality!r}")
    return modality


@torch.no_grad()
def deletion_curve(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    task: str,
    modality: str,
    k_fractions: Iterable[float] = DEFAULT_K_FRACTIONS,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Compute mean prediction sensitivity at each ``k_fraction`` + AULC."""
    _ensure_xmofe_inputs(modality)
    k_list = sorted(set(k_fractions))
    sens_per_k: dict[float, list[float]] = {k: [] for k in k_list}

    was_training = model.training
    model.eval()
    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = _model_inputs(batch)
            clean_out = model(**inputs)
            attention = clean_out.temporal_attention[modality]
            lengths = batch[f"{modality}_length"]

            for k in k_list:
                masked = mask_top_k_positions(batch[modality], attention, lengths, k)
                pred = model(**_modified_inputs(batch, modality, masked)).prediction
                delta = prediction_sensitivity(clean_out.prediction, pred, task)
                sens_per_k[k].extend(delta.detach().cpu().tolist())
    finally:
        if was_training:
            model.train()

    means = [
        float(torch.tensor(sens_per_k[k]).mean().item()) if sens_per_k[k] else float("nan")
        for k in k_list
    ]
    aulc = trapezoid_auc(list(k_list), means)
    return {
        "modality": modality,
        "k_fractions": list(k_list),
        "mean_sensitivity": means,
        "aulc": aulc,
        "n_samples": len(sens_per_k[k_list[0]]) if k_list else 0,
    }


@torch.no_grad()
def insertion_curve(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    task: str,
    modality: str,
    k_fractions: Iterable[float] = DEFAULT_K_FRACTIONS,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Mean residual prediction sensitivity when keeping only top-k features.

    Lower mean / lower AULC ⇒ better explanation (top-k features are
    sufficient to reproduce the clean prediction).
    """
    _ensure_xmofe_inputs(modality)
    k_list = sorted(set(k_fractions))
    sens_per_k: dict[float, list[float]] = {k: [] for k in k_list}

    was_training = model.training
    model.eval()
    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = _model_inputs(batch)
            clean_out = model(**inputs)
            attention = clean_out.temporal_attention[modality]
            lengths = batch[f"{modality}_length"]

            for k in k_list:
                kept = keep_only_top_k_positions(batch[modality], attention, lengths, k)
                pred = model(**_modified_inputs(batch, modality, kept)).prediction
                delta = prediction_sensitivity(clean_out.prediction, pred, task)
                sens_per_k[k].extend(delta.detach().cpu().tolist())
    finally:
        if was_training:
            model.train()

    means = [
        float(torch.tensor(sens_per_k[k]).mean().item()) if sens_per_k[k] else float("nan")
        for k in k_list
    ]
    aulc = trapezoid_auc(list(k_list), means)
    return {
        "modality": modality,
        "k_fractions": list(k_list),
        "mean_residual_sensitivity": means,
        "aulc": aulc,
        "n_samples": len(sens_per_k[k_list[0]]) if k_list else 0,
    }
