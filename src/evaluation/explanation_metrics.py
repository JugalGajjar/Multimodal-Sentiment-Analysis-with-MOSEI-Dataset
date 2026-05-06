"""Math primitives shared by the explainability evaluators.

Kept as pure tensor utilities so the deletion/insertion, sufficiency, and
reliability-alignment routines all consume the same metric definitions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``KL(p || q)`` over the last dim. Returns a ``(...)`` tensor."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def spearman_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Spearman ρ between two 1-D vectors via Pearson on ranks."""
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() < 2 or x.numel() != y.numel():
        return float("nan")

    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=eps)
    return float((rx * ry).sum() / denom)


def top1_agreement(p: torch.Tensor, q: torch.Tensor) -> float:
    """Fraction of rows where ``argmax(p) == argmax(q)``."""
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    return float((p.argmax(dim=-1) == q.argmax(dim=-1)).float().mean())


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal AUC on ``(xs, ys)``. ``xs`` must be sorted."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i])
    return float(total)


def prediction_sensitivity(
    pred_clean: torch.Tensor,
    pred_modified: torch.Tensor,
    task: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-sample ``D(pred_clean, pred_modified)``.

    * Regression: absolute prediction difference, shape ``(B,)``.
    * Classification: KL between softmax distributions, shape ``(B,)``.
    """
    if task == "regression":
        return (pred_clean - pred_modified).abs()
    if task == "classification":
        p_c = F.softmax(pred_clean, dim=-1)
        p_m = F.softmax(pred_modified, dim=-1)
        return kl_divergence(p_c, p_m, eps=eps)
    raise ValueError(f"unknown task {task!r}")
