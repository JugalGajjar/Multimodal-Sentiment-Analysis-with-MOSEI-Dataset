"""Sufficiency and comprehensiveness — single-threshold readouts of
deletion/insertion (spec §17.4 + §17.5).

* **Sufficiency**  — keep only the top-k_fraction attended positions.
  A *sufficient* explanation makes the prediction stay close to the
  original, so this returns the residual prediction sensitivity at the
  chosen ``k_fraction``. Smaller is better.

* **Comprehensiveness** — remove the top-k_fraction attended positions.
  A *comprehensive* explanation makes the prediction change a lot when
  those positions are removed. Larger is better.

These reuse the deletion/insertion machinery at a single ``k_fraction``
default of 0.2 (20%), the most common literature setting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from src.evaluation.deletion_insertion import deletion_curve, insertion_curve

DEFAULT_K_FRACTION: float = 0.2


@torch.no_grad()
def sufficiency(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    task: str,
    modality: str,
    k_fraction: float = DEFAULT_K_FRACTION,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Return the insertion-test residual sensitivity at a single threshold."""
    res = insertion_curve(model, dataloader, task, modality, k_fractions=[k_fraction], max_batches=max_batches)
    return {
        "modality": modality,
        "k_fraction": k_fraction,
        "sufficiency": res["mean_residual_sensitivity"][0],
        "n_samples": res["n_samples"],
    }


@torch.no_grad()
def comprehensiveness(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    task: str,
    modality: str,
    k_fraction: float = DEFAULT_K_FRACTION,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Return the deletion-test sensitivity at a single threshold."""
    res = deletion_curve(model, dataloader, task, modality, k_fractions=[k_fraction], max_batches=max_batches)
    return {
        "modality": modality,
        "k_fraction": k_fraction,
        "comprehensiveness": res["mean_sensitivity"][0],
        "n_samples": res["n_samples"],
    }
