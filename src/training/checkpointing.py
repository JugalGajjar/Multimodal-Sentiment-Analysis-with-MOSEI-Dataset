"""Checkpoint save/load with optimizer + scheduler + RNG state.

Two checkpoint roles:
    * ``best.pt``  — saved on every val-metric improvement; the artifact
      shipped to evaluation.
    * ``latest.pt`` — saved every epoch for resumability after interruption.

A resumed run restores model + optimizer + scheduler + epoch + best metric,
plus the W&B run id so the same run continues in the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Persist a checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extras:
        payload["extras"] = extras

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring optimizer/scheduler state.

    Returns the raw payload so callers can inspect ``epoch``, ``best_metric``,
    or ``extras`` (e.g. the W&B run id).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
