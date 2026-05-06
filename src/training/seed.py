"""Reproducibility helpers — seeds Python / numpy / torch / MPS / CUDA in one call."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG we touch.

    Args:
        seed: integer seed shared across libraries.
        deterministic: if True, enables PyTorch's strict deterministic mode.
            This trades off some speed for guaranteed bitwise reproducibility
            on the same hardware. Off by default — most multimodal-sentiment
            papers don't need it and MPS doesn't fully support it yet.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
