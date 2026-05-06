"""Task losses: MSE for regression (MOSEI, CH-SIMS), CE for classification (MELD)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TaskLoss(nn.Module):
    """Dispatch wrapper around ``nn.MSELoss`` / ``nn.CrossEntropyLoss``.

    Args:
        task: ``"regression"`` or ``"classification"``.
        class_weights: optional ``(num_classes,)`` tensor for imbalanced
            classification (MELD's emotion distribution is skewed: 47% neutral
            vs 2.6% fear). Ignored for regression.
    """

    def __init__(self, task: str, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if task == "regression":
            self.criterion: nn.Module = nn.MSELoss()
        elif task == "classification":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
        self.task = task

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return self.criterion(prediction, label.to(prediction.dtype))
        # Cross-entropy expects long-typed targets
        return self.criterion(prediction, label.long())
