"""Task-specific prediction heads.

Regression returns a scalar per sample (used by MOSEI/CH-SIMS). Classification
returns logits over ``num_classes`` (MELD: 7 emotions, MOSEI 7-class: 7
sentiment buckets). The trainer applies the appropriate loss; this module
deliberately *does not* apply softmax/sigmoid so logits stay numerically
clean for cross-entropy / log-likelihood losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        task: str,
        num_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
        self.task = task
        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, out_dim),
        )

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        out = self.head(h_fused)
        if self.task == "regression":
            return out.squeeze(-1)
        return out
