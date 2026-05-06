"""Composite X-MoFE training objective.

Returns a tuple of ``(total_loss, components)`` so the trainer can log every
weighted term independently. Reliability supervision is automatically
disabled for batches without unimodal labels (i.e. anything but CH-SIMS),
ensuring the same loss instance can serve all three datasets.

Spec §14.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.losses.entropy_loss import EntropyLoss
from src.losses.faithfulness_loss import FaithfulnessLoss
from src.losses.reliability_loss import ReliabilityLoss
from src.losses.stability_loss import StabilityLoss
from src.losses.task_losses import TaskLoss
from src.models.explanation_heads import XMoFEOutput


class XMoFELoss(nn.Module):
    """``L_total = L_task + α·L_reliability + β·L_faithfulness + γ·L_stability + δ·L_entropy``."""

    def __init__(
        self,
        task: str,
        alpha: float = 0.2,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.05,
        target_entropy: float = 0.7,
        similarity_temperature: float = 1.0,
        classification_metric: str = "kl",
        detach_faithfulness_target: bool = True,
        text_token_dropout: float = 0.1,
        audio_noise_std: float = 0.1,
        visual_patch_dropout: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        self.task_loss = TaskLoss(task, class_weights=class_weights)
        self.reliability_loss = ReliabilityLoss(similarity_temperature)
        self.faithfulness_loss = FaithfulnessLoss(
            task,
            classification_metric=classification_metric,
            detach_target=detach_faithfulness_target,
        )
        self.stability_loss = StabilityLoss(
            text_token_dropout=text_token_dropout,
            audio_noise_std=audio_noise_std,
            visual_patch_dropout=visual_patch_dropout,
        )
        self.entropy_loss = EntropyLoss(target_entropy)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        task: str,
        class_weights: torch.Tensor | None = None,
    ) -> "XMoFELoss":
        weights = config.get("weights", {})
        faith = config.get("faithfulness", {})
        stab = config.get("stability", {})
        rel = config.get("reliability_supervision", {})
        return cls(
            task=task,
            alpha=weights.get("alpha", 0.2),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.1),
            delta=weights.get("delta", 0.05),
            target_entropy=config.get("target_entropy", 0.7),
            similarity_temperature=rel.get("similarity_temperature", 1.0),
            classification_metric=faith.get("classification_metric", "kl"),
            detach_faithfulness_target=faith.get("detach_target", True),
            text_token_dropout=stab.get("text_token_dropout", 0.1),
            audio_noise_std=stab.get("audio_noise_std", 0.1),
            visual_patch_dropout=stab.get("visual_patch_dropout", 0.1),
            class_weights=class_weights,
        )

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        clean_output: XMoFEOutput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the composite loss.

        Auxiliary terms whose weight is exactly zero are skipped — saves the
        4 extra forward passes the baseline-comparison runs would otherwise
        burn (3 for faithfulness, 1 for stability) when training task-only.
        """
        if "label" not in batch:
            raise KeyError("batch must contain 'label' for the task loss")

        zero = clean_output.prediction.new_zeros(())
        components: dict[str, torch.Tensor] = {
            "task": self.task_loss(clean_output.prediction, batch["label"]),
        }

        components["entropy"] = (
            self.entropy_loss(clean_output.reliability) if self.delta > 0 else zero
        )
        components["faithfulness"] = (
            self.faithfulness_loss(model, batch, clean_output) if self.beta > 0 else zero
        )
        components["stability"] = (
            self.stability_loss(model, batch, clean_output) if self.gamma > 0 else zero
        )

        unimodal = batch.get("unimodal_labels")
        if self.alpha > 0 and unimodal is not None:
            components["reliability"] = self.reliability_loss(
                clean_output.reliability,
                unimodal,
                batch["label"].to(unimodal.dtype),
            )
        else:
            components["reliability"] = zero

        total = (
            components["task"]
            + self.alpha * components["reliability"]
            + self.beta * components["faithfulness"]
            + self.gamma * components["stability"]
            + self.delta * components["entropy"]
        )
        return total, components
