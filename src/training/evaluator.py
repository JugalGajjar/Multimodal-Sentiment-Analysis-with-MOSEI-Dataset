"""Task-specific metric computation for X-MoFE training and evaluation.

For each split, we collect predictions, labels, reliability/interaction
distributions, and compute:

* Regression (MOSEI sentiment, CH-SIMS sentiment_M):
    - mae, mse, pearson_r
    - binary_acc / binary_f1 — sign(prediction) vs sign(label) (the
      standard "negative vs non-negative" cut used in MOSEI papers)
* Classification (MELD emotion, optional 7-class sentiment):
    - accuracy, weighted_f1, macro_f1
    - per-class f1 (for diagnostics when imbalance is severe)

Reliability and interaction summary stats are also returned so trainers can
log how the explanation outputs evolve during training.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def _pearson(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-12) -> float:
    p = pred.float() - pred.float().mean()
    t = label.float() - label.float().mean()
    denom = (p.norm() * t.norm()).clamp(min=eps)
    return float((p * t).sum() / denom)


def _per_class_f1(label_ids: np.ndarray, pred_ids: np.ndarray, num_classes: int) -> dict[str, float]:
    f1s = f1_score(
        label_ids, pred_ids, labels=list(range(num_classes)),
        average=None, zero_division=0,
    )
    return {f"f1_class_{i}": float(v) for i, v in enumerate(f1s)}


class Evaluator:
    """Compute metrics on a dataloader.

    Args:
        task: ``"regression"`` or ``"classification"``.
        num_classes: needed for classification per-class metrics.
        device: passed to ``model``-input batches.
    """

    def __init__(self, task: str, num_classes: int = 1, device: str | torch.device = "cpu") -> None:
        if task not in {"regression", "classification"}:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")
        self.task = task
        self.num_classes = num_classes
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[dict[str, Any]],
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()

        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_reliability: list[torch.Tensor] = []
        all_interactions: list[torch.Tensor] = []

        # Only forward raw transcripts to models that carry an in-graph text
        # encoder (XMoFE with text.finetune=true). Baselines and unimodal
        # heads have a strict forward signature that rejects this kwarg.
        allowed_keys = {
            "text", "audio", "visual",
            "text_length", "audio_length", "visual_length",
            "quality",
        }
        if getattr(model, "text_encoder", None) is not None:
            allowed_keys.add("transcripts")

        for batch in dataloader:
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k in allowed_keys
            }
            out = model(**inputs)
            all_preds.append(out.prediction.detach().to("cpu"))
            all_labels.append(batch["label"].to("cpu"))
            all_reliability.append(out.reliability.detach().to("cpu"))
            all_interactions.append(out.interactions.detach().to("cpu"))

        if was_training:
            model.train()

        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        reliability = torch.cat(all_reliability, dim=0)
        interactions = torch.cat(all_interactions, dim=0)

        metrics: dict[str, float] = {}
        if self.task == "regression":
            metrics["mae"] = float((preds - labels.float()).abs().mean())
            metrics["mse"] = float(((preds - labels.float()) ** 2).mean())
            metrics["pearson_r"] = _pearson(preds, labels)
            # MOSEI / MOSI standard: sign-based binary classification
            pred_pos = (preds >= 0).long().numpy()
            label_pos = (labels >= 0).long().numpy()
            metrics["binary_acc"] = float(accuracy_score(label_pos, pred_pos))
            metrics["binary_f1"] = float(
                f1_score(label_pos, pred_pos, average="weighted", zero_division=0)
            )
        else:
            pred_ids = preds.argmax(dim=-1).numpy()
            label_ids = labels.long().numpy()
            metrics["accuracy"] = float(accuracy_score(label_ids, pred_ids))
            metrics["weighted_f1"] = float(
                f1_score(label_ids, pred_ids, average="weighted", zero_division=0)
            )
            metrics["macro_f1"] = float(
                f1_score(label_ids, pred_ids, average="macro", zero_division=0)
            )
            metrics.update(_per_class_f1(label_ids, pred_ids, self.num_classes))

        # Explanation summary statistics — useful in plots over training.
        modality_names = ("text", "audio", "visual")
        for i, name in enumerate(modality_names):
            metrics[f"reliability_mean_{name}"] = float(reliability[:, i].mean())
            metrics[f"reliability_std_{name}"] = float(reliability[:, i].std(unbiased=False))
        for i in range(interactions.shape[1]):
            metrics[f"interaction_mean_{i}"] = float(interactions[:, i].mean())
        # Average entropy of reliability across the split (uniform = ln 3 ≈ 1.0986)
        eps = 1e-12
        ent = -(reliability * (reliability + eps).log()).sum(-1)
        metrics["reliability_entropy_mean"] = float(ent.mean())

        return metrics
