"""Tests for the X-MoFE loss components and the composite objective."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.losses import (
    EntropyLoss,
    FaithfulnessLoss,
    ReliabilityLoss,
    StabilityLoss,
    TaskLoss,
    XMoFELoss,
)
from src.models import XMoFE

REPO_ROOT = Path(__file__).resolve().parents[1]
LOSS_CONFIG = REPO_ROOT / "configs" / "training" / "loss.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "models" / "xmofe.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loss_config() -> dict:
    with LOSS_CONFIG.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model_config() -> dict:
    with MODEL_CONFIG.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def synthetic_batch():
    torch.manual_seed(0)
    return {
        "text": torch.randn(4, 128, 768),
        "audio": torch.randn(4, 399, 768),
        "visual": torch.randn(4, 1568, 768),
        "text_length": torch.tensor([13, 50, 128, 7]),
        "audio_length": torch.tensor([100, 250, 399, 50]),
        "visual_length": torch.tensor([1568, 1568, 1568, 1568]),
        "label": torch.tensor([0, 3, 5, 1]),
    }


@pytest.fixture
def cls_model():
    torch.manual_seed(0)
    return XMoFE(768, 768, 768, num_classes=7, task="classification")


@pytest.fixture
def reg_model():
    torch.manual_seed(0)
    return XMoFE(768, 768, 768, num_classes=1, task="regression")


# ---------------------------------------------------------------------------
# TaskLoss
# ---------------------------------------------------------------------------


def test_task_loss_regression():
    loss = TaskLoss(task="regression")
    pred = torch.tensor([1.0, 2.0, 3.0])
    label = torch.tensor([1.5, 1.5, 3.5])
    assert loss(pred, label).item() == pytest.approx(((0.5 ** 2) + (0.5 ** 2) + (0.5 ** 2)) / 3, abs=1e-6)


def test_task_loss_classification_accepts_long_labels():
    loss = TaskLoss(task="classification")
    pred = torch.randn(4, 7)
    label_long = torch.tensor([0, 3, 5, 1])
    label_int = label_long.to(torch.int32)
    # Should accept either dtype and produce identical loss
    assert torch.allclose(loss(pred, label_long), loss(pred, label_int), atol=1e-6)


def test_task_loss_invalid_task_raises():
    with pytest.raises(ValueError):
        TaskLoss(task="multilabel")


# ---------------------------------------------------------------------------
# EntropyLoss
# ---------------------------------------------------------------------------


def test_entropy_loss_uniform_at_target():
    """Uniform distribution: H = log(K). At τ = log(3) the loss is 0."""
    loss = EntropyLoss(target_entropy=math.log(3))
    uniform = torch.full((4, 3), 1 / 3)
    assert loss(uniform).item() == pytest.approx(0.0, abs=1e-6)


def test_entropy_loss_collapsed_distance_target():
    """One-hot: H = 0. With τ = 0 the loss should be 0."""
    onehot = torch.zeros(4, 3)
    onehot[:, 0] = 1.0
    loss = EntropyLoss(target_entropy=0.0)
    assert loss(onehot).item() == pytest.approx(0.0, abs=1e-6)


def test_entropy_loss_distance_to_target():
    """For onehot rows, H = 0 → distance to τ = |τ|."""
    onehot = torch.zeros(4, 3)
    onehot[:, 0] = 1.0
    loss = EntropyLoss(target_entropy=0.7)
    assert loss(onehot).item() == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# ReliabilityLoss
# ---------------------------------------------------------------------------


def test_reliability_loss_zero_when_r_matches_target():
    """When the model's r already equals r* exactly, KL(r* || r) = 0."""
    rel_loss = ReliabilityLoss(similarity_temperature=1.0)
    unimodal = torch.tensor([[0.5, 0.4, 0.6], [-0.2, 0.1, 0.0]])
    multimodal = torch.tensor([0.5, 0.0])
    deltas = (unimodal - multimodal.unsqueeze(-1)).abs()
    target = F.softmax(-deltas, dim=-1)
    # Use target as r: loss should be ~0
    loss = rel_loss(target, unimodal, multimodal)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_reliability_loss_positive_when_misaligned():
    rel_loss = ReliabilityLoss()
    unimodal = torch.tensor([[1.0, 0.0, -1.0]])
    multimodal = torch.tensor([1.0])
    # Target favors text (smallest delta). If we hand a uniform r, KL > 0.
    uniform = torch.full((1, 3), 1 / 3)
    assert rel_loss(uniform, unimodal, multimodal).item() > 0


# ---------------------------------------------------------------------------
# FaithfulnessLoss
# ---------------------------------------------------------------------------


def test_faithfulness_loss_classification(cls_model, synthetic_batch):
    out = cls_model(**{k: v for k, v in synthetic_batch.items() if k != "label"})
    loss = FaithfulnessLoss(task="classification", classification_metric="kl")(
        cls_model, synthetic_batch, out,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_faithfulness_loss_regression(reg_model, synthetic_batch):
    batch = {**synthetic_batch, "label": torch.tensor([0.5, -0.2, 1.0, -1.0])}
    inputs = {k: v for k, v in batch.items() if k != "label"}
    out = reg_model(**inputs)
    loss = FaithfulnessLoss(task="regression")(reg_model, batch, out)
    assert torch.isfinite(loss)


def test_faithfulness_loss_supports_all_classification_metrics(cls_model, synthetic_batch):
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    out = cls_model(**inputs)
    for metric in ("kl", "tvd", "prob_drop"):
        loss = FaithfulnessLoss(task="classification", classification_metric=metric)(
            cls_model, synthetic_batch, out,
        )
        assert torch.isfinite(loss), f"{metric} produced non-finite loss"


def test_faithfulness_loss_invalid_metric():
    with pytest.raises(ValueError):
        FaithfulnessLoss(task="classification", classification_metric="cosine")


# ---------------------------------------------------------------------------
# StabilityLoss
# ---------------------------------------------------------------------------


def test_stability_loss_nonnegative(cls_model, synthetic_batch):
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    out = cls_model(**inputs)
    loss = StabilityLoss(0.1, 0.1, 0.1)(cls_model, synthetic_batch, out)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_stability_loss_zero_perturbation_in_eval_mode_is_small(cls_model, synthetic_batch):
    """With zero perturbation strengths AND eval mode (no dropout), the
    perturbed forward equals the clean forward, so the L2 distances should be
    exactly zero."""
    cls_model.eval()
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    out = cls_model(**inputs)
    loss = StabilityLoss(0.0, 0.0, 0.0)(cls_model, synthetic_batch, out)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# XMoFELoss composite
# ---------------------------------------------------------------------------


def test_xmofe_loss_composite_classification(cls_model, synthetic_batch, loss_config):
    loss_fn = XMoFELoss.from_config(loss_config, task="classification")
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    out = cls_model(**inputs)
    total, components = loss_fn(cls_model, synthetic_batch, out)

    assert set(components.keys()) == {"task", "reliability", "faithfulness", "stability", "entropy"}
    for name, c in components.items():
        assert torch.isfinite(c), f"{name} component is not finite"
    # Reliability should be exactly zero (no unimodal labels in batch)
    assert components["reliability"].item() == 0.0
    assert torch.isfinite(total)


def test_xmofe_loss_composite_regression_with_unimodal(reg_model, synthetic_batch, loss_config):
    """CH-SIMS-like batch: regression task with unimodal_labels → all five
    components should be active and finite."""
    batch = {
        **synthetic_batch,
        "label": torch.tensor([0.5, -0.2, 1.0, -1.0]),
        "unimodal_labels": torch.tensor([
            [0.5, 0.4, 0.6],
            [-0.2, 0.1, 0.0],
            [1.0, 0.8, 0.9],
            [-1.0, -0.9, -1.0],
        ]),
    }
    loss_fn = XMoFELoss.from_config(loss_config, task="regression")
    inputs = {
        k: v for k, v in batch.items()
        if k in ("text", "audio", "visual", "text_length", "audio_length", "visual_length")
    }
    out = reg_model(**inputs)
    total, components = loss_fn(reg_model, batch, out)

    assert components["reliability"].item() > 0  # should be active now
    assert all(torch.isfinite(c) for c in components.values())
    assert torch.isfinite(total)


def test_xmofe_loss_backward_flows(cls_model, synthetic_batch, loss_config):
    loss_fn = XMoFELoss.from_config(loss_config, task="classification")
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    out = cls_model(**inputs)
    total, _ = loss_fn(cls_model, synthetic_batch, out)
    cls_model.zero_grad()
    total.backward()
    no_grad = [
        n for n, p in cls_model.named_parameters()
        if p.grad is None and "missing_token" not in n
    ]
    assert no_grad == []


def test_xmofe_loss_composite_value_sums_components(cls_model, synthetic_batch, loss_config):
    """The total must equal task + α·rel + β·faith + γ·stab + δ·ent."""
    loss_fn = XMoFELoss.from_config(loss_config, task="classification")
    inputs = {k: v for k, v in synthetic_batch.items() if k != "label"}
    torch.manual_seed(7)
    out = cls_model(**inputs)
    torch.manual_seed(7)
    total, components = loss_fn(cls_model, synthetic_batch, out)

    expected = (
        components["task"]
        + loss_fn.alpha * components["reliability"]
        + loss_fn.beta * components["faithfulness"]
        + loss_fn.gamma * components["stability"]
        + loss_fn.delta * components["entropy"]
    )
    assert torch.allclose(total, expected, atol=1e-5)


def test_xmofe_loss_class_weights_propagate(loss_config):
    weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    loss_fn = XMoFELoss.from_config(loss_config, task="classification", class_weights=weights)
    assert loss_fn.task_loss.criterion.weight is not None
    assert torch.allclose(loss_fn.task_loss.criterion.weight, weights)
