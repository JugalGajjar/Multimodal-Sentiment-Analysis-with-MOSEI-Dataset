"""Tests for the controlled-fusion baselines and the model factory.

All baselines must:
  * Accept the same input signature as XMoFE
  * Return an XMoFEOutput with the right shapes
  * Backprop cleanly under the task-only loss
  * Expose a ``from_config`` constructor

Plus a check that ``XMoFELoss`` actually skips faithfulness/stability
forwards when those weights are 0 — the whole point of the task-only path
is to avoid those extra forward passes during baseline runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.losses import XMoFELoss
from src.models import (
    EarlyFusionModel,
    HybridFusionModel,
    LateFusionModel,
    UnimodalModel,
    XMoFEOutput,
    build_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO_ROOT / "configs" / "models" / "xmofe.yaml"


@pytest.fixture(scope="module")
def model_config() -> dict:
    with MODEL_CONFIG.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def synth_batch():
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


# ---------------------------------------------------------------------------
# Per-baseline forward + shape sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelCls,name", [
    (EarlyFusionModel, "early"),
    (LateFusionModel, "late"),
    (HybridFusionModel, "hybrid"),
])
def test_baseline_classification_forward(ModelCls, name, synth_batch, model_config):
    model = ModelCls.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    out = model(**inputs)
    assert isinstance(out, XMoFEOutput), f"{name}: expected XMoFEOutput"
    assert out.prediction.shape == (4, 7)
    assert out.reliability.shape == (4, 3)
    assert out.temporal_attention["text"].shape == (4, 128)
    # Reliability/interactions are valid distributions even when placeholder.
    assert torch.allclose(out.reliability.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out.interactions.sum(-1), torch.ones(4), atol=1e-5)


def test_unimodal_forward_each_modality(synth_batch, model_config):
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    for modality, expected_one_hot in (
        ("text", [1.0, 0.0, 0.0]),
        ("audio", [0.0, 1.0, 0.0]),
        ("visual", [0.0, 0.0, 1.0]),
    ):
        model = UnimodalModel.from_config(
            model_config, 768, 768, 768, task="classification", num_classes=7, modality=modality,
        )
        out = model(**inputs)
        assert out.prediction.shape == (4, 7)
        # Reliability is one-hot for the active modality
        assert torch.allclose(out.reliability[0], torch.tensor(expected_one_hot)), \
            f"{modality}: reliability not one-hot"


# ---------------------------------------------------------------------------
# Backward through baselines under task-only loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["early_fusion", "late_fusion", "hybrid_fusion"])
def test_baseline_backward_task_only(variant, synth_batch, model_config):
    model = build_model(variant, model_config, 768, 768, 768, task="classification", num_classes=7)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    out = model(**{k: v for k, v in synth_batch.items() if k != "label"})
    total, components = loss_fn(model, synth_batch, out)
    # Aux components must be exactly zero with weights=0
    assert components["faithfulness"].item() == 0.0
    assert components["stability"].item() == 0.0
    assert components["entropy"].item() == 0.0
    assert components["reliability"].item() == 0.0
    # And total == task loss alone
    assert torch.allclose(total, components["task"])
    total.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None and "missing_token" not in n]
    assert no_grad == [], f"unexpected no-grad params for {variant}: {no_grad}"


def test_unimodal_backward_only_active_modality_gets_grad(synth_batch, model_config):
    """Unimodal text-branch params should have grad; the audio/visual unused ones don't even exist."""
    model = build_model(
        "unimodal", model_config, 768, 768, 768,
        task="classification", num_classes=7, modality="text",
    )
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    out = model(**{k: v for k, v in synth_batch.items() if k != "label"})
    total, _ = loss_fn(model, synth_batch, out)
    total.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None and "missing_token" not in n]
    assert no_grad == []
    # Sanity: model only owns one set of projection weights, named "proj" — no audio/visual branches.
    param_names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith("proj.") for n in param_names)
    assert not any("audio" in n or "visual" in n for n in param_names), \
        "unimodal-text model unexpectedly contains audio/visual params"


# ---------------------------------------------------------------------------
# Loss-skip behaviour (the whole reason for adding zero-weight skipping)
# ---------------------------------------------------------------------------


class _CountingModel(torch.nn.Module):
    """Wraps a real model and counts how many times forward is called."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return self.inner(**kwargs)


def test_xmofe_loss_skips_aux_forwards_when_weights_are_zero(synth_batch, model_config):
    inner = build_model("early_fusion", model_config, 768, 768, 768, task="classification", num_classes=7)
    counting = _CountingModel(inner)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    out = counting(**{k: v for k, v in synth_batch.items() if k != "label"})
    # First forward (clean prediction) — counter should be 1 now.
    assert counting.calls == 1
    loss_fn(counting, synth_batch, out)
    # Task-only: no extra forwards beyond the clean one.
    assert counting.calls == 1, f"expected 1 forward, got {counting.calls}"


def test_xmofe_loss_runs_aux_forwards_when_weights_nonzero(synth_batch, model_config):
    inner = build_model("xmofe", model_config, 768, 768, 768, task="classification", num_classes=7)
    counting = _CountingModel(inner)
    # beta=0.3 → faithfulness runs (3 ablation forwards), gamma=0.1 → stability (1 forward)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.3, gamma=0.1, delta=0.05)
    out = counting(**{k: v for k, v in synth_batch.items() if k != "label"})
    assert counting.calls == 1
    loss_fn(counting, synth_batch, out)
    # 1 clean + 3 faithfulness ablations + 1 stability perturbation = 5 forwards total
    assert counting.calls == 5, f"expected 5 forwards, got {counting.calls}"


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_dispatches_correctly(model_config):
    expected = {
        "xmofe": "XMoFE",
        "early_fusion": "EarlyFusionModel",
        "late_fusion": "LateFusionModel",
        "hybrid_fusion": "HybridFusionModel",
    }
    for variant, cls_name in expected.items():
        m = build_model(variant, model_config, 768, 768, 768, task="regression", num_classes=1)
        assert m.__class__.__name__ == cls_name, f"{variant} -> {m.__class__.__name__}"
    for modality in ("text", "audio", "visual"):
        m = build_model(
            "unimodal", model_config, 768, 768, 768,
            task="regression", num_classes=1, modality=modality,
        )
        assert m.__class__.__name__ == "UnimodalModel"
        assert m.modality == modality


def test_factory_unimodal_requires_modality(model_config):
    with pytest.raises(ValueError, match="modality"):
        build_model("unimodal", model_config, 768, 768, 768, task="regression", num_classes=1)


def test_factory_unknown_variant_raises(model_config):
    with pytest.raises(ValueError, match="unknown variant"):
        build_model("transformer_fusion", model_config, 768, 768, 768, task="regression", num_classes=1)
