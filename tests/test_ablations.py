"""Tests for the X-MoFE component-removal ablations.

Each architectural ablation must:
  * Forward cleanly with the same input signature as XMoFE
  * Drop the parameters of the removed component (smaller param count)
  * Backprop only through the components that remain
  * Produce a valid XMoFEOutput (uniform/placeholder where the component is gone)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.losses import XMoFELoss
from src.models import (
    XMoFE,
    XMoFENoInteraction,
    XMoFENoReliability,
    XMoFENoTrimodal,
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


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Forward + shape sanity for each architectural ablation
# ---------------------------------------------------------------------------


def test_no_reliability_forward_uniform_r(synth_batch, model_config):
    model = XMoFENoReliability.from_config(
        model_config, 768, 768, 768, task="classification", num_classes=7,
    )
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    out = model(**inputs)
    assert isinstance(out, XMoFEOutput)
    assert out.prediction.shape == (4, 7)
    # Reliability is exactly uniform 1/3 — not learned at all.
    assert torch.allclose(out.reliability, torch.full((4, 3), 1.0 / 3.0), atol=1e-6)
    # Interactions remain a valid 4-way distribution (tri-modal still on by default).
    assert out.interactions.shape == (4, 4)
    assert torch.allclose(out.interactions.sum(-1), torch.ones(4), atol=1e-5)


def test_no_interaction_forward_zero_i(synth_batch, model_config):
    model = XMoFENoInteraction.from_config(
        model_config, 768, 768, 768, task="classification", num_classes=7,
    )
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    out = model(**inputs)
    assert out.prediction.shape == (4, 7)
    # Reliability is still learned and valid.
    assert torch.allclose(out.reliability.sum(-1), torch.ones(4), atol=1e-5)
    # Interactions placeholder shape (B, 1) since cross-modal blocks are gone.
    assert out.interactions.shape == (4, 1)
    # And the placeholder is exactly 1.0 (not informative — just plumbing).
    assert torch.allclose(out.interactions, torch.ones(4, 1), atol=1e-5)


def test_no_trimodal_forward_three_interactions(synth_batch, model_config):
    model = XMoFENoTrimodal.from_config(
        model_config, 768, 768, 768, task="classification", num_classes=7,
    )
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    out = model(**inputs)
    assert out.prediction.shape == (4, 7)
    # 3 interactions instead of 4 (no c_TAV).
    assert out.interactions.shape == (4, 3)
    assert out.interaction_names == ("text_audio", "text_visual", "audio_visual")
    assert torch.allclose(out.interactions.sum(-1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Param count: each ablation must shed parameters relative to full XMoFE
# ---------------------------------------------------------------------------


def test_param_count_ordering(model_config):
    full = XMoFE.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    no_rel = XMoFENoReliability.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    no_inter = XMoFENoInteraction.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    no_tri = XMoFENoTrimodal.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)

    # Each ablation drops at least the corresponding component's parameters.
    assert _param_count(no_rel) < _param_count(full)
    assert _param_count(no_inter) < _param_count(full)
    assert _param_count(no_tri) < _param_count(full)
    # No-interaction is the most-stripped (drops 3 cross-attn blocks + estimator + tri).
    assert _param_count(no_inter) < _param_count(no_tri)


# ---------------------------------------------------------------------------
# Backward: gradients flow only through retained components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["xmofe_no_reliability", "xmofe_no_interaction", "xmofe_no_trimodal"])
def test_ablation_backward_flows(variant, synth_batch, model_config):
    model = build_model(variant, model_config, 768, 768, 768, task="classification", num_classes=7)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    inputs = {k: v for k, v in synth_batch.items() if k != "label"}
    out = model(**inputs)
    total, _ = loss_fn(model, synth_batch, out)
    total.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None and "missing_token" not in n]
    assert no_grad == [], f"unexpected no-grad params for {variant}: {no_grad}"


def test_no_reliability_has_no_reliability_estimator_params(model_config):
    model = XMoFENoReliability.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith("reliability.") for n in names), \
        "no_reliability ablation still owns reliability MLP params"


def test_no_interaction_has_no_cross_modal_params(model_config):
    model = XMoFENoInteraction.from_config(model_config, 768, 768, 768, task="classification", num_classes=7)
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith(("cross_ta.", "cross_tv.", "cross_av.", "cross_tav.")) for n in names)
    assert not any(n.startswith("interaction_estimator.") for n in names)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,cls_name", [
    ("xmofe_no_reliability", "XMoFENoReliability"),
    ("xmofe_no_interaction", "XMoFENoInteraction"),
    ("xmofe_no_trimodal", "XMoFENoTrimodal"),
])
def test_factory_dispatches_ablations(variant, cls_name, model_config):
    m = build_model(variant, model_config, 768, 768, 768, task="regression", num_classes=1)
    assert m.__class__.__name__ == cls_name


# ---------------------------------------------------------------------------
# Loss-ablation configs are valid YAML and actually flip the right weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name,zeroed", [
    ("loss_no_faithfulness.yaml", "beta"),
    ("loss_no_stability.yaml", "gamma"),
    ("loss_no_entropy.yaml", "delta"),
    ("loss_no_reliability.yaml", "alpha"),
])
def test_loss_ablation_configs_zero_only_intended_weight(config_name, zeroed):
    path = REPO_ROOT / "configs" / "training" / config_name
    with path.open() as f:
        cfg = yaml.safe_load(f)
    weights = cfg["weights"]
    assert weights[zeroed] == 0.0, f"{config_name} expected {zeroed}=0.0"
    other_weights = [k for k in weights if k != zeroed]
    assert all(weights[k] > 0.0 for k in other_weights), \
        f"{config_name} should only zero {zeroed}, found {weights}"
