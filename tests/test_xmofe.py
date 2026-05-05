"""Tests for the X-MoFE architecture.

Combines the Phase-3 smoke tests (forward/backward correctness, edge cases,
real-cache flow) with the additional checks needed before we touch the model
again in later phases: determinism, eval-mode dropout disable,
``state_dict`` roundtrip, and a sweep over batch sizes.

The real-cache test auto-skips when the MELD manifest hasn't been built, so
this file is also runnable in a fresh checkout without Phase-2 outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.models import XMoFE, XMoFEOutput

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "models" / "xmofe.yaml"
MELD_MANIFEST = REPO_ROOT / "data" / "processed" / "meld" / "train.pt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def meld_model() -> XMoFE:
    """MELD-shaped X-MoFE (768/768/768 inputs, 7-class classification)."""
    torch.manual_seed(0)
    return XMoFE(
        text_dim=768, audio_dim=768, visual_dim=768,
        shared_dim=256, attention_heads=4, dropout=0.2,
        use_trimodal=True, task="classification", num_classes=7,
    )


@pytest.fixture
def meld_batch() -> dict:
    torch.manual_seed(0)
    return {
        "text": torch.randn(4, 128, 768),
        "audio": torch.randn(4, 399, 768),
        "visual": torch.randn(4, 1568, 768),
        "text_length": torch.tensor([13, 50, 128, 7]),
        "audio_length": torch.tensor([100, 250, 399, 50]),
        "visual_length": torch.tensor([1568, 1568, 1568, 1568]),
    }


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


def test_output_type(meld_model, meld_batch):
    assert isinstance(meld_model(**meld_batch), XMoFEOutput)


def test_forward_shapes(meld_model, meld_batch):
    out = meld_model(**meld_batch)
    assert out.prediction.shape == (4, 7)
    assert out.reliability.shape == (4, 3)
    assert out.interactions.shape == (4, 4)
    assert out.temporal_attention["text"].shape == (4, 128)
    assert out.temporal_attention["audio"].shape == (4, 399)
    assert out.temporal_attention["visual"].shape == (4, 1568)
    assert out.fused.shape == (4, 256)


def test_softmax_distributions_valid(meld_model, meld_batch):
    out = meld_model(**meld_batch)
    assert torch.allclose(out.reliability.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out.interactions.sum(-1), torch.ones(4), atol=1e-5)
    assert (out.reliability >= 0).all()
    assert (out.interactions >= 0).all()


def test_interaction_names(meld_model, meld_batch):
    out = meld_model(**meld_batch)
    assert out.interaction_names == ("text_audio", "text_visual", "audio_visual", "trimodal")


# ---------------------------------------------------------------------------
# Backward / gradient flow
# ---------------------------------------------------------------------------


def test_backward_grads_flow_for_active_params(meld_model, meld_batch):
    meld_model.zero_grad()
    out = meld_model(**meld_batch)
    F.cross_entropy(out.prediction, torch.tensor([0, 3, 5, 1])).backward()
    # missing_token params legitimately have no grad when no sample triggers
    # the missing-modality path; exclude them from the strict check.
    no_grad = [
        n for n, p in meld_model.named_parameters()
        if p.grad is None and "missing_token" not in n
    ]
    assert no_grad == [], f"unexpected no-grad params: {no_grad}"


# ---------------------------------------------------------------------------
# Edge cases — missing modalities
# ---------------------------------------------------------------------------


def test_one_modality_missing_no_nan():
    torch.manual_seed(42)
    model = XMoFE(768, 768, 768, num_classes=7, task="classification")
    out = model(
        torch.randn(4, 128, 768),
        torch.randn(4, 399, 768),
        torch.randn(4, 1568, 768),
        torch.tensor([13, 50, 128, 7]),
        torch.tensor([0, 250, 399, 50]),       # sample 0 has no audio
        torch.tensor([1568, 1568, 1568, 1568]),
    )
    for tag, t in [
        ("prediction", out.prediction),
        ("reliability", out.reliability),
        ("interactions", out.interactions),
        *out.temporal_attention.items(),
    ]:
        assert not torch.isnan(t).any(), f"NaN in {tag}"
    # Missing-modality row's pooling weights are exactly zero.
    assert (out.temporal_attention["audio"][0] == 0).all()


def test_one_modality_missing_activates_missing_token_grad():
    torch.manual_seed(42)
    model = XMoFE(768, 768, 768, num_classes=7, task="classification")
    out = model(
        torch.randn(4, 128, 768),
        torch.randn(4, 399, 768),
        torch.randn(4, 1568, 768),
        torch.tensor([13, 50, 128, 7]),
        torch.tensor([0, 250, 399, 50]),
        torch.tensor([1568, 1568, 1568, 1568]),
    )
    F.cross_entropy(out.prediction, torch.tensor([0, 3, 5, 1])).backward()
    audio_token = dict(model.named_parameters())["pool_audio.missing_token"]
    assert audio_token.grad is not None
    assert audio_token.grad.abs().sum().item() > 0


def test_all_modalities_missing_for_one_sample():
    torch.manual_seed(42)
    model = XMoFE(768, 768, 768, num_classes=7, task="classification")
    out = model(
        torch.randn(4, 128, 768),
        torch.randn(4, 399, 768),
        torch.randn(4, 1568, 768),
        torch.tensor([13, 0, 128, 7]),
        torch.tensor([100, 0, 399, 50]),
        torch.tensor([1568, 0, 1568, 1568]),
    )
    for tag, t in [("prediction", out.prediction), ("reliability", out.reliability), ("interactions", out.interactions)]:
        assert not torch.isnan(t).any(), f"NaN in {tag}"
    # Reliability still a valid distribution for the all-missing row.
    assert torch.allclose(out.reliability.sum(-1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Heterogeneous input dims (MOSEI: 768 / 74 / 713)
# ---------------------------------------------------------------------------


def test_from_config_with_mosei_dims(config):
    model = XMoFE.from_config(
        config, text_dim=768, audio_dim=74, visual_dim=713,
        task="regression", num_classes=1,
    )
    out = model(
        torch.randn(2, 512, 768),
        torch.randn(2, 6000, 74),
        torch.randn(2, 3600, 713),
        torch.tensor([400, 200]),
        torch.tensor([5800, 6000]),
        torch.tensor([3000, 3500]),
    )
    assert out.prediction.shape == (2,)
    assert not torch.isnan(out.prediction).any()


# ---------------------------------------------------------------------------
# Determinism, eval mode, state_dict roundtrip
# ---------------------------------------------------------------------------


def _tiny_inputs():
    torch.manual_seed(0)
    return (
        torch.randn(2, 128, 768),
        torch.randn(2, 399, 768),
        torch.randn(2, 1568, 768),
        torch.tensor([10, 20]),
        torch.tensor([50, 100]),
        torch.tensor([200, 400]),
    )


def test_seed_determinism(config):
    """Identical seeds → identical model weights → identical outputs."""
    def build_and_run():
        torch.manual_seed(123)
        m = XMoFE.from_config(config, 768, 768, 768, task="classification", num_classes=7)
        m.eval()
        return m(*_tiny_inputs())
    out1 = build_and_run()
    out2 = build_and_run()
    assert torch.allclose(out1.prediction, out2.prediction, atol=1e-6)
    assert torch.allclose(out1.reliability, out2.reliability, atol=1e-6)
    assert torch.allclose(out1.interactions, out2.interactions, atol=1e-6)


def test_eval_mode_is_deterministic(config):
    """In eval mode, two forward passes on the same input must match exactly."""
    torch.manual_seed(0)
    model = XMoFE.from_config(config, 768, 768, 768, task="classification", num_classes=7)
    model.eval()
    inputs = _tiny_inputs()
    out1 = model(*inputs)
    out2 = model(*inputs)
    assert torch.allclose(out1.prediction, out2.prediction)
    assert torch.allclose(out1.reliability, out2.reliability)


def test_train_mode_dropout_changes_output():
    """In train mode with non-zero dropout, two forward passes differ."""
    torch.manual_seed(0)
    model = XMoFE(768, 768, 768, num_classes=7, task="classification", dropout=0.5)
    model.train()
    inputs = _tiny_inputs()
    out1 = model(*inputs)
    out2 = model(*inputs)
    # Dropout-driven divergence — not exact equality.
    assert not torch.allclose(out1.prediction, out2.prediction, atol=1e-3)


def test_state_dict_roundtrip(tmp_path, config):
    torch.manual_seed(0)
    model_a = XMoFE.from_config(config, 768, 768, 768, task="classification", num_classes=7)
    model_a.eval()

    path = tmp_path / "xmofe.pt"
    torch.save(model_a.state_dict(), path)

    model_b = XMoFE.from_config(config, 768, 768, 768, task="classification", num_classes=7)
    model_b.load_state_dict(torch.load(path, weights_only=True))
    model_b.eval()

    inputs = _tiny_inputs()
    out_a = model_a(*inputs)
    out_b = model_b(*inputs)
    assert torch.allclose(out_a.prediction, out_b.prediction)
    assert torch.allclose(out_a.reliability, out_b.reliability)
    assert torch.allclose(out_a.interactions, out_b.interactions)


# ---------------------------------------------------------------------------
# Batch-size sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_batch_size_sweep(batch_size, config):
    model = XMoFE.from_config(config, 768, 768, 768, task="classification", num_classes=7)
    model.eval()
    out = model(
        torch.randn(batch_size, 128, 768),
        torch.randn(batch_size, 399, 768),
        torch.randn(batch_size, 1568, 768),
        torch.full((batch_size,), 50, dtype=torch.long),
        torch.full((batch_size,), 200, dtype=torch.long),
        torch.full((batch_size,), 1000, dtype=torch.long),
    )
    assert out.prediction.shape == (batch_size, 7)
    assert out.reliability.shape == (batch_size, 3)
    assert out.interactions.shape == (batch_size, 4)


# ---------------------------------------------------------------------------
# Real cached features — auto-skip if caches not built
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MELD_MANIFEST.exists(), reason="MELD caches not built (run Phase 2 first)")
def test_real_meld_cache_flow(config):
    manifest = torch.load(MELD_MANIFEST, map_location="cpu", weights_only=False)
    dataset_dir = MELD_MANIFEST.parent
    caches = {
        mod: torch.load(dataset_dir / info["cache_path"], map_location="cpu", weights_only=False)
        for mod, info in manifest["modalities"].items()
    }
    idx = [0, 1, 2, 3]
    batch = {
        "text": caches["text"]["features"][idx].to(torch.float32),
        "audio": caches["audio"]["features"][idx].to(torch.float32),
        "visual": caches["visual"]["features"][idx].to(torch.float32),
        "text_length": caches["text"]["lengths"][idx].long(),
        "audio_length": caches["audio"]["lengths"][idx].long(),
        "visual_length": caches["visual"]["lengths"][idx].long(),
    }

    model = XMoFE.from_config(
        config,
        text_dim=manifest["modalities"]["text"]["feature_dim"],
        audio_dim=manifest["modalities"]["audio"]["feature_dim"],
        visual_dim=manifest["modalities"]["visual"]["feature_dim"],
        task=manifest["task"], num_classes=7,
    )
    out = model(**batch)
    assert out.prediction.shape == (4, 7)
    assert not torch.isnan(out.prediction).any()
    # Backward through cross-entropy on real labels.
    F.cross_entropy(out.prediction, manifest["primary_labels"][idx]).backward()
