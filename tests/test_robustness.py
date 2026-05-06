"""Tests for the robustness corruption modules and protocols."""

from __future__ import annotations

import pytest
import torch

from src.robustness import (
    MISSING_CONDITIONS,
    SEVERITY_LEVELS,
    VALID_MODALITIES,
    apply_missing,
    apply_noise,
    audio_frame_dropout,
    audio_gaussian_noise,
    text_feature_noise,
    text_token_dropout,
    visual_gaussian_noise,
    visual_patch_dropout,
)


# ---------------------------------------------------------------------------
# Per-modality corruption functions
# ---------------------------------------------------------------------------


def test_token_dropout_zeros_some_positions_when_p_positive():
    torch.manual_seed(0)
    feats = torch.randn(2, 32, 16)
    out = text_token_dropout(feats, p=0.5)
    # Some token vectors should be all-zero (mask=False), others unchanged.
    zero_rows = (out.abs().sum(-1) == 0).any().item()
    assert zero_rows, "expected at least one dropped token vector"


def test_token_dropout_zero_p_is_identity():
    torch.manual_seed(0)
    feats = torch.randn(2, 32, 16)
    assert torch.equal(text_token_dropout(feats, p=0.0), feats)


def test_token_dropout_invalid_p_raises():
    feats = torch.randn(1, 4, 4)
    with pytest.raises(ValueError):
        text_token_dropout(feats, p=1.5)


def test_text_feature_noise_changes_values():
    torch.manual_seed(0)
    feats = torch.randn(2, 8, 4)
    out = text_feature_noise(feats, std=0.5)
    assert not torch.equal(feats, out)
    assert out.shape == feats.shape


def test_audio_gaussian_noise_zero_std_is_identity():
    torch.manual_seed(0)
    feats = torch.randn(2, 16, 4)
    assert torch.equal(audio_gaussian_noise(feats, std=0.0), feats)


def test_audio_gaussian_noise_changes_values_and_preserves_shape():
    torch.manual_seed(0)
    feats = torch.randn(2, 16, 4)
    out = audio_gaussian_noise(feats, std=0.5)
    assert out.shape == feats.shape
    assert (out - feats).abs().mean() > 0.05


def test_audio_frame_dropout_drops_frames():
    torch.manual_seed(0)
    feats = torch.randn(2, 32, 8)
    out = audio_frame_dropout(feats, p=0.5)
    zero_frames = (out.abs().sum(-1) == 0).any().item()
    assert zero_frames


def test_visual_patch_dropout_drops_patches():
    torch.manual_seed(0)
    feats = torch.randn(2, 64, 8)
    out = visual_patch_dropout(feats, p=0.4)
    zero_patches = (out.abs().sum(-1) == 0).any().item()
    assert zero_patches


def test_visual_gaussian_noise_negative_std_raises():
    feats = torch.randn(1, 4, 4)
    with pytest.raises(ValueError):
        visual_gaussian_noise(feats, std=-0.1)


def test_corruption_generator_makes_run_reproducible():
    feats = torch.randn(2, 32, 8)
    g1 = torch.Generator(device="cpu").manual_seed(7)
    g2 = torch.Generator(device="cpu").manual_seed(7)
    a = text_token_dropout(feats, p=0.3, generator=g1)
    b = text_token_dropout(feats, p=0.3, generator=g2)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Missing-modality protocol
# ---------------------------------------------------------------------------


def _synth_batch():
    torch.manual_seed(0)
    return {
        "text": torch.randn(4, 32, 16),
        "audio": torch.randn(4, 64, 16),
        "visual": torch.randn(4, 128, 16),
        "text_length": torch.tensor([10, 20, 32, 5]),
        "audio_length": torch.tensor([40, 64, 30, 20]),
        "visual_length": torch.tensor([100, 128, 64, 90]),
        "label": torch.tensor([0, 1, 2, 0]),
    }


def test_missing_conditions_count_matches_spec():
    # 1 clean + 3 single-missing + 3 pair-missing = 7
    assert len(MISSING_CONDITIONS) == 7
    assert MISSING_CONDITIONS[0] == ("clean", ())


def test_apply_missing_zeros_only_specified_lengths():
    batch = _synth_batch()
    out = apply_missing(batch, ["audio"])
    assert torch.equal(out["audio_length"], torch.zeros_like(batch["audio_length"]))
    # other lengths unchanged
    assert torch.equal(out["text_length"], batch["text_length"])
    assert torch.equal(out["visual_length"], batch["visual_length"])
    # features themselves are unmodified — only lengths are zeroed.
    assert torch.equal(out["audio"], batch["audio"])


def test_apply_missing_unknown_modality_raises():
    with pytest.raises(ValueError, match="unknown modality"):
        apply_missing(_synth_batch(), ["smell"])


def test_apply_missing_does_not_mutate_input():
    batch = _synth_batch()
    snapshot = batch["audio_length"].clone()
    _ = apply_missing(batch, ["audio"])
    assert torch.equal(batch["audio_length"], snapshot), "apply_missing mutated the input"


# ---------------------------------------------------------------------------
# Noisy-modality protocol
# ---------------------------------------------------------------------------


def test_severity_levels_have_three_modalities_each():
    for severity, levels in SEVERITY_LEVELS.items():
        assert set(levels) == set(VALID_MODALITIES), f"{severity}: {levels}"


def test_apply_noise_text_changes_text_only():
    batch = _synth_batch()
    out = apply_noise(batch, "text", severity="medium")
    assert not torch.equal(out["text"], batch["text"])
    assert torch.equal(out["audio"], batch["audio"])
    assert torch.equal(out["visual"], batch["visual"])


def test_apply_noise_audio_changes_audio_only():
    batch = _synth_batch()
    out = apply_noise(batch, "audio", severity="medium")
    assert torch.equal(out["text"], batch["text"])
    assert not torch.equal(out["audio"], batch["audio"])
    assert torch.equal(out["visual"], batch["visual"])


def test_apply_noise_visual_changes_visual_only():
    batch = _synth_batch()
    out = apply_noise(batch, "visual", severity="medium")
    assert torch.equal(out["text"], batch["text"])
    assert torch.equal(out["audio"], batch["audio"])
    assert not torch.equal(out["visual"], batch["visual"])


def test_apply_noise_invalid_severity():
    with pytest.raises(ValueError, match="severity"):
        apply_noise(_synth_batch(), "text", severity="extreme")


def test_apply_noise_invalid_modality():
    with pytest.raises(ValueError, match="modality"):
        apply_noise(_synth_batch(), "smell", severity="medium")


# ---------------------------------------------------------------------------
# Integration: run model under each condition and check sanity
# ---------------------------------------------------------------------------


def test_xmofe_handles_each_missing_condition():
    """Smoke check: model must produce finite outputs for every missing condition."""
    from src.models import XMoFE
    torch.manual_seed(0)
    model = XMoFE(text_dim=16, audio_dim=16, visual_dim=16, num_classes=3, task="classification")
    model.eval()
    batch = _synth_batch()
    inputs_keys = {"text", "audio", "visual", "text_length", "audio_length", "visual_length"}

    for name, drop in MISSING_CONDITIONS:
        modified = apply_missing(batch, drop)
        inputs = {k: v for k, v in modified.items() if k in inputs_keys}
        out = model(**inputs)
        assert torch.isfinite(out.prediction).all(), f"{name} produced NaN/inf"
        assert torch.isfinite(out.reliability).all(), f"{name} reliability not finite"


def test_xmofe_handles_each_noisy_condition():
    from src.models import XMoFE
    torch.manual_seed(0)
    model = XMoFE(text_dim=16, audio_dim=16, visual_dim=16, num_classes=3, task="classification")
    model.eval()
    batch = _synth_batch()
    inputs_keys = {"text", "audio", "visual", "text_length", "audio_length", "visual_length"}

    for severity in ("low", "medium", "high"):
        for modality in VALID_MODALITIES:
            modified = apply_noise(batch, modality, severity)
            inputs = {k: v for k, v in modified.items() if k in inputs_keys}
            out = model(**inputs)
            assert torch.isfinite(out.prediction).all(), f"{severity}/{modality} not finite"
