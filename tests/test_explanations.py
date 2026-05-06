"""Tests for the explainability evaluation suite (spec §17)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.evaluation import (
    chsims_reliability_alignment,
    comprehensiveness,
    deletion_curve,
    insertion_curve,
    keep_only_top_k_positions,
    kl_divergence,
    mask_top_k_positions,
    modality_faithfulness,
    prediction_sensitivity,
    spearman_correlation,
    sufficiency,
    top1_agreement,
    trapezoid_auc,
)
from src.models import XMoFE


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def test_kl_divergence_zero_when_equal():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert kl_divergence(p, p).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_positive_when_different():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.2, 0.3, 0.5]])
    assert kl_divergence(p, q).item() > 0


def test_spearman_perfect_increasing():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(x, y) == pytest.approx(1.0, abs=1e-6)


def test_spearman_perfect_decreasing():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert spearman_correlation(x, y) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_returns_nan_on_too_few_points():
    x = torch.tensor([1.0])
    y = torch.tensor([1.0])
    assert math.isnan(spearman_correlation(x, y))


def test_top1_agreement():
    p = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    q = torch.tensor([[0.0, 0.9, 0.1], [0.8, 0.1, 0.1]])
    assert top1_agreement(p, q) == pytest.approx(1.0)
    p2 = torch.tensor([[0.1, 0.7, 0.2]])
    q2 = torch.tensor([[0.7, 0.2, 0.1]])
    assert top1_agreement(p2, q2) == pytest.approx(0.0)


def test_trapezoid_auc_constant_function():
    # Flat y=0.5 over x=[0, 1] → area = 0.5
    assert trapezoid_auc([0.0, 1.0], [0.5, 0.5]) == pytest.approx(0.5)


def test_trapezoid_auc_handles_too_few_points():
    assert math.isnan(trapezoid_auc([0.0], [0.5]))


def test_prediction_sensitivity_regression():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([0.5, 3.0])
    out = prediction_sensitivity(a, b, task="regression")
    assert torch.allclose(out, torch.tensor([0.5, 1.0]))


def test_prediction_sensitivity_classification_zero_when_equal():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = prediction_sensitivity(logits, logits, task="classification")
    assert out.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Top-k masking primitives
# ---------------------------------------------------------------------------


def test_mask_top_k_zeros_attended_positions():
    feats = torch.ones(2, 8, 3)
    # Sample 0 has length 8, sample 1 has length 4
    lengths = torch.tensor([8, 4])
    # Attention concentrated at indices 0 and 7 for sample 0; 0 and 3 for sample 1
    attn = torch.zeros(2, 8)
    attn[0, 0] = 1.0
    attn[0, 7] = 0.9
    attn[1, 0] = 1.0
    attn[1, 3] = 0.9

    out = mask_top_k_positions(feats, attn, lengths, k_fraction=0.25)
    # Sample 0: 25% of 8 = 2 positions (indices 0, 7)
    assert (out[0, 0] == 0).all()
    assert (out[0, 7] == 0).all()
    assert (out[0, 1] == 1).all()  # untouched
    # Sample 1: 25% of 4 = 1 position (index 0)
    assert (out[1, 0] == 0).all()
    # Sample 1's index 3 is the second-highest; with k=1 only top-1 zeroed
    assert (out[1, 3] == 1).all()


def test_keep_only_top_k_inverts_mask():
    feats = torch.ones(1, 8, 3)
    lengths = torch.tensor([8])
    attn = torch.tensor([[0.9, 0.1, 0.05, 0.5, 0.0, 0.0, 0.7, 0.0]])
    out = keep_only_top_k_positions(feats, attn, lengths, k_fraction=0.5)
    # 50% of 8 = 4 → top-4 by attention: indices 0, 6, 3, 1 (in order)
    kept = (out[0].abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert sorted(kept) == [0, 1, 3, 6]


def test_mask_top_k_zero_fraction_is_identity():
    feats = torch.randn(2, 8, 3)
    attn = torch.rand(2, 8)
    lengths = torch.tensor([8, 8])
    assert torch.equal(mask_top_k_positions(feats, attn, lengths, 0.0), feats)


def test_mask_top_k_handles_zero_length_samples():
    feats = torch.ones(2, 8, 3)
    attn = torch.rand(2, 8)
    lengths = torch.tensor([4, 0])
    out = mask_top_k_positions(feats, attn, lengths, k_fraction=0.5)
    # Sample 1 (length 0) must not crash and must remain unchanged.
    assert torch.equal(out[1], feats[1])


# ---------------------------------------------------------------------------
# Integration with a tiny XMoFE model
# ---------------------------------------------------------------------------


def _tiny_model():
    torch.manual_seed(0)
    return XMoFE(
        text_dim=16, audio_dim=16, visual_dim=16,
        shared_dim=32, attention_heads=2, dropout=0.1,
        task="regression", num_classes=1,
    )


def _tiny_loader(num_batches: int = 2):
    torch.manual_seed(1)

    class _Loader:
        def __iter__(self):
            for _ in range(num_batches):
                yield {
                    "text": torch.randn(4, 8, 16),
                    "audio": torch.randn(4, 8, 16),
                    "visual": torch.randn(4, 8, 16),
                    "text_length": torch.tensor([5, 8, 3, 7]),
                    "audio_length": torch.tensor([4, 8, 6, 5]),
                    "visual_length": torch.tensor([8, 7, 8, 6]),
                    "label": torch.tensor([0.5, -0.2, 1.0, -1.0]),
                }

        def __len__(self):
            return num_batches

    return _Loader()


def test_deletion_curve_returns_expected_shape():
    model = _tiny_model()
    out = deletion_curve(model, _tiny_loader(2), task="regression", modality="text")
    assert out["modality"] == "text"
    assert len(out["k_fractions"]) == len(out["mean_sensitivity"])
    assert all(s >= 0 for s in out["mean_sensitivity"])
    assert isinstance(out["aulc"], float)
    assert out["n_samples"] > 0


def test_insertion_curve_returns_expected_shape():
    model = _tiny_model()
    out = insertion_curve(model, _tiny_loader(2), task="regression", modality="audio")
    assert out["modality"] == "audio"
    assert all(s >= 0 for s in out["mean_residual_sensitivity"])
    assert isinstance(out["aulc"], float)


def test_sufficiency_and_comprehensiveness_single_threshold():
    model = _tiny_model()
    suff = sufficiency(model, _tiny_loader(2), task="regression", modality="visual", k_fraction=0.2)
    comp = comprehensiveness(model, _tiny_loader(2), task="regression", modality="visual", k_fraction=0.2)
    assert suff["k_fraction"] == 0.2
    assert comp["k_fraction"] == 0.2
    assert suff["sufficiency"] >= 0
    assert comp["comprehensiveness"] >= 0


def test_modality_faithfulness_computes_summary():
    model = _tiny_model()
    out = modality_faithfulness(model, _tiny_loader(2), task="regression")
    assert out["n_samples"] > 0
    assert -1.0 <= out["spearman"] <= 1.0
    assert out["kl_s_to_r_mean"] >= 0
    assert 0.0 <= out["top1_agreement"] <= 1.0
    assert len(out["reliability_mean"]) == 3
    assert len(out["sensitivity_mean"]) == 3


def test_chsims_alignment_returns_none_without_unimodal_labels():
    model = _tiny_model()
    out = chsims_reliability_alignment(model, _tiny_loader(2))
    assert out is None


def test_chsims_alignment_with_unimodal_labels():
    model = _tiny_model()

    class _CHSimsLoader:
        def __iter__(self):
            yield {
                "text": torch.randn(4, 8, 16),
                "audio": torch.randn(4, 8, 16),
                "visual": torch.randn(4, 8, 16),
                "text_length": torch.tensor([5, 8, 3, 7]),
                "audio_length": torch.tensor([4, 8, 6, 5]),
                "visual_length": torch.tensor([8, 7, 8, 6]),
                "label": torch.tensor([0.5, -0.2, 1.0, -1.0]),
                "unimodal_labels": torch.tensor([
                    [0.5, 0.4, 0.6],
                    [-0.2, 0.1, 0.0],
                    [1.0, 0.8, 0.9],
                    [-1.0, -0.9, -1.0],
                ]),
            }
        def __len__(self): return 1

    out = chsims_reliability_alignment(model, _CHSimsLoader())
    assert out is not None
    assert out["n_samples"] == 4
    assert -1.0 <= out["spearman"] <= 1.0
    assert out["kl_rstar_to_r_mean"] >= 0
