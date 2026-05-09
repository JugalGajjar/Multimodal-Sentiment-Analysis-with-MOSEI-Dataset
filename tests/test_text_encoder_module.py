"""Tests for the trainable TextEncoderModule.

Uses ``prajjwal1/bert-tiny`` (4 MB, ~4M params) so unit tests don't
download the full ModernBERT-base. The architecture-agnostic logic
(trainability gating, gradient flow) is verified against this tiny model;
it'll behave identically against ModernBERT in production.
"""

from __future__ import annotations

import pytest
import torch

from src.encoders.text_encoder_module import TextEncoderModule

# Tiny encoder for fast tests. Has its own ``encoder.encoder.layer`` ModuleList.
TINY_MODEL = "prajjwal1/bert-tiny"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_full() -> TextEncoderModule:
    return TextEncoderModule(model_name=TINY_MODEL, max_length=16, trainable_layers="full")


@pytest.fixture(scope="module")
def tiny_last_n() -> TextEncoderModule:
    return TextEncoderModule(model_name=TINY_MODEL, max_length=16, trainable_layers="last_n", last_n=1)


@pytest.fixture(scope="module")
def tiny_none() -> TextEncoderModule:
    return TextEncoderModule(model_name=TINY_MODEL, max_length=16, trainable_layers="none")


def test_invalid_trainable_layers_raises():
    with pytest.raises(ValueError, match="trainable_layers"):
        TextEncoderModule(model_name=TINY_MODEL, trainable_layers="bogus")


def test_invalid_last_n_raises():
    with pytest.raises(ValueError, match="last_n"):
        TextEncoderModule(model_name=TINY_MODEL, trainable_layers="last_n", last_n=0)


# ---------------------------------------------------------------------------
# Trainability gating
# ---------------------------------------------------------------------------


def test_full_makes_all_params_trainable(tiny_full):
    assert all(p.requires_grad for p in tiny_full.encoder.parameters())
    assert tiny_full.num_trainable_params == sum(p.numel() for p in tiny_full.encoder.parameters())


def test_none_freezes_all_params(tiny_none):
    assert not any(p.requires_grad for p in tiny_none.encoder.parameters())
    assert tiny_none.num_trainable_params == 0


def test_last_n_freezes_most_unfreezes_top_n(tiny_last_n):
    """Only the last 1 transformer layer should be trainable.

    bert-tiny has 2 layers; with last_n=1 we expect roughly half the
    transformer-block params to be trainable, plus any final layer-norm.
    """
    full_total = sum(p.numel() for p in tiny_last_n.encoder.parameters())
    trainable = tiny_last_n.num_trainable_params
    assert 0 < trainable < full_total
    # Sanity: trainable count should be much smaller than full count.
    assert trainable / full_total < 0.7


# ---------------------------------------------------------------------------
# Forward shape contract
# ---------------------------------------------------------------------------


def test_forward_returns_features_and_lengths(tiny_last_n):
    transcripts = ["hello world", "this is a longer sample with more tokens"]
    features, lengths = tiny_last_n(transcripts)
    B, L, D = features.shape
    assert B == 2
    assert L == tiny_last_n.max_length
    assert D == tiny_last_n.feature_dim
    assert lengths.shape == (2,)
    assert lengths.dtype == torch.long
    # Lengths are unpadded token counts, so each must be in [1, L].
    assert (lengths >= 1).all() and (lengths <= L).all()


def test_forward_handles_empty_string():
    """Empty transcripts should still produce valid (CLS+SEP-only) features."""
    enc = TextEncoderModule(model_name=TINY_MODEL, max_length=16, trainable_layers="none")
    features, lengths = enc([""])
    assert features.shape == (1, 16, enc.feature_dim)
    # With most BERT tokenizers, empty string still produces CLS + SEP = 2 tokens.
    assert lengths.item() >= 2


# ---------------------------------------------------------------------------
# Gradient flow — the actual contract that matters for fine-tuning
# ---------------------------------------------------------------------------


def test_full_finetune_produces_gradients_on_almost_all_layers(tiny_full):
    """In full mode, every encoder param that participates in
    ``last_hidden_state`` must receive a gradient. The pooler / classification
    head sit on a different output (``pooler_output``) and so their params
    won't get gradients — that's expected, not a bug.
    """
    for p in tiny_full.encoder.parameters():
        p.grad = None
    transcripts = ["a positive review", "a negative review"]
    features, _ = tiny_full(transcripts)
    loss = features.mean()
    loss.backward()

    n_total = sum(1 for _ in tiny_full.encoder.parameters())
    n_with_grad = sum(
        1 for p in tiny_full.encoder.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    # ≥90% of parameters should receive gradients (only the unused
    # pooler/classification-head tail is excluded).
    assert n_with_grad / n_total >= 0.9, (
        f"only {n_with_grad}/{n_total} params got gradients in full mode"
    )


def test_last_n_only_produces_gradients_on_top_layers(tiny_last_n):
    # Reset existing grads first (fixture is module-scoped, may carry state).
    for p in tiny_last_n.encoder.parameters():
        p.grad = None

    transcripts = ["positive sentiment", "negative sentiment"]
    features, _ = tiny_last_n(transcripts)
    loss = features.mean()
    loss.backward()

    trainable_with_grad = 0
    trainable_total = 0
    frozen_with_grad = 0
    for p in tiny_last_n.encoder.parameters():
        if p.requires_grad:
            trainable_total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                trainable_with_grad += 1
        else:
            if p.grad is not None and p.grad.abs().sum() > 0:
                frozen_with_grad += 1

    assert trainable_total > 0, "fixture should have at least one trainable param"
    assert trainable_with_grad == trainable_total, (
        f"trainable params should all have grads: "
        f"{trainable_with_grad}/{trainable_total}"
    )
    assert frozen_with_grad == 0, "frozen params must not accumulate gradients"


def test_none_mode_produces_no_gradients_anywhere(tiny_none):
    for p in tiny_none.encoder.parameters():
        p.grad = None

    transcripts = ["text one", "text two"]
    features, _ = tiny_none(transcripts)
    loss = features.mean()
    # Backward still works (output requires_grad=False), but nothing should
    # accumulate on encoder params.
    if features.requires_grad:
        loss.backward()
    grads_anywhere = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in tiny_none.encoder.parameters())
    assert not grads_anywhere


# ---------------------------------------------------------------------------
# Device transfer (CPU here; same code path drives CUDA / MPS).
# ---------------------------------------------------------------------------


def test_module_can_be_moved_with_to(tiny_last_n):
    moved = tiny_last_n.to("cpu")
    assert next(moved.encoder.parameters()).device.type == "cpu"
    features, _ = moved(["device-transfer test"])
    assert features.device.type == "cpu"
