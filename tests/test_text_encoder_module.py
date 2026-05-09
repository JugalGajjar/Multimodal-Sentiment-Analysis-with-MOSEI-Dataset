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


# ---------------------------------------------------------------------------
# End-to-end integration: XMoFE.forward(transcripts=...) with fine-tunable
# text encoder. Uses bert-tiny (128-dim) so the test runs quickly.
# ---------------------------------------------------------------------------


def test_xmofe_with_text_encoder_forward_and_backward():
    """XMoFE with text_encoder_finetune=True consumes raw transcripts
    instead of cached text features and produces predictions.
    """
    from src.models import XMoFE
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = XMoFE(
        text_dim=128,    # ignored in fine-tune mode (overridden by encoder dim)
        audio_dim=64, visual_dim=64,
        num_classes=3, task="classification",
        shared_dim=32, attention_heads=2, dropout=0.1,
        text_encoder_finetune=True,
        text_encoder_name=TINY_MODEL,
        text_encoder_max_length=16,
        text_encoder_trainable_layers="last_n",
        text_encoder_last_n=1,
    )
    # The encoder should be present and the model should have absorbed the
    # encoder's hidden_size rather than the text_dim we passed.
    assert model.text_encoder is not None
    assert model.proj_text.proj.in_features == model.text_encoder.feature_dim

    transcripts = ["positive review one", "negative review two"]
    batch = {
        "audio": torch.randn(2, 8, 64),
        "visual": torch.randn(2, 8, 64),
        "audio_length": torch.tensor([6, 5]),
        "visual_length": torch.tensor([7, 8]),
    }
    out = model(transcripts=transcripts, **batch)
    assert out.prediction.shape == (2, 3)
    labels = torch.tensor([0, 2])
    F.cross_entropy(out.prediction, labels).backward()

    # The encoder's last layer must have received gradients through the loss.
    layer_list = model.text_encoder._find_layer_list()
    assert layer_list is not None
    last_layer_grads = [
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer_list[-1].parameters()
    ]
    assert all(last_layer_grads), "encoder's last layer should receive gradients"


def test_xmofe_finetune_mode_falls_back_to_cached_path_without_transcripts():
    """If no transcripts are passed, the fine-tune model should still be
    able to consume cached features. This matters at eval time when we
    might pass only cached features for speed."""
    from src.models import XMoFE

    torch.manual_seed(0)
    model = XMoFE(
        text_dim=128, audio_dim=64, visual_dim=64,
        num_classes=1, task="regression",
        shared_dim=32, attention_heads=2, dropout=0.1,
        text_encoder_finetune=True,
        text_encoder_name=TINY_MODEL,
        text_encoder_max_length=16,
        text_encoder_trainable_layers="none",
        text_encoder_last_n=1,
    )
    batch = {
        "text": torch.randn(2, 8, 128),
        "audio": torch.randn(2, 8, 64),
        "visual": torch.randn(2, 8, 64),
        "text_length": torch.tensor([5, 7]),
        "audio_length": torch.tensor([6, 5]),
        "visual_length": torch.tensor([7, 8]),
    }
    out = model(**batch)
    # Regression with num_classes=1 squeezes the trailing dim.
    assert out.prediction.shape == (2,)


def test_evaluator_forwards_transcripts_to_model():
    """Phase-1 regression guard: Evaluator must include `transcripts` in the
    keys it forwards to ``model(**inputs)`` so fine-tuned models actually
    use the in-graph encoder at val/test time. Without this, fine-tuned
    models would silently fall back to stale cached features."""
    import inspect
    from src.training import Evaluator
    src = inspect.getsource(Evaluator.__call__)
    assert '"transcripts"' in src, (
        "Evaluator.__call__ must whitelist `transcripts` in the inputs filter"
    )


def test_apply_missing_clears_transcripts_when_text_missing():
    """When the missing-modality protocol drops text AND the batch carries
    raw transcripts (fine-tune path), the transcripts must be emptied so
    the encoder doesn't re-introduce text via its forward pass."""
    from src.robustness import apply_missing

    batch = {
        "text": torch.zeros(2, 4, 8),
        "audio": torch.zeros(2, 4, 8),
        "visual": torch.zeros(2, 4, 8),
        "text_length": torch.tensor([4, 4]),
        "audio_length": torch.tensor([4, 4]),
        "visual_length": torch.tensor([4, 4]),
        "transcripts": ["this is the original text", "another original transcript"],
    }
    out = apply_missing(batch, ("text",))
    assert out["transcripts"] == ["", ""], (
        "transcripts must be emptied when text is dropped"
    )
    assert (out["text_length"] == 0).all()
    # Other modalities untouched.
    assert (out["audio_length"] == 4).all()
    assert (out["visual_length"] == 4).all()


def test_apply_missing_leaves_transcripts_alone_when_only_audio_or_visual_dropped():
    """Dropping audio or visual must not touch transcripts."""
    from src.robustness import apply_missing

    batch = {
        "text": torch.zeros(2, 4, 8),
        "audio": torch.zeros(2, 4, 8),
        "visual": torch.zeros(2, 4, 8),
        "text_length": torch.tensor([4, 4]),
        "audio_length": torch.tensor([4, 4]),
        "visual_length": torch.tensor([4, 4]),
        "transcripts": ["alpha", "beta"],
    }
    out_a = apply_missing(batch, ("audio",))
    out_v = apply_missing(batch, ("visual",))
    assert out_a["transcripts"] == ["alpha", "beta"]
    assert out_v["transcripts"] == ["alpha", "beta"]


def test_apply_missing_no_transcripts_field_is_no_op():
    """Backwards compat: if the batch does not carry transcripts (cached
    feature path / un-patched manifest), apply_missing must not crash."""
    from src.robustness import apply_missing

    batch = {
        "text": torch.zeros(2, 4, 8),
        "audio": torch.zeros(2, 4, 8),
        "visual": torch.zeros(2, 4, 8),
        "text_length": torch.tensor([4, 4]),
        "audio_length": torch.tensor([4, 4]),
        "visual_length": torch.tensor([4, 4]),
    }
    out = apply_missing(batch, ("text",))
    assert "transcripts" not in out


def test_split_encoder_params_partitions_correctly():
    """The optimizer-grouping helper in train_xmofe.py must put encoder
    params in one group and head params in another."""
    import importlib.util
    from src.models import XMoFE

    spec = importlib.util.spec_from_file_location(
        "train_xmofe", "scripts/train/train_xmofe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Model with fine-tune on
    m_ft = XMoFE(
        text_dim=128, audio_dim=64, visual_dim=64,
        num_classes=1, task="regression",
        shared_dim=32, attention_heads=2, dropout=0.1,
        text_encoder_finetune=True,
        text_encoder_name=TINY_MODEL,
        text_encoder_max_length=16,
        text_encoder_trainable_layers="last_n",
        text_encoder_last_n=1,
    )
    enc_params, head_params = mod._split_encoder_params(m_ft)
    assert len(enc_params) > 0
    assert len(head_params) > 0
    enc_ids = {id(p) for p in enc_params}
    head_ids = {id(p) for p in head_params}
    assert enc_ids.isdisjoint(head_ids), "param groups must be disjoint"

    # Model without fine-tune: encoder group should be empty
    m_frozen = XMoFE(
        text_dim=128, audio_dim=64, visual_dim=64,
        num_classes=1, task="regression",
        shared_dim=32, attention_heads=2, dropout=0.1,
        text_encoder_finetune=False,
    )
    enc_params2, head_params2 = mod._split_encoder_params(m_frozen)
    assert enc_params2 == []
    assert len(head_params2) > 0
