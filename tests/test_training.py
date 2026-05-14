"""Tests for the X-MoFE training pipeline.

Covers seed determinism, checkpoint roundtrip, evaluator math, dataloader
shape sanity, and a single-step trainer smoke test on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.losses import XMoFELoss
from src.models import XMoFE
from src.training import (
    Evaluator,
    Trainer,
    TrainingLogger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MELD_MANIFEST = REPO_ROOT / "data" / "processed" / "meld" / "train.pt"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def test_set_seed_python_numpy_torch():
    set_seed(123)
    a_py = __import__("random").random()
    a_np = np.random.rand()
    a_t = torch.rand(3)

    set_seed(123)
    b_py = __import__("random").random()
    b_np = np.random.rand()
    b_t = torch.rand(3)

    assert a_py == b_py
    assert a_np == b_np
    assert torch.equal(a_t, b_t)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    model_a = XMoFE(768, 768, 768, num_classes=7, task="classification")
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    # Take one step so optimizer state is non-trivial
    dummy_loss = sum(p.sum() for p in model_a.parameters())
    dummy_loss.backward()
    optimizer_a.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path, model_a, optimizer_a,
        epoch=3, best_metric=0.42,
        extras={"wandb_run_id": "abc123"},
    )

    set_seed(99)  # different seed — verify load actually overrides
    model_b = XMoFE(768, 768, 768, num_classes=7, task="classification")
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
    payload = load_checkpoint(path, model_b, optimizer_b)

    assert payload["epoch"] == 3
    assert payload["best_metric"] == 0.42
    assert payload["extras"]["wandb_run_id"] == "abc123"

    # Models should now be parameter-identical
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), f"param {na} differs after load"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Returns a fixed output regardless of input — for evaluator unit tests."""

    def __init__(self, prediction, num_modalities: int = 3, num_interactions: int = 4) -> None:
        super().__init__()
        self.prediction = prediction
        self.reliability = torch.full((prediction.shape[0], num_modalities), 1 / num_modalities)
        self.interactions = torch.full((prediction.shape[0], num_interactions), 1 / num_interactions)

    def forward(self, **kwargs):  # noqa: ARG002
        from src.models import XMoFEOutput
        return XMoFEOutput(
            prediction=self.prediction,
            reliability=self.reliability,
            interactions=self.interactions,
            temporal_attention={"text": torch.zeros(1), "audio": torch.zeros(1), "visual": torch.zeros(1)},
        )


def test_evaluator_regression_metrics():
    pred = torch.tensor([1.0, -2.0, 0.5, -0.1])
    label = torch.tensor([0.5, -1.5, 1.0, 0.5])
    stub = _StubModel(pred)

    def loader():
        yield {
            "text": torch.zeros(4, 1, 1), "audio": torch.zeros(4, 1, 1), "visual": torch.zeros(4, 1, 1),
            "text_length": torch.zeros(4, dtype=torch.long),
            "audio_length": torch.zeros(4, dtype=torch.long),
            "visual_length": torch.zeros(4, dtype=torch.long),
            "label": label,
        }

    evaluator = Evaluator(task="regression", device="cpu")
    metrics = evaluator(stub, loader())

    expected_mae = float((pred - label).abs().mean())
    assert metrics["mae"] == pytest.approx(expected_mae, abs=1e-5)
    assert "pearson_r" in metrics
    assert 0.0 <= metrics["binary_acc"] <= 1.0


def test_evaluator_classification_metrics():
    # 4 samples, 3 classes — pred argmax = [2, 0, 1, 2], labels = [2, 0, 1, 1]
    pred = torch.tensor([
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.1, 0.8],
    ])
    label = torch.tensor([2, 0, 1, 1])
    stub = _StubModel(pred, num_modalities=3, num_interactions=4)

    def loader():
        yield {
            "text": torch.zeros(4, 1, 1), "audio": torch.zeros(4, 1, 1), "visual": torch.zeros(4, 1, 1),
            "text_length": torch.zeros(4, dtype=torch.long),
            "audio_length": torch.zeros(4, dtype=torch.long),
            "visual_length": torch.zeros(4, dtype=torch.long),
            "label": label,
        }

    evaluator = Evaluator(task="classification", num_classes=3, device="cpu")
    metrics = evaluator(stub, loader())
    assert metrics["accuracy"] == pytest.approx(0.75, abs=1e-6)  # 3/4 correct
    for i in range(3):
        assert f"f1_class_{i}" in metrics


# ---------------------------------------------------------------------------
# Dataloader (real-cache, auto-skip if not built)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MELD_MANIFEST.exists(), reason="MELD caches not built")
def test_dataloader_real_meld_shapes():
    from src.data import make_dataloader
    loader = make_dataloader(MELD_MANIFEST, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["text"].shape == (2, 128, 768)
    assert batch["text"].dtype == torch.float32
    assert batch["audio"].shape == (2, 399, 768)
    assert batch["visual"].shape == (2, 1568, 768)
    assert batch["text_length"].dtype == torch.long
    assert batch["label"].shape == (2,)
    assert "unimodal_labels" not in batch          # MELD has no unimodal labels
    assert isinstance(batch["sample_ids"], list)
    assert len(batch["sample_ids"]) == 2


# ---------------------------------------------------------------------------
# Trainer single-step smoke
# ---------------------------------------------------------------------------


def _make_trainer(tmp_path, *, precision: str = "fp32", device: str = "cpu"):
    """Build a fully-wired one-batch Trainer for precision/smoke tests."""
    set_seed(0)
    model = XMoFE(64, 64, 64, num_classes=3, task="classification", shared_dim=32, attention_heads=2, dropout=0.1)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.1, gamma=0.1, delta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = {
        "text": torch.randn(4, 8, 64),
        "audio": torch.randn(4, 8, 64),
        "visual": torch.randn(4, 8, 64),
        "text_length": torch.tensor([5, 8, 3, 7]),
        "audio_length": torch.tensor([4, 8, 6, 5]),
        "visual_length": torch.tensor([8, 7, 8, 6]),
        "label": torch.tensor([0, 2, 1, 0]),
    }

    class _OneBatchLoader:
        def __iter__(self):
            yield batch

        def __len__(self) -> int:
            return 1

    evaluator = Evaluator(task="classification", num_classes=3, device=device)
    logger = TrainingLogger(log_dir=tmp_path / "logs", run_name="t", config={}, use_wandb=False)
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        evaluator=evaluator,
        train_loader=_OneBatchLoader(), val_loader=_OneBatchLoader(),
        device=torch.device(device), logger=logger,
        checkpoint_dir=tmp_path / "ckpt",
        gradient_clip=1.0,
        early_stopping_metric="accuracy", early_stopping_mode="max",
        early_stopping_patience=10, log_every=1,
        precision=precision,
    )
    return trainer, logger


def test_trainer_single_step(tmp_path):
    """End-to-end: build everything, run 1 epoch with 1 batch, no crash."""
    trainer, logger = _make_trainer(tmp_path)
    result = trainer.train(num_epochs=1)
    logger.finish()

    assert "history" in result
    assert (tmp_path / "ckpt" / "latest.pt").exists()
    # best.pt is written if monitored metric is present in val_metrics
    assert (tmp_path / "ckpt" / "best.pt").exists()
    # File logger wrote at least one line
    assert (tmp_path / "logs" / "training.log").exists()
    assert (tmp_path / "logs" / "metrics.jsonl").exists()


# ---------------------------------------------------------------------------
# Precision / autocast wiring
# ---------------------------------------------------------------------------


def test_trainer_default_precision_is_fp32(tmp_path):
    trainer, logger = _make_trainer(tmp_path)
    try:
        assert trainer.precision == "fp32"
        assert trainer._autocast_enabled is False
    finally:
        logger.finish()


def test_trainer_invalid_precision_raises(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        _make_trainer(tmp_path, precision="fp16")


def test_trainer_bf16_autocast_runs_on_cpu(tmp_path):
    """bf16 autocast should be enabled on cpu and not crash forward/backward.

    CPU autocast supports bf16 in PyTorch, so this gives us coverage of the
    autocast wrapper without needing a GPU.
    """
    trainer, logger = _make_trainer(tmp_path, precision="bf16", device="cpu")
    try:
        assert trainer.precision == "bf16"
        assert trainer._autocast_enabled is True
        result = trainer.train(num_epochs=1)
        assert "history" in result
    finally:
        logger.finish()


# ---------------------------------------------------------------------------
# Class weights resolver (train_xmofe.resolve_class_weights)
# ---------------------------------------------------------------------------


def _import_resolve_class_weights():
    """Import the helper from the train script without invoking main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_xmofe", REPO_ROOT / "scripts" / "train" / "train_xmofe.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_class_weights


def test_resolve_class_weights_none_returns_none():
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 1, 2, 0, 1])
    assert fn(None, primary_labels=labels, num_classes=3, task="classification") is None


def test_resolve_class_weights_regression_task_returns_none():
    """Even with a non-null spec, regression silently skips weighting."""
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0.5, -0.2, 1.0])
    assert fn("inverse_frequency", primary_labels=labels, num_classes=1, task="regression") is None


def test_resolve_class_weights_inverse_frequency_balanced():
    """Balanced labels should produce uniform-ish weights summing to num_classes."""
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 1, 2, 0, 1, 2])  # 2 of each
    w = fn("inverse_frequency", primary_labels=labels, num_classes=3, task="classification")
    assert w.shape == (3,)
    assert torch.allclose(w, torch.ones(3))   # 6/(2*3) = 1.0 each


def test_resolve_class_weights_inverse_frequency_imbalanced():
    """A 70/20/10 split should give inverse-frequency weights in that order."""
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0] * 70 + [1] * 20 + [2] * 10)
    w = fn("inverse_frequency", primary_labels=labels, num_classes=3, task="classification")
    # weights = total / (count * num_classes) = 100 / (count * 3)
    assert w[0] == pytest.approx(100 / (70 * 3), abs=1e-5)
    assert w[1] == pytest.approx(100 / (20 * 3), abs=1e-5)
    assert w[2] == pytest.approx(100 / (10 * 3), abs=1e-5)
    # Rare class gets larger weight
    assert w[2] > w[1] > w[0]


def test_resolve_class_weights_inverse_frequency_handles_absent_class():
    """Class with zero examples should not blow up — weight clamps to ≤ total/num_classes."""
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 0, 0, 1, 1])  # class 2 is absent
    w = fn("inverse_frequency", primary_labels=labels, num_classes=3, task="classification")
    assert torch.isfinite(w).all()
    # Class 2 (count clamped to 1): weight = 5 / (1 * 3)
    assert w[2] == pytest.approx(5 / 3, abs=1e-5)


def test_resolve_class_weights_explicit_list():
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 1, 2])
    w = fn([1.0, 5.0, 0.5], primary_labels=labels, num_classes=3, task="classification")
    assert torch.equal(w, torch.tensor([1.0, 5.0, 0.5]))


def test_resolve_class_weights_unknown_string_raises():
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="class_weights"):
        fn("balanced", primary_labels=labels, num_classes=2, task="classification")


def test_resolve_class_weights_wrong_length_list_raises():
    fn = _import_resolve_class_weights()
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="length"):
        fn([1.0, 2.0], primary_labels=labels, num_classes=3, task="classification")


# ---------------------------------------------------------------------------
# Config sanity — Colab overlays must load cleanly and parity-match the
# default configs except for the knobs that should differ.
# ---------------------------------------------------------------------------


def test_loss_class_weighted_yaml_sets_inverse_frequency_flag():
    cfg = REPO_ROOT / "configs" / "training" / "loss_class_weighted.yaml"
    assert cfg.exists()
    with cfg.open() as f:
        data = yaml.safe_load(f)
    assert data.get("class_weights") == "inverse_frequency"
    # Auxiliary loss weights should match the default loss.yaml — the only
    # intentional difference is the class_weights field.
    base = REPO_ROOT / "configs" / "training" / "loss.yaml"
    with base.open() as f:
        base_data = yaml.safe_load(f)
    assert data["weights"] == base_data["weights"]


@pytest.mark.parametrize("name", [
    "meld_lr2e4", "meld_lr4e4", "meld_lr2e4_cw",
    "meld_lr2e4_cw_long", "meld_lr2e4_fp32",
    "meld_lr2e4_aux_lite", "meld_lr2e4_aux_lite_cw",
    "ch_sims_lr2e4", "ch_sims_lr4e4", "ch_sims_lr2e4_long",
    "ch_sims_lr2e4_aux_lite",
    "mosei_lr2e4", "mosei_lr2e4_aux_lite",
])
def test_colab_v2_sweep_configs_load(name):
    """v2 overlays must load and have the knobs they advertise."""
    cfg_path = REPO_ROOT / "configs" / "experiments" / "colab_v2" / f"{name}.yaml"
    assert cfg_path.exists()
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    assert "training" in cfg and "optimizer" in cfg["training"]
    lr = float(cfg["training"]["optimizer"]["lr"])
    if "lr2e4" in name:
        assert lr == pytest.approx(2.0e-4)
    elif "lr4e4" in name:
        assert lr == pytest.approx(4.0e-4)
    # Loss-config wiring — most-specific variant first to avoid false matches.
    if "aux_lite_cw" in name:
        assert cfg["loss_config"].endswith("loss_aux_lite_cw.yaml")
    elif "aux_lite" in name:
        assert cfg["loss_config"].endswith("loss_aux_lite.yaml")
    elif "_cw" in name:
        assert cfg["loss_config"].endswith("loss_class_weighted.yaml")
    if "_long" in name:
        assert cfg["training"]["num_epochs"] >= 50
        assert cfg["training"]["early_stopping"]["patience"] >= 10
    if "_fp32" in name:
        assert cfg["training"]["precision"] == "fp32"
    else:
        # Default for v2 sweep is bf16 — only the explicit fp32 variant differs.
        assert cfg["training"]["precision"] == "bf16"


@pytest.mark.parametrize("loss_file,expect_cw", [
    ("loss_aux_lite.yaml", False),
    ("loss_aux_lite_cw.yaml", True),
])
def test_loss_aux_lite_yaml_weights_are_lighter(loss_file, expect_cw):
    """Confirm aux_lite weights are lower than the default loss.yaml."""
    cfg = REPO_ROOT / "configs" / "training" / loss_file
    assert cfg.exists()
    with cfg.open() as f:
        data = yaml.safe_load(f)
    base = REPO_ROOT / "configs" / "training" / "loss.yaml"
    with base.open() as f:
        base_data = yaml.safe_load(f)
    for key in ("alpha", "beta", "gamma", "delta"):
        assert data["weights"][key] < base_data["weights"][key], (
            f"{loss_file}: {key}={data['weights'][key]} not lighter than "
            f"base {base_data['weights'][key]}"
        )
    if expect_cw:
        assert data.get("class_weights") == "inverse_frequency"
    else:
        assert "class_weights" not in data or data.get("class_weights") is None


@pytest.mark.parametrize("dataset", ["mosei", "meld", "ch_sims"])
def test_colab_config_loads_and_overrides_expected_fields(dataset):
    base = REPO_ROOT / "configs" / "experiments" / f"{dataset}.yaml"
    colab = REPO_ROOT / "configs" / "experiments" / "colab" / f"{dataset}.yaml"
    assert base.exists() and colab.exists()
    with base.open() as f:
        base_cfg = yaml.safe_load(f)
    with colab.open() as f:
        colab_cfg = yaml.safe_load(f)

    # Top-level fields that must match the base.
    for key in ("dataset", "manifest_dir", "model_config", "loss_config", "task", "num_classes"):
        assert colab_cfg.get(key) == base_cfg.get(key), f"{dataset}: {key} drifted from base config"
    # Run name_prefix must match so collected results bucket together.
    assert colab_cfg["run"]["name_prefix"] == base_cfg["run"]["name_prefix"]

    # Knobs the overlay specifically changes.
    assert colab_cfg["training"]["precision"] == "bf16"
    assert colab_cfg["training"]["batch_size"] >= base_cfg["training"]["batch_size"]


def test_trainer_bf16_falls_back_on_mps(tmp_path, monkeypatch):
    """bf16 on MPS should silently fall back to fp32 (autocast disabled)."""
    # Build the trainer with cpu device but tell it the device.type is "mps"
    # by faking the torch.device. Easiest: build with a cpu trainer then
    # check the fallback branch directly with a fresh construction.
    set_seed(0)
    model = XMoFE(64, 64, 64, num_classes=3, task="classification", shared_dim=32, attention_heads=2, dropout=0.1)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.1, gamma=0.1, delta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    evaluator = Evaluator(task="classification", num_classes=3, device="cpu")
    logger = TrainingLogger(log_dir=tmp_path / "logs", run_name="t", config={}, use_wandb=False)

    class _Empty:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    # We pass an actual mps-typed device. mps may or may not be available
    # at runtime; constructing torch.device("mps") doesn't require it to be.
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        evaluator=evaluator,
        train_loader=_Empty(), val_loader=_Empty(),
        device=torch.device("mps"), logger=logger,
        checkpoint_dir=tmp_path / "ckpt",
        gradient_clip=1.0,
        early_stopping_metric="accuracy", early_stopping_mode="max",
        early_stopping_patience=10, log_every=1,
        precision="bf16",
    )
    try:
        # Precision flag preserved, but autocast disabled to avoid the
        # unsupported mps bf16 path.
        assert trainer.precision == "bf16"
        assert trainer._autocast_enabled is False
    finally:
        logger.finish()


# ---------------------------------------------------------------------------
# Modality dropout (improvement #2 — train-time regularisation)
# ---------------------------------------------------------------------------


def _modality_dropout_trainer(tmp_path, *, p: float):
    """Build a one-batch Trainer with modality_dropout_p set to p."""
    set_seed(0)
    model = XMoFE(64, 64, 64, num_classes=3, task="classification",
                  shared_dim=32, attention_heads=2, dropout=0.1)
    loss_fn = XMoFELoss(task="classification", alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class _Empty:
        def __iter__(self): return iter([])
        def __len__(self): return 0

    evaluator = Evaluator(task="classification", num_classes=3, device="cpu")
    logger = TrainingLogger(log_dir=tmp_path / "logs", run_name="t", config={}, use_wandb=False)
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        evaluator=evaluator,
        train_loader=_Empty(), val_loader=_Empty(),
        device=torch.device("cpu"), logger=logger,
        checkpoint_dir=tmp_path / "ckpt",
        gradient_clip=1.0,
        early_stopping_metric="accuracy", early_stopping_mode="max",
        early_stopping_patience=10, log_every=1,
        modality_dropout_p=p,
    )
    return trainer, logger


def test_modality_dropout_default_is_zero(tmp_path):
    trainer, logger = _modality_dropout_trainer(tmp_path, p=0.0)
    try:
        assert trainer.modality_dropout_p == 0.0
    finally:
        logger.finish()


def test_modality_dropout_invalid_value_raises(tmp_path):
    with pytest.raises(ValueError, match="modality_dropout_p"):
        _modality_dropout_trainer(tmp_path, p=1.5)
    with pytest.raises(ValueError, match="modality_dropout_p"):
        _modality_dropout_trainer(tmp_path, p=-0.1)


def test_modality_dropout_zero_returns_batch_unchanged(tmp_path):
    """With p=0, _apply_modality_dropout must be a no-op."""
    trainer, logger = _modality_dropout_trainer(tmp_path, p=0.0)
    try:
        batch = {
            "text": torch.randn(4, 8, 64),
            "audio": torch.randn(4, 8, 64),
            "visual": torch.randn(4, 8, 64),
            "text_length": torch.tensor([5, 8, 3, 7]),
            "audio_length": torch.tensor([4, 8, 6, 5]),
            "visual_length": torch.tensor([8, 7, 8, 6]),
        }
        result = trainer._apply_modality_dropout(batch)
        for k in batch:
            assert torch.equal(result[k], batch[k]), f"batch[{k}] modified despite p=0"
    finally:
        logger.finish()


def test_modality_dropout_p1_always_zeros_one_modality(tmp_path):
    """With p=1, every call must zero exactly one of {text, audio, visual}."""
    trainer, logger = _modality_dropout_trainer(tmp_path, p=0.999)  # very near 1
    try:
        torch.manual_seed(42)
        for _ in range(20):  # multiple trials to exercise the random modality choice
            batch = {
                "text": torch.randn(4, 8, 64),
                "audio": torch.randn(4, 8, 64),
                "visual": torch.randn(4, 8, 64),
                "text_length": torch.tensor([5, 8, 3, 7]),
                "audio_length": torch.tensor([4, 8, 6, 5]),
                "visual_length": torch.tensor([8, 7, 8, 6]),
            }
            result = trainer._apply_modality_dropout(batch)
            zeroed_modalities = []
            for m in ("text", "audio", "visual"):
                if torch.all(result[m] == 0) and torch.all(result[f"{m}_length"] == 0):
                    zeroed_modalities.append(m)
            # Exactly one modality should be zeroed (the dropout-target).
            assert len(zeroed_modalities) == 1, (
                f"expected exactly 1 modality zeroed, got {zeroed_modalities}"
            )
    finally:
        logger.finish()


# ---------------------------------------------------------------------------
# Dialogue context (Lever 2 — MELD dialogue modeling)
# ---------------------------------------------------------------------------


def _build_synthetic_dialogue_manifest(tmp_path, *, with_dialogue_fields: bool = True):
    """Create a small synthetic manifest matching the patched format,
    with two dialogues of 3 + 2 utterances. Avoids needing real cached
    features by writing tiny per-modality cache files alongside.
    """
    import json as _json
    from src.data.features import write_feature_cache

    n = 5
    sample_ids = [f"dia0_utt{i}" for i in range(3)] + [f"dia1_utt{i}" for i in range(2)]
    transcripts = [
        "hi there",                      # dia0 utt0 Alice
        "hello back",                    # dia0 utt1 Bob
        "how is your day",               # dia0 utt2 Alice
        "totally agree",                 # dia1 utt0 Carol
        "me too",                        # dia1 utt1 Dave
    ]
    speakers = ["Alice", "Bob", "Alice", "Carol", "Dave"]
    dialogues = ["0", "0", "0", "1", "1"]
    utt_idxs  = [0, 1, 2, 0, 1]
    labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    feature_dim = 4
    L = 3
    feats = torch.zeros(n, L, feature_dim, dtype=torch.float32)
    lengths = torch.full((n,), L, dtype=torch.int32)

    base = tmp_path / "ch_sims"   # use a real dataset name so collate is happy
    (base / "text_features").mkdir(parents=True, exist_ok=True)
    (base / "audio_features").mkdir(parents=True, exist_ok=True)
    (base / "visual_features").mkdir(parents=True, exist_ok=True)
    for mod in ("text", "audio", "visual"):
        write_feature_cache(
            base / f"{mod}_features" / "test.pt",
            sample_ids, feats, lengths,
            {
                "dataset": "ch_sims", "split": "test",
                "modality": mod, "encoder_name": "stub", "encoder_source": "stub",
                "feature_dim": feature_dim, "max_length": L,
                "num_samples": len(sample_ids),
            },
        )

    manifest = {
        "dataset": "ch_sims",
        "split": "test",
        "task": "classification",
        "sample_ids": sample_ids,
        "primary_labels": labels,
        "labels": [{} for _ in range(n)],
        "modalities": {
            mod: {
                "cache_path": f"{mod}_features/test.pt",
                "feature_dim": feature_dim,
                "max_length": L,
            } for mod in ("text", "audio", "visual")
        },
        "transcripts": transcripts,
    }
    if with_dialogue_fields:
        manifest["speaker_ids"] = speakers
        manifest["dialogue_ids"] = dialogues
        manifest["utterance_indices"] = utt_idxs
    out_path = base / "test.pt"
    torch.save(manifest, out_path)
    return out_path


def test_dataset_context_window_zero_returns_raw_transcript(tmp_path):
    """Default behaviour: no context prepended."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path)
    ds = XMoFEDataset(p, context_window=0)
    sample = ds[2]   # dia0_utt2 (3rd utterance)
    assert sample["transcript"] == "how is your day"


def test_dataset_context_window_includes_prior_utterances(tmp_path):
    """With window=2, dia0_utt2 should see Alice + Bob's prior turns."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path)
    ds = XMoFEDataset(p, context_window=2)
    sample = ds[2]   # dia0_utt2 Alice "how is your day"
    text = sample["transcript"]
    assert "Alice: hi there" in text
    assert "Bob: hello back" in text
    # Current utterance also tagged with speaker.
    assert text.endswith("Alice: how is your day") or text.endswith("Alice: how is your day ")


def test_dataset_context_window_first_utterance_has_no_prior(tmp_path):
    """First utterance in a dialogue has empty prefix."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path)
    ds = XMoFEDataset(p, context_window=5)
    sample = ds[0]   # dia0_utt0 (first utterance)
    # No preceding utterances; transcript should be unchanged (no prefix).
    assert sample["transcript"] == "hi there"


def test_dataset_context_window_capped_to_available_history(tmp_path):
    """Window of 10 on a 1-prior-turn dialogue uses only the one available."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path)
    ds = XMoFEDataset(p, context_window=10)
    sample = ds[4]   # dia1_utt1 — only one prior turn (Carol)
    text = sample["transcript"]
    assert "Carol: totally agree" in text
    assert "Alice" not in text   # not even cross-dialogue leakage
    assert "Bob" not in text


def test_dataset_context_window_respects_dialogue_boundary(tmp_path):
    """Context is scoped to the SAME dialogue id; no leakage across dialogues."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path)
    ds = XMoFEDataset(p, context_window=5)
    sample = ds[3]   # dia1_utt0 — first turn in dia1
    text = sample["transcript"]
    # Even though dia0 has 3 utterances, none belong to dia1's context.
    assert "Alice" not in text
    assert "Bob" not in text
    assert text == "totally agree"


def test_dataset_context_window_silent_no_op_without_dialogue_fields(tmp_path):
    """When the manifest lacks dialogue_ids (legacy / non-dialogue dataset),
    context_window > 0 silently degrades to the raw-transcript behaviour."""
    from src.data import XMoFEDataset
    p = _build_synthetic_dialogue_manifest(tmp_path, with_dialogue_fields=False)
    ds = XMoFEDataset(p, context_window=5)
    sample = ds[2]
    assert sample["transcript"] == "how is your day"


def test_dataset_context_window_propagated_via_make_dataloader(tmp_path):
    """make_dataloader's context_window arg reaches the dataset."""
    from src.data import make_dataloader
    p = _build_synthetic_dialogue_manifest(tmp_path)
    loader = make_dataloader(p, batch_size=2, shuffle=False, context_window=3)
    assert loader.dataset.context_window == 3


def test_modality_dropout_validation_loader_unaffected(tmp_path):
    """The dropout helper is only called from the train loop. Verify the
    method itself respects mode by not being invoked on val batches; we
    do this indirectly by checking `_apply_modality_dropout` isn't called
    inside the evaluator."""
    # The Trainer's _validate calls self.evaluator(...) which iterates
    # val_loader directly — no modality dropout pathway. Quick structural
    # check: the helper is only referenced inside _train_epoch.
    import inspect
    from src.training import trainer as t_mod
    src = inspect.getsource(t_mod.Trainer)
    train_idx = src.index("def _train_epoch")
    validate_idx = src.index("def _validate")
    # _apply_modality_dropout must appear within _train_epoch but not _validate.
    train_block = src[train_idx:validate_idx]
    validate_block = src[validate_idx:]
    assert "_apply_modality_dropout" in train_block
    assert "_apply_modality_dropout" not in validate_block
