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
# Config sanity — Colab overlays must load cleanly and parity-match the
# default configs except for the knobs that should differ.
# ---------------------------------------------------------------------------


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
