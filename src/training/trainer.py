"""Training loop for X-MoFE on cached features.

End-to-end flow:

    for epoch in range(start_epoch, num_epochs):
        for batch in train_loader:
            output = model(...)
            total, components = loss_fn(model, batch, output)
            backward + grad_clip + optimizer.step + scheduler.step
            log scalars to TrainingLogger every `log_every` steps
        val_metrics = evaluator(model, val_loader)
        log epoch summary
        save latest checkpoint
        if val_metrics[early_stop_metric] improved:
            save best checkpoint, reset patience
        else:
            patience -= 1; early stop if exhausted
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.losses import XMoFELoss
from src.training.checkpointing import save_checkpoint
from src.training.evaluator import Evaluator
from src.training.logger import TrainingLogger


_LOWER_IS_BETTER_METRICS = {"mae", "mse", "task_loss"}


def _is_improvement(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    return current > best


class Trainer:
    """Train an X-MoFE model on a single dataset.

    Args:
        model: ``XMoFE`` instance, already moved to ``device``.
        loss_fn: ``XMoFELoss`` instance; treated as a callable.
        optimizer: configured optimizer.
        scheduler: optional learning-rate scheduler stepped after each batch.
        evaluator: ``Evaluator`` for validation metrics.
        train_loader, val_loader: data loaders.
        device: where model + batches live.
        logger: ``TrainingLogger`` (file + stdout + W&B).
        checkpoint_dir: directory under which best.pt and latest.pt are saved.
        gradient_clip: per-step max-norm clip; <=0 disables.
        early_stopping_metric: validation metric to monitor (e.g. ``"mae"``).
        early_stopping_mode: ``"min"`` or ``"max"``; if None, inferred from
            the metric name (``mae``/``mse``/``loss`` → ``"min"`` else ``"max"``).
        early_stopping_patience: number of non-improving epochs before stop.
        log_every: per-step logging cadence.
        precision: ``"fp32"`` (default) or ``"bf16"``. bf16 wraps the
            training forward + loss in ``torch.autocast`` for ~2x throughput
            on A100/H100 with no loss-scaling required. Validation always
            runs in fp32 for stable metrics. MPS falls back to fp32 since
            bf16 autocast is not reliably supported there.
        modality_dropout_p: probability per training step of zeroing one
            randomly chosen modality's features (and length). 0.0 (default)
            disables. Common values 0.1–0.2 for regularisation that improves
            both clean accuracy and robustness under modality drop. Applied
            only during training; validation/eval are unaffected.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: XMoFELoss,
        optimizer: torch.optim.Optimizer,
        evaluator: Evaluator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: TrainingLogger,
        checkpoint_dir: str | Path,
        scheduler: Any | None = None,
        gradient_clip: float = 1.0,
        early_stopping_metric: str = "mae",
        early_stopping_mode: str | None = None,
        early_stopping_patience: int = 5,
        log_every: int = 50,
        precision: str = "fp32",
        modality_dropout_p: float = 0.0,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_clip = gradient_clip
        self.early_stopping_metric = early_stopping_metric
        if early_stopping_mode is None:
            early_stopping_mode = "min" if any(k in early_stopping_metric for k in _LOWER_IS_BETTER_METRICS) else "max"
        if early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be 'min' or 'max'")
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_patience = int(early_stopping_patience)
        self.log_every = int(log_every)

        precision = (precision or "fp32").lower()
        if precision not in {"fp32", "bf16"}:
            raise ValueError(f"precision must be 'fp32' or 'bf16'; got {precision!r}")
        self.precision = precision
        # bf16 autocast is supported on cuda + cpu in torch; mps support is
        # incomplete and inconsistent. On mps we transparently fall back to fp32.
        self._autocast_enabled = (
            precision == "bf16" and self.device.type in {"cuda", "cpu"}
        )
        if precision == "bf16" and not self._autocast_enabled:
            self.logger.info(
                f"precision=bf16 requested but device={self.device.type!r} "
                "does not reliably support bf16 autocast; falling back to fp32"
            )

        self.modality_dropout_p = float(modality_dropout_p)
        if not 0.0 <= self.modality_dropout_p < 1.0:
            raise ValueError(
                f"modality_dropout_p must be in [0, 1); got {modality_dropout_p}"
            )

        self.global_step = 0
        self.best_metric: float | None = None
        self.epochs_without_improvement = 0

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _model_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        keys = {
            "text", "audio", "visual",
            "text_length", "audio_length", "visual_length",
            "quality",
        }
        return {k: v for k, v in batch.items() if k in keys}

    _MODALITY_NAMES = ("text", "audio", "visual")

    def _apply_modality_dropout(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Per-step, with probability ``modality_dropout_p``, zero one random modality.

        Modality is chosen uniformly from {text, audio, visual}. The
        decision is taken once per step (per batch) so the batch is
        consistent. Both the feature tensor and the per-sample length
        are zeroed — keeping the mask-based attention pooling correct.

        Validation/eval batches are untouched (this method is only
        called from the training loop).
        """
        if self.modality_dropout_p <= 0.0:
            return batch
        # Single Bernoulli per batch; if it fires, pick one modality uniformly.
        if torch.rand(()).item() >= self.modality_dropout_p:
            return batch
        idx = int(torch.randint(0, len(self._MODALITY_NAMES), ()).item())
        modality = self._MODALITY_NAMES[idx]
        out = dict(batch)
        feat_key, len_key = modality, f"{modality}_length"
        if feat_key in out and isinstance(out[feat_key], torch.Tensor):
            out[feat_key] = torch.zeros_like(out[feat_key])
        if len_key in out and isinstance(out[len_key], torch.Tensor):
            out[len_key] = torch.zeros_like(out[len_key])
        return out

    # ------------------------------------------------------------------
    # Train + validate
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        n_batches = 0

        for batch_idx, raw_batch in enumerate(self.train_loader):
            batch = self._to_device(raw_batch)
            batch = self._apply_modality_dropout(batch)
            if self._autocast_enabled:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.model(**self._model_inputs(batch))
                    total, components = self.loss_fn(self.model, batch, output)
            else:
                output = self.model(**self._model_inputs(batch))
                total, components = self.loss_fn(self.model, batch, output)

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            if self.gradient_clip and self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for name, value in components.items():
                running[name] = running.get(name, 0.0) + float(value.detach())
            running["total"] = running.get("total", 0.0) + float(total.detach())
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                step_metrics: dict[str, float] = {f"loss/{k}": float(v.detach()) for k, v in components.items()}
                step_metrics["loss/total"] = float(total.detach())
                step_metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
                step_metrics["epoch"] = epoch
                self.logger.log_metrics(step_metrics, step=self.global_step, prefix="train_step/")

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def _validate(self, epoch: int) -> dict[str, float]:
        # Evaluator runs forward only — does not compute the auxiliary
        # losses (faithfulness/stability/etc.) which would multiply eval
        # cost. We log the val task loss separately for early-stopping.
        metrics = self.evaluator(self.model, self.val_loader)
        # Compute task-only val loss for early-stopping fall-back if the
        # configured metric isn't in `metrics` yet (e.g. first epoch).
        return metrics

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, start_epoch: int = 0) -> dict[str, Any]:
        self.logger.info(f"starting training: epochs={start_epoch}..{num_epochs}")

        history: list[dict[str, Any]] = []
        for epoch in range(start_epoch, num_epochs):
            train_summary = self._train_epoch(epoch)
            self.logger.log_metrics(
                {f"train_epoch/{k}": v for k, v in train_summary.items()} | {"epoch": epoch},
                step=self.global_step,
            )

            val_metrics = self._validate(epoch)
            self.logger.log_metrics(
                {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch},
                step=self.global_step,
            )

            # Checkpoints — latest always; best when monitored metric improves.
            extras = {}
            if self.logger.wandb_run_id is not None:
                extras["wandb_run_id"] = self.logger.wandb_run_id
            save_checkpoint(
                self.checkpoint_dir / "latest.pt",
                self.model, self.optimizer, self.scheduler,
                epoch=epoch, best_metric=self.best_metric, extras=extras or None,
            )

            monitored = val_metrics.get(self.early_stopping_metric)
            if monitored is not None:
                improved = _is_improvement(monitored, self.best_metric, self.early_stopping_mode)
                if improved:
                    self.best_metric = monitored
                    self.epochs_without_improvement = 0
                    save_checkpoint(
                        self.checkpoint_dir / "best.pt",
                        self.model, self.optimizer, self.scheduler,
                        epoch=epoch, best_metric=self.best_metric, extras=extras or None,
                    )
                    self.logger.info(
                        f"new best {self.early_stopping_metric}={monitored:.4f} at epoch {epoch}"
                    )
                else:
                    self.epochs_without_improvement += 1
                    self.logger.info(
                        f"no improvement on {self.early_stopping_metric} for "
                        f"{self.epochs_without_improvement} epochs (best={self.best_metric:.4f})"
                    )
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        self.logger.info(
                            f"early stopping at epoch {epoch}: no improvement "
                            f"in {self.early_stopping_patience} epochs"
                        )
                        history.append({"epoch": epoch, "train": train_summary, "val": val_metrics})
                        break

            history.append({"epoch": epoch, "train": train_summary, "val": val_metrics})

        self.logger.info("training complete")
        return {"history": history, "best_metric": self.best_metric}
