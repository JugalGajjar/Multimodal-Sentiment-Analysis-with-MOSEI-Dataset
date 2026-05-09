"""Train X-MoFE on cached features for one dataset.

Reads the experiment config, loads the merged ``<split>.pt`` manifests
(already produced by Phase 2's ``merge_cached_features.py``), instantiates
the model + composite loss + optimizer + scheduler + evaluator, and runs the
trainer to completion. Logging goes to stdout, ``logs/<run>/training.log``,
JSONL metrics, and W&B (if ``WANDB_API_KEY`` is in ``.env`` or the
environment).

Examples
--------
    # Full training run
    python scripts/train/train_xmofe.py --experiment mosei

    # Override a few knobs at the CLI
    python scripts/train/train_xmofe.py --experiment meld --epochs 5 --batch-size 8

    # Smoke test — limit each epoch to N steps (no W&B)
    python scripts/train/train_xmofe.py --experiment mosei --max-steps 5 --no-wandb

    # Resume from a checkpoint
    python scripts/train/train_xmofe.py --experiment ch_sims --resume-from checkpoints/<run>/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import make_dataloader  # noqa: E402
from src.losses import XMoFELoss  # noqa: E402
from src.models import VARIANTS, build_model  # noqa: E402
from src.training import (  # noqa: E402
    Evaluator,
    Trainer,
    TrainingLogger,
    load_checkpoint,
    set_seed,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_class_weights(
    cw_spec,
    primary_labels: torch.Tensor,
    num_classes: int,
    task: str,
) -> torch.Tensor | None:
    """Translate a config-level ``class_weights`` spec into a tensor.

    Supports three forms:

    * ``None`` / missing → return ``None`` (no weighting).
    * ``"inverse_frequency"`` → compute from training labels at runtime.
      Weights normalized so ``sum(w) == num_classes`` (≡ uniform if balanced).
    * ``list[float]`` of length ``num_classes`` → use as-is.

    Silently returns ``None`` for regression tasks regardless of spec, so a
    single shared loss config can be used across regression/classification
    datasets without erroring.
    """
    if cw_spec is None:
        return None
    if task != "classification":
        return None
    if isinstance(cw_spec, str):
        if cw_spec.lower() != "inverse_frequency":
            raise ValueError(
                f"unknown class_weights spec {cw_spec!r}; "
                "expected 'inverse_frequency' or a list of floats"
            )
        labels = primary_labels.long()
        counts = torch.bincount(labels, minlength=num_classes).float()
        counts = counts.clamp(min=1.0)  # avoid /0 if a class is absent
        return labels.numel() / (counts * num_classes)
    if isinstance(cw_spec, (list, tuple)):
        if len(cw_spec) != num_classes:
            raise ValueError(
                f"class_weights list length {len(cw_spec)} != num_classes={num_classes}"
            )
        return torch.tensor(list(cw_spec), dtype=torch.float32)
    raise TypeError(
        f"class_weights must be None, 'inverse_frequency', or a list; got {type(cw_spec).__name__}"
    )


def _split_encoder_params(model: torch.nn.Module) -> tuple[list, list]:
    """Partition ``model.parameters()`` into (encoder_params, head_params).

    The text-encoder, when present, lives at ``model.text_encoder``. Returns
    two lists of parameters that PyTorch optimizers accept as separate
    parameter groups (so the encoder can get its own LR / weight decay).
    Encoder list is empty when the model has no text encoder.
    """
    encoder_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    text_enc = getattr(model, "text_encoder", None)
    if text_enc is None:
        head_params = [p for p in model.parameters() if p.requires_grad]
        return encoder_params, head_params
    encoder_param_set = set(id(p) for p in text_enc.parameters())
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in encoder_param_set:
            encoder_params.append(p)
        else:
            head_params.append(p)
    return encoder_params, head_params


def build_optimizer(
    name: str,
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    *,
    encoder_lr: float | None = None,
    encoder_weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer with optional separate encoder param group.

    When ``encoder_lr`` is provided AND the model owns a trainable text
    encoder, the encoder gets its own LR / weight decay group. Standard
    practice for fine-tuning: head LR (1e-4) is much higher than encoder LR
    (1e-5). When no encoder or no override, falls back to a single param
    group at the head LR — identical behaviour to the previous signature.
    """
    name = (name or "adamw").lower()
    encoder_params, head_params = _split_encoder_params(model)

    if encoder_lr is not None and encoder_params:
        groups = [
            {"params": head_params, "lr": float(lr),
             "weight_decay": float(weight_decay)},
            {"params": encoder_params, "lr": float(encoder_lr),
             "weight_decay": float(encoder_weight_decay if encoder_weight_decay is not None else weight_decay)},
        ]
    else:
        groups = [
            {"params": head_params + encoder_params, "lr": float(lr),
             "weight_decay": float(weight_decay)},
        ]

    if name == "adamw":
        return torch.optim.AdamW(groups)
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(name: str | None, optimizer, total_steps: int):
    if name is None or name == "none":
        return None
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(total_steps, 1),
        )
    raise ValueError(f"unknown scheduler: {name}")


class _MaxStepsLoader:
    """Wraps a DataLoader to yield at most ``max_steps`` batches per epoch.

    Used for the ``--max-steps`` smoke-test path so we can validate the full
    training pipeline without a multi-hour run.
    """

    def __init__(self, loader, max_steps: int) -> None:
        self.loader = loader
        self.max_steps = int(max_steps)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_steps:
                break
            yield batch

    def __len__(self) -> int:
        try:
            return min(len(self.loader), self.max_steps)
        except TypeError:
            return self.max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Name under configs/experiments/ (e.g. mosei).")
    parser.add_argument("--config", type=Path, default=None, help="Override config path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap each epoch to N batches — for smoke tests.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Checkpoint to resume from (restores model + optimizer + scheduler + epoch).")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this run.")
    parser.add_argument("--run-name", default=None, help="Override the auto-generated run name.")
    parser.add_argument(
        "--variant", choices=VARIANTS, default="xmofe",
        help="Model variant to train (default: xmofe).",
    )
    parser.add_argument(
        "--modality", choices=("text", "audio", "visual"), default=None,
        help="Required when --variant=unimodal.",
    )
    parser.add_argument(
        "--loss-config", type=Path, default=None,
        help="Override the loss config (e.g. configs/training/loss_task_only.yaml for baselines).",
    )
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default=None,
        help="Override training.precision. Use bf16 on A100/H100 for ~2x speedup.",
    )
    args = parser.parse_args()

    config_path = args.config or REPO_ROOT / "configs" / "experiments" / f"{args.experiment}.yaml"
    config = load_yaml(config_path)
    model_config = load_yaml(REPO_ROOT / config["model_config"])
    loss_config_path = args.loss_config or (REPO_ROOT / config["loss_config"])
    loss_config = load_yaml(loss_config_path)

    training_cfg = config["training"]
    if args.epochs is not None:
        training_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.precision is not None:
        training_cfg["precision"] = args.precision

    set_seed(args.seed)
    device = resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_tag = args.variant if args.variant == "xmofe" else f"{args.variant}{('_' + args.modality) if args.modality else ''}"
    run_name = args.run_name or f"{config['run']['name_prefix']}_{variant_tag}_{timestamp}"
    run_log_dir = REPO_ROOT / "logs" / run_name
    run_ckpt_dir = REPO_ROOT / "checkpoints" / run_name

    # Logger has to start first so we can route everything else through it.
    extras_for_run = {
        "experiment": args.experiment,
        "variant": args.variant,
        "modality": args.modality,
        "seed": args.seed,
        "device": str(device),
        "config_path": str(config_path),
        "loss_config_path": str(loss_config_path),
    }
    logger = TrainingLogger(
        log_dir=run_log_dir,
        run_name=run_name,
        config={"experiment": config, "model": model_config, "loss": loss_config, **extras_for_run},
        wandb_project=config["run"]["project"],
        use_wandb=not args.no_wandb,
    )
    logger.info(f"device={device}  config={config_path}")

    # ---- Data --------------------------------------------------------
    manifest_dir = REPO_ROOT / config["manifest_dir"]
    train_loader = make_dataloader(
        manifest_dir / "train.pt",
        batch_size=training_cfg["batch_size"], shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
    )
    val_loader = make_dataloader(
        manifest_dir / "val.pt",
        batch_size=training_cfg["batch_size"], shuffle=False,
        num_workers=training_cfg.get("num_workers", 0),
    )
    train_dataset = train_loader.dataset
    feature_dims = train_dataset.feature_dims
    logger.info(
        f"train={len(train_dataset)}  val={len(val_loader.dataset)}  "
        f"dims=text:{feature_dims['text']} audio:{feature_dims['audio']} visual:{feature_dims['visual']}"
    )

    # ---- Model + loss ------------------------------------------------
    model = build_model(
        variant=args.variant,
        config=model_config,
        text_dim=feature_dims["text"],
        audio_dim=feature_dims["audio"],
        visual_dim=feature_dims["visual"],
        task=config["task"],
        num_classes=config.get("num_classes", 1),
        modality=args.modality,
    ).to(device)
    class_weights = resolve_class_weights(
        loss_config.get("class_weights"),
        primary_labels=train_dataset.primary_labels,
        num_classes=int(config.get("num_classes", 1)),
        task=config["task"],
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"class_weights: {[round(float(w), 4) for w in class_weights.cpu().tolist()]}")
    loss_fn = XMoFELoss.from_config(
        loss_config, task=config["task"], class_weights=class_weights,
    ).to(device)
    logger.info(
        f"variant={args.variant}{(' modality=' + args.modality) if args.modality else ''}  "
        f"model params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ---- Optimizer + scheduler ---------------------------------------
    opt_cfg = training_cfg["optimizer"]
    encoder_lr = opt_cfg.get("encoder_lr")
    encoder_wd = opt_cfg.get("encoder_weight_decay")
    optimizer = build_optimizer(
        opt_cfg.get("name", "adamw"),
        model,
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        encoder_lr=float(encoder_lr) if encoder_lr is not None else None,
        encoder_weight_decay=float(encoder_wd) if encoder_wd is not None else None,
    )
    if any("text_encoder" in n for n, _ in model.named_parameters() if _.requires_grad):
        n_groups = len(optimizer.param_groups)
        logger.info(
            f"optimizer param groups: {n_groups}  "
            f"(head_lr={optimizer.param_groups[0]['lr']}, "
            f"encoder_lr={optimizer.param_groups[1]['lr'] if n_groups > 1 else 'shared'})"
        )

    if args.max_steps is not None:
        steps_per_epoch = min(len(train_loader), args.max_steps)
    else:
        steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * training_cfg["num_epochs"]
    scheduler = build_scheduler(
        (training_cfg.get("scheduler") or {}).get("name") if isinstance(training_cfg.get("scheduler"), dict) else training_cfg.get("scheduler"),
        optimizer, total_steps,
    )

    start_epoch = 0
    if args.resume_from is not None:
        payload = load_checkpoint(args.resume_from, model, optimizer, scheduler, map_location=str(device))
        start_epoch = int(payload.get("epoch", 0)) + 1
        logger.info(f"resumed from {args.resume_from} at epoch {start_epoch}")

    # ---- Evaluator ---------------------------------------------------
    evaluator = Evaluator(
        task=config["task"], num_classes=config.get("num_classes", 1), device=device,
    )

    # ---- Optional: cap each epoch to N batches -----------------------
    if args.max_steps is not None:
        train_loader_iter = _MaxStepsLoader(train_loader, args.max_steps)
        logger.info(f"smoke mode: capping each epoch to {args.max_steps} batches")
    else:
        train_loader_iter = train_loader

    # ---- Trainer -----------------------------------------------------
    es_cfg = training_cfg.get("early_stopping", {})
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        evaluator=evaluator,
        train_loader=train_loader_iter, val_loader=val_loader,
        device=device, logger=logger,
        checkpoint_dir=run_ckpt_dir,
        scheduler=scheduler,
        gradient_clip=float(training_cfg.get("gradient_clip", 1.0)),
        early_stopping_metric=es_cfg.get("metric", "mae"),
        early_stopping_mode=es_cfg.get("mode"),
        early_stopping_patience=int(es_cfg.get("patience", 5)),
        log_every=int(training_cfg.get("log_every", 50)),
        precision=str(training_cfg.get("precision", "fp32")),
        modality_dropout_p=float(training_cfg.get("modality_dropout_p", 0.0)),
    )

    try:
        trainer.train(num_epochs=training_cfg["num_epochs"], start_epoch=start_epoch)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
