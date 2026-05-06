"""Run the spec §16 robustness battery for a trained checkpoint.

Loads the checkpoint, runs the test split through:

  * 7 missing-modality conditions (spec §16.1):
      clean, text-missing, audio-missing, visual-missing, and the three
      pair-missing combinations.

  * 3 noisy-modality conditions at the chosen severity (spec §16.2):
      text token-dropout, audio Gaussian noise, visual patch-dropout.

Writes a JSON report at ``results/robustness_<run_name>.json`` containing
per-condition metrics plus a summary block:

  * ``missing_avg_drop``        — average task-metric drop across the 6 missing conds
  * ``noisy_avg_drop``          — average task-metric drop across the 3 noisy conds
  * ``reliability_adaptation_<m>_missing`` — Δ on r_m when modality m is dropped
    (negative is the "expected behaviour" from spec §16.1: model should
    transfer trust away from a missing modality)

Examples
--------
    python scripts/evaluate/evaluate_robustness.py \\
        --experiment mosei \\
        --checkpoint checkpoints/<run>/best.pt

    python scripts/evaluate/evaluate_robustness.py \\
        --experiment ch_sims --checkpoint checkpoints/<run>/best.pt \\
        --severity high --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import make_dataloader  # noqa: E402
from src.models import VARIANTS, build_model  # noqa: E402
from src.robustness import (  # noqa: E402
    MISSING_CONDITIONS,
    SEVERITY_LEVELS,
    VALID_MODALITIES,
    apply_missing,
    apply_noise,
)
from src.training import Evaluator, load_checkpoint, set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_robustness")


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


class _TransformedLoader:
    """Wraps a DataLoader so each batch is transformed before yielding.

    Avoids materializing the full dataset and stays compatible with
    ``Evaluator``'s batch-by-batch iteration.
    """

    def __init__(self, loader, transform) -> None:
        self.loader = loader
        self.transform = transform

    def __iter__(self):
        for batch in self.loader:
            yield self.transform(batch)

    def __len__(self) -> int:
        return len(self.loader)


def primary_metric_for(task: str) -> tuple[str, str]:
    """Return ``(metric_name, mode)`` where mode is "min" or "max"."""
    if task == "regression":
        return "mae", "min"
    return "weighted_f1", "max"


def compute_drops(
    clean: dict[str, float],
    conditions: dict[str, dict[str, float]],
    metric: str,
    mode: str,
) -> list[float]:
    drops = []
    base = clean[metric]
    for cond, m in conditions.items():
        if metric not in m:
            continue
        delta = (m[metric] - base) if mode == "min" else (base - m[metric])
        drops.append(delta)
    return drops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Name under configs/experiments/.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="xmofe")
    parser.add_argument("--modality", choices=VALID_MODALITIES, default=None,
                        help="Required when --variant=unimodal.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--severity", choices=tuple(SEVERITY_LEVELS), default="medium")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path (default: results/robustness_<run>.json).")
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / "experiments" / f"{args.experiment}.yaml"
    config = load_yaml(config_path)
    model_config = load_yaml(REPO_ROOT / config["model_config"])

    set_seed(args.seed)
    device = resolve_device(args.device)

    manifest_path = REPO_ROOT / config["manifest_dir"] / f"{args.split}.pt"
    loader = make_dataloader(manifest_path, batch_size=args.batch_size, shuffle=False)
    feature_dims = loader.dataset.feature_dims
    log.info(
        "experiment=%s split=%s n=%d dims=text:%d audio:%d visual:%d",
        args.experiment, args.split, len(loader.dataset),
        feature_dims["text"], feature_dims["audio"], feature_dims["visual"],
    )

    model = build_model(
        variant=args.variant, config=model_config,
        text_dim=feature_dims["text"], audio_dim=feature_dims["audio"], visual_dim=feature_dims["visual"],
        task=config["task"], num_classes=config.get("num_classes", 1),
        modality=args.modality,
    ).to(device)
    payload = load_checkpoint(args.checkpoint, model, map_location=str(device))
    log.info(
        "loaded checkpoint epoch=%s best=%s",
        payload.get("epoch"), payload.get("best_metric"),
    )

    evaluator = Evaluator(task=config["task"], num_classes=config.get("num_classes", 1), device=device)
    metric_name, mode = primary_metric_for(config["task"])

    results: dict[str, Any] = {
        "experiment": args.experiment,
        "variant": args.variant,
        "modality": args.modality,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "severity": args.severity,
        "seed": args.seed,
        "device": str(device),
        "primary_metric": metric_name,
        "task": config["task"],
        "missing_modality": {},
        "noisy_modality": {},
        "summary": {},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # ---- Missing modality conditions --------------------------------
    log.info("running missing-modality conditions: %d", len(MISSING_CONDITIONS))
    for name, drop in MISSING_CONDITIONS:
        wrapped = _TransformedLoader(loader, lambda batch, d=drop: apply_missing(batch, d))
        metrics = evaluator(model, wrapped)
        results["missing_modality"][name] = metrics
        log.info("[missing/%s] %s=%.4f", name, metric_name, metrics.get(metric_name, float("nan")))

    # ---- Noisy modality conditions ----------------------------------
    log.info("running noisy-modality conditions at severity=%s", args.severity)
    # Use a stable seed so re-runs are reproducible per modality.
    for modality in VALID_MODALITIES:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        wrapped = _TransformedLoader(
            loader,
            lambda batch, m=modality, g=gen: apply_noise(batch, m, args.severity, generator=g),
        )
        metrics = evaluator(model, wrapped)
        key = f"{modality}_{args.severity}"
        results["noisy_modality"][key] = metrics
        log.info("[noisy/%s] %s=%.4f", key, metric_name, metrics.get(metric_name, float("nan")))

    # ---- Summary block ----------------------------------------------
    clean = results["missing_modality"]["clean"]
    missing_drops = compute_drops(
        clean,
        {k: v for k, v in results["missing_modality"].items() if k != "clean"},
        metric_name, mode,
    )
    noisy_drops = compute_drops(clean, results["noisy_modality"], metric_name, mode)

    results["summary"]["missing_avg_drop"] = (
        sum(missing_drops) / len(missing_drops) if missing_drops else 0.0
    )
    results["summary"]["noisy_avg_drop"] = (
        sum(noisy_drops) / len(noisy_drops) if noisy_drops else 0.0
    )

    # Reliability adaptation: Δ on r_m when modality m is dropped.
    for m in VALID_MODALITIES:
        clean_r = clean.get(f"reliability_mean_{m}")
        miss_r = results["missing_modality"][f"{m}_missing"].get(f"reliability_mean_{m}")
        if clean_r is not None and miss_r is not None:
            results["summary"][f"reliability_adaptation_{m}_missing"] = miss_r - clean_r

    # ---- Persist -----------------------------------------------------
    run_name = args.checkpoint.parent.name
    output_path = args.output or (REPO_ROOT / "results" / f"robustness_{run_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        f.write("\n")
    log.info("wrote %s", output_path)
    log.info(
        "summary: missing_avg_drop=%.4f noisy_avg_drop=%.4f",
        results["summary"]["missing_avg_drop"],
        results["summary"]["noisy_avg_drop"],
    )


if __name__ == "__main__":
    main()
