"""Run the spec §17 explainability evaluation suite for a trained checkpoint.

Computes, on the chosen split:

  * Modality-level faithfulness (§17.1) — Spearman, KL, top-1 between
    model reliability ``r`` and observed prediction sensitivity.
  * Temporal deletion (§17.2) per modality — mean sensitivity at
    k_fraction ∈ {10, 20, 30, 40, 50}% and AULC. Larger AULC ⇒ better.
  * Temporal insertion (§17.3) per modality — same x-axis, residual
    sensitivity when keeping only top-k. Smaller AULC ⇒ better.
  * Sufficiency / comprehensiveness (§17.4 + §17.5) per modality at the
    fixed 20% threshold.
  * CH-SIMS reliability alignment (§17.6) — auto-skipped when the dataset
    has no unimodal labels.

Writes a JSON report to ``results/explanations_<run>.json``.

Examples
--------
    python scripts/evaluate/evaluate_explanations.py \\
        --experiment mosei \\
        --checkpoint checkpoints/<run>/best.pt

    python scripts/evaluate/evaluate_explanations.py \\
        --experiment ch_sims --checkpoint checkpoints/<run>/best.pt \\
        --max-batches 5
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
from src.evaluation import (  # noqa: E402
    DEFAULT_K_FRACTION,
    DEFAULT_K_FRACTIONS,
    chsims_reliability_alignment,
    comprehensiveness,
    deletion_curve,
    insertion_curve,
    modality_faithfulness,
    sufficiency,
)
from src.models import VARIANTS, build_model  # noqa: E402
from src.training import load_checkpoint, set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_explanations")

VALID_MODALITIES = ("text", "audio", "visual")


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


class _DeviceLoader:
    """Wraps a DataLoader to move tensor batches to the model's device."""

    def __init__(self, loader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            yield {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

    def __len__(self) -> int:
        return len(self.loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="xmofe")
    parser.add_argument("--modality", choices=VALID_MODALITIES, default=None,
                        help="Required when --variant=unimodal.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap each sub-evaluation to N batches (smoke runs).")
    parser.add_argument("--k-fraction", type=float, default=DEFAULT_K_FRACTION,
                        help="Threshold used by sufficiency / comprehensiveness (default 0.2).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / "experiments" / f"{args.experiment}.yaml"
    config = load_yaml(config_path)
    model_config = load_yaml(REPO_ROOT / config["model_config"])

    set_seed(args.seed)
    device = resolve_device(args.device)

    manifest_path = REPO_ROOT / config["manifest_dir"] / f"{args.split}.pt"
    raw_loader = make_dataloader(manifest_path, batch_size=args.batch_size, shuffle=False)
    loader = _DeviceLoader(raw_loader, device)
    feature_dims = raw_loader.dataset.feature_dims
    log.info(
        "experiment=%s split=%s n=%d dims=text:%d audio:%d visual:%d device=%s",
        args.experiment, args.split, len(raw_loader.dataset),
        feature_dims["text"], feature_dims["audio"], feature_dims["visual"], device,
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

    results: dict[str, Any] = {
        "experiment": args.experiment,
        "variant": args.variant,
        "modality": args.modality,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "task": config["task"],
        "k_fraction": args.k_fraction,
        "k_fractions_curve": list(DEFAULT_K_FRACTIONS),
        "device": str(device),
        "max_batches": args.max_batches,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # ---- Modality-level faithfulness (§17.1) ------------------------
    log.info("modality faithfulness (§17.1)...")
    results["modality_faithfulness"] = modality_faithfulness(
        model, loader, task=config["task"], max_batches=args.max_batches,
    )

    # ---- Temporal deletion / insertion (§17.2 + §17.3) --------------
    deletion: dict[str, Any] = {}
    insertion: dict[str, Any] = {}
    for modality in VALID_MODALITIES:
        log.info("deletion %s (§17.2)...", modality)
        deletion[modality] = deletion_curve(
            model, loader, task=config["task"], modality=modality, max_batches=args.max_batches,
        )
        log.info("insertion %s (§17.3)...", modality)
        insertion[modality] = insertion_curve(
            model, loader, task=config["task"], modality=modality, max_batches=args.max_batches,
        )
    results["deletion"] = deletion
    results["insertion"] = insertion

    # ---- Sufficiency / comprehensiveness (§17.4 + §17.5) ------------
    suff: dict[str, Any] = {}
    comp: dict[str, Any] = {}
    for modality in VALID_MODALITIES:
        suff[modality] = sufficiency(
            model, loader, task=config["task"], modality=modality,
            k_fraction=args.k_fraction, max_batches=args.max_batches,
        )
        comp[modality] = comprehensiveness(
            model, loader, task=config["task"], modality=modality,
            k_fraction=args.k_fraction, max_batches=args.max_batches,
        )
    results["sufficiency"] = suff
    results["comprehensiveness"] = comp

    # ---- CH-SIMS reliability alignment (§17.6) ----------------------
    align = chsims_reliability_alignment(model, loader, max_batches=args.max_batches)
    if align is not None:
        results["chsims_reliability_alignment"] = align
        log.info(
            "CH-SIMS reliability alignment: spearman=%.3f kl=%.3f top1=%.3f",
            align["spearman"], align["kl_rstar_to_r_mean"], align["top1_agreement"],
        )
    else:
        log.info("dataset has no unimodal labels — skipping CH-SIMS reliability alignment")
        results["chsims_reliability_alignment"] = None

    # ---- Persist ----------------------------------------------------
    run_name = args.checkpoint.parent.name
    output_path = args.output or (REPO_ROOT / "results" / f"explanations_{run_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        f.write("\n")
    log.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
