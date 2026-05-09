"""Run robustness + explanation eval on every checkpoint, batched by dataset.

The single-checkpoint scripts (``evaluate_robustness.py`` and
``evaluate_explanations.py``) each rebuild the test loader on every call.
For MELD that's ~7 min of cache reload per invocation; over 50+
checkpoints the redundant reloads dominate total wall time.

This wrapper:

  * Discovers all ``checkpoints/*/best.pt`` paths.
  * Parses each run name into ``(dataset, variant, modality)``.
  * Groups checkpoints by dataset.
  * For each dataset: builds the test loader **once**, then iterates over
    every checkpoint in the group — building model, loading weights,
    running both evals, writing JSONs, freeing the model.
  * Skips checkpoints whose output JSONs already exist (resumable).

Examples
--------
    # Full sweep — robustness + explanations on every checkpoint
    python scripts/evaluate/run_eval_matrix.py

    # Robustness only (skip explanations)
    python scripts/evaluate/run_eval_matrix.py --skip-explanations

    # Restrict to specific datasets / variants
    python scripts/evaluate/run_eval_matrix.py --datasets meld --variants xmofe_no_interaction

    # Smoke (cap eval batches; useful for sanity check)
    python scripts/evaluate/run_eval_matrix.py --max-batches 3
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import make_dataloader  # noqa: E402
from src.models import VARIANTS, build_model  # noqa: E402
from src.training import load_checkpoint, set_seed  # noqa: E402

# Reuse the function-form evals we extracted from the single-checkpoint scripts.
from scripts.evaluate.evaluate_robustness import run_robustness_eval  # noqa: E402
from scripts.evaluate.evaluate_explanations import run_explanations_eval  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_eval_matrix")

DATASETS = ("ch_sims", "meld", "mosei")
ARCH_ABLATIONS = {"xmofe_no_reliability", "xmofe_no_interaction", "xmofe_no_trimodal"}
LOSS_ABLATION_VARIANTS = {
    "xmofe_no_faithfulness", "xmofe_no_stability",
    "xmofe_no_entropy", "xmofe_no_reliability_loss",
}
FUSION_BASELINES = {"early_fusion", "late_fusion", "hybrid_fusion"}


def parse_run_name(run_name: str) -> tuple[str, str, str | None]:
    """Translate a run name into ``(experiment, variant, modality)``.

    Run-name conventions emitted by ``run_matrix.py``:

    * ``{ds}_xmofe_s{seed}``                                    → variant=xmofe
    * ``{ds}_xmofe_no_{reliability,interaction,trimodal}_s{seed}`` → tier-3 architectural ablations
    * ``{ds}_xmofe_no_{faithfulness,stability,entropy}_s{seed}`` → tier-4 loss ablations (variant=xmofe)
    * ``{ds}_xmofe_no_reliability_loss_s{seed}``                → tier-4 loss ablation (variant=xmofe)
    * ``{ds}_unimodal_{text,audio,visual}_s{seed}``             → variant=unimodal, modality=...
    * ``{ds}_{early,late,hybrid}_fusion_s{seed}``               → variant=...fusion
    """
    base = re.sub(r"_s\d+$", "", run_name)
    for ds in DATASETS:
        prefix = f"{ds}_"
        if base.startswith(prefix):
            rest = base[len(prefix):]
            if rest.startswith("unimodal_"):
                modality = rest[len("unimodal_"):]
                if modality not in ("text", "audio", "visual"):
                    raise ValueError(f"unknown unimodal modality {modality!r} in {run_name!r}")
                return ds, "unimodal", modality
            if rest in ARCH_ABLATIONS:
                return ds, rest, None
            if rest in LOSS_ABLATION_VARIANTS:
                # Loss ablations use the standard xmofe architecture; only the
                # training loss differed — for inference, variant is xmofe.
                return ds, "xmofe", None
            if rest == "xmofe" or rest in FUSION_BASELINES:
                return ds, rest, None
            raise ValueError(f"can't parse variant from {run_name!r}: rest={rest!r}")
    raise ValueError(f"can't parse dataset from {run_name!r}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_checkpoints(checkpoints_root: Path) -> list[str]:
    """Return run names whose ``best.pt`` exists, sorted by dataset then name."""
    runs = []
    if not checkpoints_root.exists():
        return runs
    for d in sorted(checkpoints_root.iterdir()):
        if d.is_dir() and (d / "best.pt").exists():
            runs.append(d.name)
    return runs


def both_outputs_exist(results_dir: Path, run_name: str, *, want_robustness: bool, want_explanations: bool) -> bool:
    rob_ok = (not want_robustness) or (results_dir / f"robustness_{run_name}.json").exists()
    exp_ok = (not want_explanations) or (results_dir / f"explanations_{run_name}.json").exists()
    return rob_ok and exp_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--config-prefix", default="",
                        help="Subdir under configs/experiments/ to read configs from "
                             "(e.g. 'colab' → configs/experiments/colab/<ds>.yaml). "
                             "Empty = base configs (configs/experiments/<ds>.yaml).")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test",
                        help="Which split to evaluate on. Test by default.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for the eval loader (used for both evals).")
    parser.add_argument("--severity", default="medium",
                        help="Noise severity for robustness eval.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap each explanation sub-eval to N batches (smoke runs).")
    parser.add_argument("--k-fraction", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS),
                        help="Restrict to specific datasets.")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Restrict to specific variants (matches against parsed run-name variant).")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-explanations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.skip_robustness and args.skip_explanations:
        parser.error("--skip-robustness and --skip-explanations together leaves nothing to do")

    set_seed(args.seed)
    device = resolve_device(args.device)

    runs = discover_checkpoints(args.checkpoints_dir)
    log.info("discovered %d checkpoints under %s", len(runs), args.checkpoints_dir)

    # Group by dataset; each entry: (run_name, variant, modality)
    groups: dict[str, list[tuple[str, str, str | None]]] = {ds: [] for ds in args.datasets}
    skipped_unparseable: list[str] = []
    for run_name in runs:
        try:
            ds, variant, modality = parse_run_name(run_name)
        except ValueError as exc:
            skipped_unparseable.append(f"{run_name}: {exc}")
            continue
        if ds not in args.datasets:
            continue
        if args.variants and variant not in args.variants:
            continue
        groups[ds].append((run_name, variant, modality))

    if skipped_unparseable:
        for entry in skipped_unparseable:
            log.warning("skipping (unparseable run name): %s", entry)

    total = sum(len(v) for v in groups.values())
    log.info("planning %d evaluations across datasets=%s", total, list(groups.keys()))

    if args.dry_run:
        for ds in args.datasets:
            print(f"\n=== {ds} ({len(groups[ds])} checkpoints) ===")
            for run_name, variant, modality in groups[ds]:
                tag = f"{variant}" + (f"+{modality}" if modality else "")
                done_rob = (args.results_dir / f"robustness_{run_name}.json").exists()
                done_exp = (args.results_dir / f"explanations_{run_name}.json").exists()
                marks = []
                if not args.skip_robustness:
                    marks.append("R" if done_rob else "r")
                if not args.skip_explanations:
                    marks.append("E" if done_exp else "e")
                status = "".join(marks)  # uppercase = already done
                print(f"  [{status}]  {run_name}  ({tag})")
        return 0

    overall_start = time.time()
    completed_runs = 0

    for ds in args.datasets:
        ds_runs = groups[ds]
        if not ds_runs:
            continue
        log.info("=" * 70)
        log.info("dataset=%s  %d checkpoints", ds, len(ds_runs))

        # ---- Build dataset config + loader ONCE per dataset --------------
        ds_cfg_path = REPO_ROOT / "configs" / "experiments"
        if args.config_prefix:
            ds_cfg_path = ds_cfg_path / args.config_prefix
        ds_cfg_path = ds_cfg_path / f"{ds}.yaml"
        if not ds_cfg_path.exists():
            log.warning("skipping dataset %s: config not found at %s", ds, ds_cfg_path)
            continue

        ds_config = load_yaml(ds_cfg_path)
        model_config = load_yaml(REPO_ROOT / ds_config["model_config"])

        manifest_path = REPO_ROOT / ds_config["manifest_dir"] / f"{args.split}.pt"
        if not manifest_path.exists():
            log.warning("skipping dataset %s: manifest not found at %s", ds, manifest_path)
            continue

        log.info("loading %s loader...", manifest_path)
        loader = make_dataloader(manifest_path, batch_size=args.batch_size, shuffle=False)
        feature_dims = loader.dataset.feature_dims
        log.info(
            "loader ready: split=%s n=%d dims=text:%d audio:%d visual:%d",
            args.split, len(loader.dataset),
            feature_dims["text"], feature_dims["audio"], feature_dims["visual"],
        )

        # ---- Iterate over checkpoints in this dataset --------------------
        for i, (run_name, variant, modality) in enumerate(ds_runs, 1):
            ckpt_path = args.checkpoints_dir / run_name / "best.pt"
            log.info("[%s %d/%d] %s  (variant=%s%s)",
                     ds, i, len(ds_runs), run_name, variant,
                     f"+{modality}" if modality else "")

            if both_outputs_exist(args.results_dir, run_name,
                                  want_robustness=not args.skip_robustness,
                                  want_explanations=not args.skip_explanations):
                log.info("    SKIP (both outputs already exist)")
                completed_runs += 1
                continue

            try:
                model = build_model(
                    variant=variant, config=model_config,
                    text_dim=feature_dims["text"], audio_dim=feature_dims["audio"], visual_dim=feature_dims["visual"],
                    task=ds_config["task"], num_classes=ds_config.get("num_classes", 1),
                    modality=modality,
                ).to(device)
                payload = load_checkpoint(ckpt_path, model, map_location=str(device))
                log.info("    checkpoint epoch=%s best=%s",
                         payload.get("epoch"), payload.get("best_metric"))
            except Exception as exc:
                log.exception("    failed to build/load model for %s: %s", run_name, exc)
                continue

            try:
                if not args.skip_robustness:
                    rob_path = args.results_dir / f"robustness_{run_name}.json"
                    if rob_path.exists():
                        log.info("    robustness already exists; skipping")
                    else:
                        run_robustness_eval(
                            model=model, loader=loader,
                            task=ds_config["task"], num_classes=ds_config.get("num_classes", 1),
                            device=device,
                            experiment=ds, variant=variant, modality=modality,
                            split=args.split, severity=args.severity, seed=args.seed,
                            checkpoint_path=ckpt_path, output_path=rob_path,
                        )

                if not args.skip_explanations:
                    exp_path = args.results_dir / f"explanations_{run_name}.json"
                    if exp_path.exists():
                        log.info("    explanations already exist; skipping")
                    else:
                        run_explanations_eval(
                            model=model, loader=loader,
                            task=ds_config["task"], device=device,
                            experiment=ds, variant=variant, modality=modality,
                            split=args.split, seed=args.seed,
                            checkpoint_path=ckpt_path, output_path=exp_path,
                            k_fraction=args.k_fraction, max_batches=args.max_batches,
                        )
            except Exception as exc:
                log.exception("    eval failed for %s: %s", run_name, exc)
                continue
            finally:
                # Free model from GPU before loading next.
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            completed_runs += 1

    elapsed = time.time() - overall_start
    log.info("=" * 70)
    log.info("matrix complete in %.1f min  (%d/%d runs evaluated)",
             elapsed / 60, completed_runs, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
