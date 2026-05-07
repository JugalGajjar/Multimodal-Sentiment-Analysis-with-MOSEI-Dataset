"""Run the X-MoFE training matrix in a single command.

Iterates through the (dataset, variant, modality, loss_config, seed) tuples
defined by the spec §21.2 experiment matrix and shells out to
``scripts/train/train_xmofe.py`` for each entry. Designed for long Colab
sessions where disconnects are likely:

* Skips runs whose ``checkpoints/<run_name>/best.pt`` already exists, so
  rerunning the script after a disconnect resumes from where it left off.
* Continues on failure — a single crashing run does not abort the matrix.
* Per-run stdout/stderr lands in ``logs/<run_name>/run_matrix.log`` (in
  addition to the trainer's own ``training.log`` and W&B).

The matrix:

    Tier 1: X-MoFE x 3 seeds x 3 datasets        =  9 runs
    Tier 2: 6 fusion baselines x 1 seed x 3 ds   = 18 runs
    Tier 3: 3 architectural ablations x 1 x 3 ds =  9 runs
    Tier 4: 4 loss ablations (chsims-only repl.) = 10 runs
                                              total ≈ 46 runs

Examples
--------
    # All tiers
    python scripts/train/run_matrix.py --experiment-prefix colab

    # Tier 1 only (X-MoFE main results)
    python scripts/train/run_matrix.py --experiment-prefix colab --tiers 1

    # Dry run — print what would happen
    python scripts/train/run_matrix.py --experiment-prefix colab --dry-run

    # Disable W&B for the whole matrix
    python scripts/train/run_matrix.py --experiment-prefix colab --no-wandb
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("ch_sims", "meld", "mosei")


def make_matrix() -> list[dict]:
    """Build the full (~46-run) experiment matrix.

    Each entry is the kwargs needed to call train_xmofe.py.
    """
    runs: list[dict] = []

    # Tier 1: X-MoFE main results (3 seeds x 3 datasets) ----------------
    for ds in DATASETS:
        for seed in (0, 1, 2):
            runs.append({
                "tier": 1,
                "dataset": ds,
                "variant": "xmofe",
                "modality": None,
                "loss_config": None,
                "seed": seed,
                "run_name": f"{ds}_xmofe_s{seed}",
            })

    # Tier 2: Controlled fusion baselines (1 seed x 3 datasets x 6 var) -
    for ds in DATASETS:
        for variant in ("early_fusion", "late_fusion", "hybrid_fusion"):
            runs.append({
                "tier": 2,
                "dataset": ds,
                "variant": variant,
                "modality": None,
                "loss_config": "configs/training/loss_task_only.yaml",
                "seed": 0,
                "run_name": f"{ds}_{variant}_s0",
            })
        for modality in ("text", "audio", "visual"):
            runs.append({
                "tier": 2,
                "dataset": ds,
                "variant": "unimodal",
                "modality": modality,
                "loss_config": "configs/training/loss_task_only.yaml",
                "seed": 0,
                "run_name": f"{ds}_unimodal_{modality}_s0",
            })

    # Tier 3: Architectural ablations (1 seed x 3 datasets x 3 abl) -----
    for ds in DATASETS:
        for variant in ("xmofe_no_reliability", "xmofe_no_interaction", "xmofe_no_trimodal"):
            runs.append({
                "tier": 3,
                "dataset": ds,
                "variant": variant,
                "modality": None,
                "loss_config": None,
                "seed": 0,
                "run_name": f"{ds}_{variant}_s0",
            })

    # Tier 4: Loss ablations --------------------------------------------
    # faithfulness/stability/entropy: full set (3 datasets each).
    for loss_name in ("loss_no_faithfulness", "loss_no_stability", "loss_no_entropy"):
        for ds in DATASETS:
            tag = loss_name.replace("loss_", "")  # "no_faithfulness"
            runs.append({
                "tier": 4,
                "dataset": ds,
                "variant": "xmofe",
                "modality": None,
                "loss_config": f"configs/training/{loss_name}.yaml",
                "seed": 0,
                "run_name": f"{ds}_xmofe_{tag}_s0",
            })
    # Reliability supervision only meaningful on CH-SIMS (only dataset
    # with unimodal labels). Per spec §21.2.
    runs.append({
        "tier": 4,
        "dataset": "ch_sims",
        "variant": "xmofe",
        "modality": None,
        "loss_config": "configs/training/loss_no_reliability.yaml",
        "seed": 0,
        "run_name": "ch_sims_xmofe_no_reliability_loss_s0",
    })

    return runs


def build_command(entry: dict, *, experiment_prefix: str, no_wandb: bool) -> list[str]:
    """Construct the train_xmofe.py CLI command for an entry."""
    experiment = (
        f"{experiment_prefix}/{entry['dataset']}" if experiment_prefix
        else entry["dataset"]
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "train_xmofe.py"),
        "--experiment", experiment,
        "--variant", entry["variant"],
        "--seed", str(entry["seed"]),
        "--run-name", entry["run_name"],
    ]
    if entry["modality"] is not None:
        cmd += ["--modality", entry["modality"]]
    if entry["loss_config"] is not None:
        cmd += ["--loss-config", entry["loss_config"]]
    if no_wandb:
        cmd += ["--no-wandb"]
    return cmd


def is_run_complete(run_name: str) -> bool:
    """A run is considered complete if it has a best.pt checkpoint."""
    return (REPO_ROOT / "checkpoints" / run_name / "best.pt").exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-prefix", default="colab",
                        help="Subdir under configs/experiments/ (e.g. 'colab' loads colab/<dataset>.yaml). "
                        "Pass empty string to use the base configs (M4 Pro).")
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Which tiers to run (default: all).")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS),
                        help="Restrict to specific datasets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned commands without executing.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B for every run.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if best.pt already exists.")
    args = parser.parse_args()

    matrix = make_matrix()
    matrix = [e for e in matrix if e["tier"] in args.tiers and e["dataset"] in args.datasets]

    print(f"Matrix: {len(matrix)} runs across tiers={args.tiers} datasets={args.datasets}")
    print(f"Experiment prefix: {args.experiment_prefix or '(base)'}")
    print(f"Dry run: {args.dry_run}")
    print()

    summary = {"completed": [], "skipped": [], "failed": [], "started_at": datetime.now().isoformat()}
    matrix_log_path = REPO_ROOT / "logs" / "run_matrix_summary.json"
    matrix_log_path.parent.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    for i, entry in enumerate(matrix, 1):
        run_name = entry["run_name"]
        cmd = build_command(entry, experiment_prefix=args.experiment_prefix, no_wandb=args.no_wandb)
        header = (
            f"[{i}/{len(matrix)}] tier={entry['tier']}  "
            f"{entry['dataset']:7s}  variant={entry['variant']}"
            f"{('+' + entry['modality']) if entry['modality'] else ''}  "
            f"seed={entry['seed']}"
            f"{('  loss=' + Path(entry['loss_config']).stem) if entry['loss_config'] else ''}"
        )
        print(header)
        print(f"    run_name: {run_name}")
        print(f"    cmd:      {' '.join(cmd)}")

        if not args.force and is_run_complete(run_name):
            print(f"    SKIP (best.pt already exists)")
            summary["skipped"].append(run_name)
            print()
            continue

        if args.dry_run:
            summary["completed"].append(run_name)
            print()
            continue

        # Stream output to terminal AND tee to a per-run matrix log.
        per_run_log_dir = REPO_ROOT / "logs" / run_name
        per_run_log_dir.mkdir(parents=True, exist_ok=True)
        per_run_log = per_run_log_dir / "run_matrix.log"

        run_start = time.time()
        try:
            with per_run_log.open("w") as logf:
                proc = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, check=False,
                )
            elapsed = time.time() - run_start
            if proc.returncode == 0 and is_run_complete(run_name):
                print(f"    OK  ({timedelta(seconds=int(elapsed))})")
                summary["completed"].append(run_name)
            else:
                print(f"    FAIL  rc={proc.returncode}  ({timedelta(seconds=int(elapsed))})")
                print(f"          tail of {per_run_log}:")
                tail = subprocess.run(
                    ["tail", "-n", "20", str(per_run_log)],
                    capture_output=True, text=True,
                )
                for line in tail.stdout.splitlines():
                    print(f"          | {line}")
                summary["failed"].append({"run_name": run_name, "rc": proc.returncode})
        except KeyboardInterrupt:
            print("    interrupted by user; saving summary and exiting")
            summary["failed"].append({"run_name": run_name, "rc": "KeyboardInterrupt"})
            summary["finished_at"] = datetime.now().isoformat()
            matrix_log_path.write_text(json.dumps(summary, indent=2))
            return 130

        # Persist summary after each run so a crash doesn't lose progress.
        summary["finished_at"] = datetime.now().isoformat()
        matrix_log_path.write_text(json.dumps(summary, indent=2))
        print()

    overall_elapsed = time.time() - overall_start
    print("=" * 70)
    print(f"Matrix complete in {timedelta(seconds=int(overall_elapsed))}")
    print(f"  completed: {len(summary['completed'])}")
    print(f"  skipped:   {len(summary['skipped'])}")
    print(f"  failed:    {len(summary['failed'])}")
    if summary["failed"]:
        print(f"  failures: {[e['run_name'] for e in summary['failed']]}")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
