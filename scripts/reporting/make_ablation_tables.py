"""Build Table 3 (ablation study) per spec §19.1 / §20.3.

Rows: full X-MoFE + each ablation variant.
Columns: performance, faithfulness, robustness summary metrics.

Reads ``collected.json`` for training results, robustness summary, and
explanation faithfulness; combines into a single per-dataset ablation table.

Examples
--------
    python scripts/reporting/make_ablation_tables.py
    python scripts/reporting/make_ablation_tables.py --format md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reporting import VALID_FORMATS, write_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_ablation_tables")

ABLATION_VARIANTS = (
    "xmofe",
    "xmofe_no_reliability",
    "xmofe_no_interaction",
    "xmofe_no_trimodal",
    "loss_no_faithfulness",
    "loss_no_stability",
    "loss_no_entropy",
    "loss_no_reliability",
)

DATASET_PERF_KEYS = {
    "mosei": "val/mae",
    "ch_sims": "val/mae",
    "meld": "val/weighted_f1",
}


def _filter_runs(runs, dataset: str, variant: str) -> list[dict]:
    """Keep training-summary rows whose run_name contains both tags."""
    tag_dataset = dataset
    return [r for r in runs if tag_dataset in r.get("run_name", "") and variant in r.get("run_name", "")]


def _row_for(runs, robustness, explanations, dataset: str, variant: str) -> dict | None:
    matched = _filter_runs(runs, dataset, variant)
    if not matched:
        return None
    perf_key = DATASET_PERF_KEYS[dataset]
    if perf_key in matched[0]:
        # Aggregate across seeds: mean of best-epoch metrics
        perf_values = [r[perf_key] for r in matched if perf_key in r]
        perf = sum(perf_values) / len(perf_values) if perf_values else None
    else:
        perf = None

    # Pull faithfulness from explanation runs that mention both tags.
    expl = next((
        r for r in explanations
        if r.get("experiment") == dataset and (r.get("variant") == variant or variant in r.get("checkpoint", ""))
    ), None)
    faithfulness = (expl or {}).get("modality_faithfulness", {}).get("spearman") if expl else None

    # Pull robustness drop from robustness runs.
    rob = next((
        r for r in robustness
        if r.get("experiment") == dataset and (r.get("variant") == variant or variant in r.get("checkpoint", ""))
    ), None)
    rob_drop = (rob or {}).get("summary", {}).get("missing_avg_drop") if rob else None

    return {
        "variant": variant,
        "performance": perf,
        "faithfulness_spearman": faithfulness,
        "missing_avg_drop": rob_drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", type=Path, default=REPO_ROOT / "results" / "collected.json")
    parser.add_argument("--dataset", choices=tuple(DATASET_PERF_KEYS) + ("all",), default="all")
    parser.add_argument("--format", choices=VALID_FORMATS, default="tex")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper" / "emnlp2026" / "tables")
    args = parser.parse_args()

    if not args.collected.exists():
        log.error("collected.json not found at %s. Run scripts/reporting/collect_results.py first.",
                  args.collected)
        sys.exit(2)
    bundle = json.loads(args.collected.read_text(encoding="utf-8"))
    runs = bundle.get("training", [])
    rob = bundle.get("robustness", [])
    expl = bundle.get("explanations", [])

    target_datasets = list(DATASET_PERF_KEYS) if args.dataset == "all" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in target_datasets:
        rows = []
        for variant in ABLATION_VARIANTS:
            row = _row_for(runs, rob, expl, dataset, variant)
            if row is not None:
                rows.append(row)

        if not rows:
            log.warning("no ablation rows for dataset=%s; skipping", dataset)
            continue

        ext = args.format
        out_path = args.output_dir / f"ablation_{dataset}.{ext}"
        write_table(
            out_path,
            rows,
            columns=["variant", "performance", "faithfulness_spearman", "missing_avg_drop"],
            headers=["Variant", "Performance", "Faithfulness ρ", "Robustness Δ"],
            caption=f"Ablation study on {dataset.upper()} (spec §19.1).",
            label=f"tab:ablation_{dataset}",
            fmt=args.format,
        )
        log.info("wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
