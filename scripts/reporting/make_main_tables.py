"""Build Table 1 (main performance) and Table 2 (fusion strategy comparison).

Reads ``results/collected.json`` plus the external-baselines YAML and
emits per-dataset tables of metrics. External baselines are folded in with
their citation note so the LaTeX caption can clearly mark them as
"reported from prior work" per spec §15.3.

Examples
--------
    python scripts/reporting/make_main_tables.py
    python scripts/reporting/make_main_tables.py --format md --dataset mosei
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.reporting import VALID_FORMATS, write_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_main_tables")

DATASET_METRIC_COLUMNS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "mosei": {
        "regression_columns": [
            ("mae", "MAE"),
            ("pearson_r", "Pearson r"),
            ("binary_acc", "Acc-2"),
            ("binary_f1", "F1-2"),
        ],
    },
    "ch_sims": {
        "regression_columns": [
            ("mae", "MAE"),
            ("pearson_r", "Pearson r"),
            ("binary_acc", "Acc-2"),
            ("binary_f1", "F1-2"),
        ],
    },
    "meld": {
        "classification_columns": [
            ("accuracy", "Acc"),
            ("weighted_f1", "Weighted F1"),
            ("macro_f1", "Macro F1"),
        ],
    },
}


def _val_metric(run: dict, key: str) -> float | None:
    """Pull ``val/<key>`` from a training-run summary."""
    return run.get(f"val/{key}")


def _experiment_of(run: dict) -> str | None:
    """Best-effort: split run_name like ``mosei_xmofe_20260507_*`` into dataset prefix."""
    name = run.get("run_name", "")
    for ds in DATASET_METRIC_COLUMNS:
        if name.startswith(ds + "_") or name.startswith(ds):
            return ds
    return None


def _variant_of(run: dict) -> str:
    """Best-effort variant tag from run_name (the segment after the dataset)."""
    name = run.get("run_name", "")
    parts = name.split("_")
    if len(parts) >= 2:
        # Skip the first dataset token
        return "_".join(parts[1:-2]) if len(parts) >= 3 else parts[1]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", type=Path, default=REPO_ROOT / "results" / "collected.json")
    parser.add_argument(
        "--external-baselines",
        type=Path,
        default=REPO_ROOT / "configs" / "reporting" / "external_baselines.yaml",
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_METRIC_COLUMNS) + ("all",), default="all")
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

    external = {}
    if args.external_baselines.exists():
        external = yaml.safe_load(args.external_baselines.read_text(encoding="utf-8")) or {}
    else:
        log.warning("external baselines config not found at %s; skipping", args.external_baselines)

    target_datasets = list(DATASET_METRIC_COLUMNS) if args.dataset == "all" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in target_datasets:
        spec = DATASET_METRIC_COLUMNS[dataset]
        # Pick the column set; mosei/ch_sims are regression, meld is classification.
        if "regression_columns" in spec:
            metric_cols = spec["regression_columns"]
        else:
            metric_cols = spec["classification_columns"]

        rows: list[dict] = []

        # Our own runs for this dataset
        for run in runs:
            if _experiment_of(run) != dataset:
                continue
            row = {"variant": _variant_of(run), "source": "ours"}
            for key, _label in metric_cols:
                row[key] = _val_metric(run, key)
            rows.append(row)

        # External baselines from YAML
        for name, baseline in (external.get(dataset) or {}).items():
            row = {"variant": name, "source": baseline.get("note", "reported from prior work")}
            metrics = baseline.get("metrics") or {}
            for key, _label in metric_cols:
                row[key] = metrics.get(key)
            rows.append(row)

        if not rows:
            log.warning("no rows for dataset=%s; skipping", dataset)
            continue

        columns = ["variant"] + [k for k, _ in metric_cols] + ["source"]
        headers = ["Variant"] + [h for _, h in metric_cols] + ["Source"]

        ext = "tex" if args.format == "tex" else args.format
        out_path = args.output_dir / f"main_{dataset}.{ext}"
        write_table(
            out_path, rows, columns, headers,
            caption=f"Main performance on {dataset.upper()} (ours vs reported).",
            label=f"tab:main_{dataset}",
            fmt=args.format,
        )
        log.info("wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
