"""Build Tables 4 (missing modality) and 5 (noisy modality) per spec §20.4 / §20.5.

One row per condition, with the primary metric and reliability adaptation
where available. Pulls directly from ``results/robustness_*.json``.

Examples
--------
    python scripts/reporting/make_robustness_tables.py
    python scripts/reporting/make_robustness_tables.py --variant xmofe --dataset mosei --format md
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
log = logging.getLogger("make_robustness_tables")


def _build_missing_table(record: dict) -> tuple[list[dict], list[str], list[str]]:
    """Return (rows, columns, headers) for one robustness JSON's missing block."""
    primary = record.get("primary_metric", "mae")
    rows: list[dict] = []
    for cond_name, cond_metrics in (record.get("missing_modality") or {}).items():
        rows.append({
            "condition": cond_name,
            "metric": cond_metrics.get(primary),
            "reliability_text": cond_metrics.get("reliability_mean_text"),
            "reliability_audio": cond_metrics.get("reliability_mean_audio"),
            "reliability_visual": cond_metrics.get("reliability_mean_visual"),
        })
    columns = ["condition", "metric", "reliability_text", "reliability_audio", "reliability_visual"]
    headers = ["Condition", primary.upper(), "r_T", "r_A", "r_V"]
    return rows, columns, headers


def _build_noisy_table(record: dict) -> tuple[list[dict], list[str], list[str]]:
    primary = record.get("primary_metric", "mae")
    rows: list[dict] = []
    for cond_name, cond_metrics in (record.get("noisy_modality") or {}).items():
        rows.append({
            "condition": cond_name,
            "metric": cond_metrics.get(primary),
            "reliability_text": cond_metrics.get("reliability_mean_text"),
            "reliability_audio": cond_metrics.get("reliability_mean_audio"),
            "reliability_visual": cond_metrics.get("reliability_mean_visual"),
        })
    columns = ["condition", "metric", "reliability_text", "reliability_audio", "reliability_visual"]
    headers = ["Condition", primary.upper(), "r_T", "r_A", "r_V"]
    return rows, columns, headers


def _select_record(records: list[dict], dataset: str, variant: str | None) -> dict | None:
    matches = [r for r in records if r.get("experiment") == dataset]
    if variant is not None:
        matches = [r for r in matches if r.get("variant") == variant]
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collected", type=Path, default=REPO_ROOT / "results" / "collected.json")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--variant", default=None,
                        help="Filter to a specific model variant (default: any).")
    parser.add_argument("--format", choices=VALID_FORMATS, default="tex")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "paper" / "emnlp2026" / "tables")
    args = parser.parse_args()

    if not args.collected.exists():
        log.error("collected.json not found at %s. Run scripts/reporting/collect_results.py first.",
                  args.collected)
        sys.exit(2)
    bundle = json.loads(args.collected.read_text(encoding="utf-8"))
    records = bundle.get("robustness", [])
    if not records:
        log.warning("no robustness JSON files found")
        return

    datasets = sorted({r.get("experiment") for r in records if r.get("experiment")}) if args.dataset == "all" \
        else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        rec = _select_record(records, dataset, args.variant)
        if rec is None:
            log.warning("no robustness record for dataset=%s variant=%s", dataset, args.variant)
            continue

        # Missing-modality table
        rows, columns, headers = _build_missing_table(rec)
        out_path = args.output_dir / f"robustness_missing_{dataset}.{args.format}"
        write_table(
            out_path, rows, columns, headers,
            caption=f"Missing-modality robustness on {dataset.upper()}.",
            label=f"tab:robust_missing_{dataset}",
            fmt=args.format,
        )
        log.info("wrote %s", out_path)

        # Noisy-modality table
        rows, columns, headers = _build_noisy_table(rec)
        out_path = args.output_dir / f"robustness_noisy_{dataset}.{args.format}"
        write_table(
            out_path, rows, columns, headers,
            caption=f"Noisy-modality robustness on {dataset.upper()} (severity={rec.get('severity')}).",
            label=f"tab:robust_noisy_{dataset}",
            fmt=args.format,
        )
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
