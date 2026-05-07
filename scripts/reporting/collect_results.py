"""Walk training logs and evaluation outputs into a single consolidated bundle.

Produces:

    results/collected.json                     — nested superset of everything
    results/collected/training.csv             — one row per training run
    results/collected/robustness.csv           — one row per robustness condition
    results/collected/explanations.csv         — one row per explanation eval
    results/collected/vlm.csv                  — one row per VLM run

Downstream ``make_*_tables.py`` scripts read either ``collected.json`` or the
domain-specific CSVs. Idempotent — re-runs simply overwrite.

Examples
--------
    python scripts/reporting/collect_results.py
    python scripts/reporting/collect_results.py --logs-dir logs --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reporting import (  # noqa: E402
    load_explanation_runs,
    load_robustness_runs,
    load_training_runs,
    load_vlm_runs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_results")


def _flatten_for_csv(record: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a one-level dict-of-dicts payload for CSV emission."""
    flat: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_for_csv(v, prefix=key, sep=sep))
        elif isinstance(v, list):
            flat[key] = json.dumps(v)
        else:
            flat[key] = v
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        # Even with no rows, leave a header-only stub so reviewers know the
        # collector ran. Writing an empty file would be more confusing.
        path.write_text("(no records)\n", encoding="utf-8")
        return

    flat_rows = [_flatten_for_csv(r) for r in rows]
    columns = sorted({k for r in flat_rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in flat_rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def _summarize_robustness(record: dict[str, Any]) -> dict[str, Any]:
    """Pull the summary block plus identifying metadata into a flat row."""
    out: dict[str, Any] = {
        "experiment": record.get("experiment"),
        "variant": record.get("variant"),
        "modality": record.get("modality"),
        "split": record.get("split"),
        "severity": record.get("severity"),
        "primary_metric": record.get("primary_metric"),
        "task": record.get("task"),
        "checkpoint": record.get("checkpoint"),
        "source_file": record.get("_source_file"),
    }
    summary = record.get("summary") or {}
    out.update({f"summary.{k}": v for k, v in summary.items()})
    # Also surface the clean metric so the table generator can compute drops.
    clean = (record.get("missing_modality") or {}).get("clean") or {}
    primary = record.get("primary_metric")
    if primary and primary in clean:
        out[f"clean.{primary}"] = clean[primary]
    return out


def _summarize_explanations(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "experiment": record.get("experiment"),
        "variant": record.get("variant"),
        "modality": record.get("modality"),
        "split": record.get("split"),
        "task": record.get("task"),
        "checkpoint": record.get("checkpoint"),
        "source_file": record.get("_source_file"),
    }
    mf = record.get("modality_faithfulness") or {}
    out["modality_faithfulness.spearman"] = mf.get("spearman")
    out["modality_faithfulness.kl_s_to_r_mean"] = mf.get("kl_s_to_r_mean")
    out["modality_faithfulness.top1_agreement"] = mf.get("top1_agreement")

    for modality in ("text", "audio", "visual"):
        d = (record.get("deletion") or {}).get(modality, {}) or {}
        out[f"deletion.{modality}.aulc"] = d.get("aulc")
        i = (record.get("insertion") or {}).get(modality, {}) or {}
        out[f"insertion.{modality}.aulc"] = i.get("aulc")
        s = (record.get("sufficiency") or {}).get(modality, {}) or {}
        out[f"sufficiency.{modality}"] = s.get("sufficiency")
        c = (record.get("comprehensiveness") or {}).get(modality, {}) or {}
        out[f"comprehensiveness.{modality}"] = c.get("comprehensiveness")
    align = record.get("chsims_reliability_alignment") or {}
    if align:
        out["chsims.spearman"] = align.get("spearman")
        out["chsims.kl_rstar_to_r_mean"] = align.get("kl_rstar_to_r_mean")
        out["chsims.top1_agreement"] = align.get("top1_agreement")
    return out


def _summarize_vlm(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    return {
        "experiment": record.get("experiment"),
        "vlm": record.get("vlm"),
        "split": record.get("split"),
        "task": record.get("task"),
        "n_evaluated": record.get("n_evaluated"),
        "max_samples": record.get("max_samples"),
        "dry_run": record.get("dry_run"),
        "accuracy": metrics.get("accuracy"),
        "weighted_f1": metrics.get("weighted_f1"),
        "macro_f1": metrics.get("macro_f1"),
        "parse_success_rate": metrics.get("parse_success_rate"),
        "source_file": record.get("_source_file"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    log.info("logs_dir=%s results_dir=%s", args.logs_dir, args.results_dir)

    training = load_training_runs(args.logs_dir)
    robustness = load_robustness_runs(args.results_dir)
    explanations = load_explanation_runs(args.results_dir)
    vlm = load_vlm_runs(args.results_dir)

    log.info(
        "found training=%d robustness=%d explanations=%d vlm=%d",
        len(training), len(robustness), len(explanations), len(vlm),
    )

    bundle = {
        "training": training,
        "robustness": robustness,
        "explanations": explanations,
        "vlm": vlm,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "logs_dir": str(args.logs_dir),
        "results_dir": str(args.results_dir),
    }

    # Nested JSON
    json_path = args.output_dir / "collected.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
        f.write("\n")
    log.info("wrote %s", json_path)

    # Per-domain flat CSVs
    csv_dir = args.output_dir / "collected"
    csv_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(csv_dir / "training.csv", training)
    _write_csv(csv_dir / "robustness.csv", [_summarize_robustness(r) for r in robustness])
    _write_csv(csv_dir / "explanations.csv", [_summarize_explanations(r) for r in explanations])
    _write_csv(csv_dir / "vlm.csv", [_summarize_vlm(r) for r in vlm])
    log.info("wrote per-domain CSVs under %s", csv_dir)


if __name__ == "__main__":
    main()
