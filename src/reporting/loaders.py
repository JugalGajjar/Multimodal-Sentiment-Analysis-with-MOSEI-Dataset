"""Discover and load result artifacts from training and evaluation runs.

Loaders here read the JSON / JSONL files written by Phases 5–10 so the
reporting scripts have a uniform view of "what we measured". Each loader
returns a list of dicts (rows), keyed by the most-relevant metadata
(run_name, dataset, variant, condition, …).

* ``load_training_runs`` — parses ``logs/<run>/metrics.jsonl`` to find the
  best-epoch validation metrics.
* ``load_robustness_runs`` — reads ``results/robustness_*.json``.
* ``load_explanation_runs`` — reads ``results/explanations_*.json``.
* ``load_vlm_runs`` — reads ``results/vlm_*_*_*.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Heuristic: which val metric to optimize for, given a run's task. Mirrors
# the Trainer's early-stopping conventions.
LOWER_IS_BETTER = ("mae", "mse", "task_loss")


def _is_lower_better(metric_name: str) -> bool:
    return any(token in metric_name.lower() for token in LOWER_IS_BETTER)


def _find_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_training_runs(logs_dir: Path) -> list[dict[str, Any]]:
    """Walk ``logs/`` for ``metrics.jsonl`` files and extract best-epoch summaries.

    Returns a list of dicts, one per run, with keys:
        ``run_name``, ``best_epoch``, plus all val/* metrics at that epoch.
    Runs that don't have any val records are skipped.
    """
    if not logs_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        records = _find_jsonl_records(run_dir / "metrics.jsonl")
        # Only consider records that carry at least one val/* metric.
        val_records = [r for r in records if any(k.startswith("val/") for k in r)]
        if not val_records:
            continue

        # Choose the early-stopping metric: prefer val/mae if present (smaller
        # is better), else val/weighted_f1 (larger is better), else fall back
        # to the most recent val record.
        best_record = None
        if any("val/mae" in r for r in val_records):
            best_record = min(
                (r for r in val_records if "val/mae" in r),
                key=lambda r: r["val/mae"],
            )
        elif any("val/weighted_f1" in r for r in val_records):
            best_record = max(
                (r for r in val_records if "val/weighted_f1" in r),
                key=lambda r: r["val/weighted_f1"],
            )
        else:
            best_record = val_records[-1]

        summary = {"run_name": run_dir.name}
        # Pull every val/* metric and the surrounding metadata.
        for k, v in best_record.items():
            if k.startswith(("val/", "epoch", "step", "timestamp")):
                summary[k] = v
        runs.append(summary)
    return runs


def _load_json_glob(results_dir: Path, prefix: str) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob(f"{prefix}*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not read %s: %s", path, e)
            continue
        payload["_source_file"] = path.name
        out.append(payload)
    return out


def load_robustness_runs(results_dir: Path) -> list[dict[str, Any]]:
    return _load_json_glob(results_dir, "robustness_")


def load_explanation_runs(results_dir: Path) -> list[dict[str, Any]]:
    return _load_json_glob(results_dir, "explanations_")


def load_vlm_runs(results_dir: Path) -> list[dict[str, Any]]:
    return _load_json_glob(results_dir, "vlm_")
