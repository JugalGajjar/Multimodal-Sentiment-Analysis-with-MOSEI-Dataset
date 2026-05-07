"""Run VLM evaluation on a dataset's test split (spec §10 + §18).

Per sample:
  1. Adaptive frame sampling from raw mp4 (4 frames if < 4 sec, else 8).
  2. Build sentiment- or emotion-classification prompt with the transcript.
  3. Generate response with the chosen VLM (Qwen2.5-VL or LLaVA-OneVision).
  4. Parse JSON / fallback-extract the predicted label.

Aggregates accuracy, weighted F1, macro F1, and parse-success rate, then
writes a JSON report to ``results/vlm_<vlm>_<run_name>.json``.

Datasets:
  * MELD     — 7-class emotion (raw mp4s present)
  * CH-SIMS  — 3-class sentiment via the official ``annotation`` column
  * MOSEI    — **not supported** (only CSDs, no raw video)

Examples
--------
    # Smoke run with a stub VLM (no model load)
    python scripts/vlms/run_vlm.py --vlm qwen --experiment meld \\
        --max-samples 10 --dry-run

    # Real Qwen2.5-VL on a 500-sample stratified subset
    python scripts/vlms/run_vlm.py --vlm qwen --experiment meld \\
        --max-samples 500

    # Real LLaVA-OneVision on CH-SIMS test
    python scripts/vlms/run_vlm.py --vlm llava --experiment ch_sims \\
        --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.data import read_jsonl  # noqa: E402
from src.vlms import (  # noqa: E402
    build_prompt,
    labels_for_task,
    parse_response,
    sample_frames_adaptive,
    stratified_subsample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_vlm")

# Dataset → (VLM task, label-extraction function from rich-label dict)
DATASET_TASK_MAP: dict[str, tuple[str, str]] = {
    "meld": ("emotion", "emotion"),               # rich label key holding the gold label string
    "ch_sims": ("sentiment", "annotation"),       # CH-SIMS has Negative/Neutral/Positive in annotation
}


class _StubVLM:
    """Dry-run stub: returns a canned valid JSON response.

    Lets us validate the entire pipeline (frame sampling → prompting →
    parsing → metrics → JSON write) without loading a 14 GB checkpoint.
    """

    def __init__(self, default_label: str) -> None:
        self.default_label = default_label
        self.calls = 0

    def generate(self, frames, prompt, max_new_tokens=128, temperature=0.0) -> str:
        self.calls += 1
        return (
            f'{{"label": "{self.default_label}", "confidence": 0.5, '
            f'"explanation": "stub response (dry-run mode)"}}'
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_vlm(name: str, device: str, dry_run: bool, default_label: str):
    if dry_run:
        log.info("dry-run mode — using StubVLM")
        return _StubVLM(default_label=default_label)
    if name == "qwen":
        from src.vlms.qwen_vl import Qwen25VL
        return Qwen25VL(device=device)
    if name == "llava":
        from src.vlms.llava_onevision import LLaVAOneVision
        return LLaVAOneVision(device=device)
    raise ValueError(f"unknown --vlm {name!r}; expected 'qwen' or 'llava'")


def extract_gold_label(sample, dataset: str) -> str | None:
    """Pull the gold label string out of a Sample's rich labels."""
    rich = sample.labels or {}
    _, label_key = DATASET_TASK_MAP[dataset]
    value = rich.get(label_key)
    if value is None:
        return None
    return str(value).strip().lower() or None


def run(
    vlm,
    samples: Sequence,
    dataset: str,
    task: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run inference over ``samples`` and return per-sample records + metrics."""
    valid_labels = labels_for_task(task)
    valid_labels_lower = tuple(label.lower() for label in valid_labels)

    records: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if not sample.video_path:
            log.warning("[%d/%d] %s: no video_path; skipping", i + 1, len(samples), sample.sample_id)
            continue

        gold = extract_gold_label(sample, dataset)
        if gold is None or gold not in valid_labels_lower:
            log.warning(
                "[%d/%d] %s: gold label %r not in valid labels for %s; skipping",
                i + 1, len(samples), sample.sample_id, gold, task,
            )
            continue

        frames = sample_frames_adaptive(REPO_ROOT / sample.video_path)
        prompt = build_prompt(sample.transcript, task=task)

        try:
            raw = vlm.generate(frames, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        except Exception as e:  # noqa: BLE001 — keep the harness running on per-sample failures
            log.warning("[%d/%d] %s: generation failed (%s)", i + 1, len(samples), sample.sample_id, e)
            raw = ""
        parsed = parse_response(raw, valid_labels)
        record = {
            "sample_id": sample.sample_id,
            "transcript": sample.transcript,
            "gold": gold,
            "predicted": parsed["label"],
            "confidence": parsed["confidence"],
            "explanation": parsed["explanation"],
            "parsed_ok": parsed["parsed_ok"],
            "raw": raw,
            "num_frames": len(frames),
        }
        records.append(record)
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            log.info("[%d/%d] processed", i + 1, len(samples))

    # Compute metrics over records that produced both gold and predicted labels.
    label_index = {label.lower(): idx for idx, label in enumerate(valid_labels)}
    gold_ids: list[int] = []
    pred_ids: list[int] = []
    for r in records:
        if r["predicted"] is None:
            # Unparseable → tally as a wrong prediction by mapping to a sentinel
            # so accuracy reflects the failure honestly.
            gold_ids.append(label_index[r["gold"]])
            pred_ids.append(-1)
            continue
        gold_ids.append(label_index[r["gold"]])
        pred_ids.append(label_index[r["predicted"].lower()])

    n = len(records)
    n_parsed = sum(1 for r in records if r["parsed_ok"])
    metrics: dict[str, float] = {
        "n_samples": float(n),
        "parse_success_rate": float(n_parsed) / n if n else 0.0,
    }
    if n > 0:
        # sklearn doesn't accept the -1 sentinel cleanly; treat unparsed as a
        # distinct "not-a-class" id by remapping it to len(valid_labels).
        valid_size = len(valid_labels)
        pred_ids_safe = [v if v >= 0 else valid_size for v in pred_ids]
        metrics["accuracy"] = float(accuracy_score(gold_ids, pred_ids_safe))
        metrics["weighted_f1"] = float(
            f1_score(gold_ids, pred_ids_safe, labels=list(range(valid_size)),
                     average="weighted", zero_division=0)
        )
        metrics["macro_f1"] = float(
            f1_score(gold_ids, pred_ids_safe, labels=list(range(valid_size)),
                     average="macro", zero_division=0)
        )

    return records, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm", choices=("qwen", "llava"), required=True)
    parser.add_argument(
        "--experiment", choices=tuple(DATASET_TASK_MAP), required=True,
        help=f"Dataset name; supported: {list(DATASET_TASK_MAP)} (MOSEI excluded — no raw video).",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Stratified subset size; pass 0 to use the full split.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a stub VLM that returns canned JSON (validates the pipeline without GPU).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / "experiments" / f"{args.experiment}.yaml"
    config = load_yaml(config_path)
    metadata_path = REPO_ROOT / config["paths"]["metadata_file"] if "paths" in config else \
        REPO_ROOT / "data" / "interim" / args.experiment / "metadata.jsonl"

    if not metadata_path.exists():
        log.error("metadata file not found: %s", metadata_path)
        sys.exit(2)

    log.info("loading metadata: %s", metadata_path)
    all_samples = [s for s in read_jsonl(metadata_path) if s.split == args.split]
    log.info("%d samples in split=%s", len(all_samples), args.split)

    task, label_key = DATASET_TASK_MAP[args.experiment]
    valid_labels = labels_for_task(task)

    # Filter to samples whose gold label is in the valid label set.
    keep: list = []
    for s in all_samples:
        gold = extract_gold_label(s, args.experiment)
        if gold is None or gold not in (label.lower() for label in valid_labels):
            continue
        keep.append(s)
    log.info("%d samples remain after gold-label filtering", len(keep))

    if args.max_samples and args.max_samples > 0 and args.max_samples < len(keep):
        keep = stratified_subsample(
            keep, label_fn=lambda s, dataset=args.experiment: extract_gold_label(s, dataset),
            n_total=args.max_samples, seed=args.seed,
        )
        log.info("subsampled to %d (stratified, seed=%d)", len(keep), args.seed)

    default_label = valid_labels[0]
    vlm = build_vlm(args.vlm, args.device, args.dry_run, default_label=default_label)

    records, metrics = run(
        vlm, keep, dataset=args.experiment, task=task, max_new_tokens=args.max_new_tokens,
    )

    output_payload = {
        "experiment": args.experiment,
        "split": args.split,
        "vlm": args.vlm,
        "dry_run": args.dry_run,
        "max_samples": args.max_samples,
        "task": task,
        "valid_labels": list(valid_labels),
        "n_total_in_split": len(all_samples),
        "n_evaluated": len(records),
        "seed": args.seed,
        "metrics": metrics,
        "predictions": records,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    output_path = args.output or (
        REPO_ROOT / "results" / f"vlm_{args.vlm}_{args.experiment}_{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        f.write("\n")
    log.info(
        "wrote %s — accuracy=%.3f weighted_f1=%.3f parse_ok=%.3f",
        output_path,
        metrics.get("accuracy", float("nan")),
        metrics.get("weighted_f1", float("nan")),
        metrics.get("parse_success_rate", float("nan")),
    )


if __name__ == "__main__":
    main()
