"""Validate a prepared dataset's standardized metadata and splits.

Reads ``data/interim/<dataset>/{metadata.jsonl,splits.json}`` produced by
``prepare_<dataset>.py`` and reports:

  * sample counts per split
  * label distribution (binned for regression, raw counts for classification)
  * transcript length stats
  * duration stats (when start/end times are populated)
  * missing-file checks for audio_path / video_path
  * sample_id consistency between metadata and splits
  * duplicate sample_ids

Examples
--------
    python scripts/data/validate_dataset.py --dataset mosei
    python scripts/data/validate_dataset.py --dataset meld --check-files
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.data import VALID_DATASETS, VALID_SPLITS, Sample, read_jsonl, read_splits  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "datasets"


def load_config(dataset: str) -> dict:
    with (CONFIG_DIR / f"{dataset}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_stats(values: list[float], unit: str = "") -> str:
    if not values:
        return "n=0"
    suffix = f" {unit}" if unit else ""
    return (
        f"n={len(values)} "
        f"min={min(values):.2f}{suffix} "
        f"max={max(values):.2f}{suffix} "
        f"mean={statistics.fmean(values):.2f}{suffix} "
        f"median={statistics.median(values):.2f}{suffix}"
    )


def bin_regression_labels(values: list[float], edges: list[float]) -> Counter:
    counts: Counter = Counter()
    for v in values:
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            if lo <= v < hi:
                counts[f"[{lo:+.1f},{hi:+.1f})"] += 1
                break
        else:
            counts[f">={edges[-1]:+.1f}"] += 1
    return counts


def validate(dataset: str, check_files: bool) -> int:
    """Print a validation report. Returns the number of errors detected."""
    cfg = load_config(dataset)
    metadata_file = REPO_ROOT / cfg["paths"]["metadata_file"]
    splits_file = REPO_ROOT / cfg["paths"]["splits_file"]

    print(f"=== {cfg['display_name']} ({dataset}) ===")
    print(f"metadata: {metadata_file}")
    print(f"splits:   {splits_file}")

    if not metadata_file.exists():
        print(f"ERROR: metadata file not found. Run prepare_{dataset}.py first.")
        return 1
    if not splits_file.exists():
        print(f"ERROR: splits file not found. Run prepare_{dataset}.py first.")
        return 1

    errors = 0
    samples: list[Sample] = list(read_jsonl(metadata_file))
    splits = read_splits(splits_file)

    # Per-split bookkeeping
    by_split: dict[str, list[Sample]] = defaultdict(list)
    seen_ids: Counter = Counter()
    for s in samples:
        by_split[s.split].append(s)
        seen_ids[s.sample_id] += 1

    duplicates = [sid for sid, n in seen_ids.items() if n > 1]
    if duplicates:
        errors += len(duplicates)
        print(f"ERROR: {len(duplicates)} duplicate sample_id(s); first 5: {duplicates[:5]}")

    # Cross-check splits.json vs metadata.jsonl
    metadata_ids_by_split = {split: {s.sample_id for s in by_split[split]} for split in VALID_SPLITS}
    for split in VALID_SPLITS:
        split_ids = set(splits.get(split, []))
        meta_ids = metadata_ids_by_split[split]
        only_in_splits = split_ids - meta_ids
        only_in_meta = meta_ids - split_ids
        if only_in_splits:
            errors += len(only_in_splits)
            print(
                f"ERROR: {split}: {len(only_in_splits)} ids in splits.json "
                f"but missing from metadata.jsonl"
            )
        if only_in_meta:
            errors += len(only_in_meta)
            print(
                f"ERROR: {split}: {len(only_in_meta)} ids in metadata.jsonl "
                f"but missing from splits.json"
            )

    # Per-split summaries
    print("")
    print(f"{'split':<6} {'samples':>8} {'transcript_chars':>20} {'duration_s':>20}")
    for split in VALID_SPLITS:
        items = by_split[split]
        n = len(items)
        if n == 0:
            print(f"{split:<6} {0:>8}")
            continue
        char_lens = [len(s.transcript) for s in items]
        durations = [s.duration for s in items if s.duration is not None]
        char_summary = f"{statistics.fmean(char_lens):>6.1f} avg"
        dur_summary = f"{statistics.fmean(durations):>6.2f}s avg" if durations else "—"
        print(f"{split:<6} {n:>8} {char_summary:>20} {dur_summary:>20}")

    # Task / label distribution
    cfg_labels = cfg.get("labels", {})
    task = cfg_labels.get("task", "regression")
    print(f"\ntask: {task}")

    primary_values = [s.primary_label for s in samples if s.primary_label is not None]
    if not primary_values:
        errors += 1
        print("ERROR: no primary_label values present")
    elif task == "regression":
        floats = [float(v) for v in primary_values]
        rng = cfg_labels.get("range", [min(floats), max(floats)])
        edges = [rng[0] + (rng[1] - rng[0]) * i / 6 for i in range(7)]
        edges[-1] = rng[1] + 1e-6  # include the upper bound
        bins = bin_regression_labels(floats, edges)
        print(f"primary_label stats: {fmt_stats(floats)}")
        for k in sorted(bins):
            print(f"  {k:>16}: {bins[k]}")
    else:  # classification
        counts = Counter(int(v) for v in primary_values)
        emotions = cfg_labels.get("emotions") or cfg_labels.get("classes")
        for cls_id in sorted(counts):
            label = emotions[cls_id] if emotions and cls_id < len(emotions) else f"class_{cls_id}"
            print(f"  {cls_id:>3} {label:<16}: {counts[cls_id]}")

    # File checks
    if check_files:
        print("\nfile checks:")
        for kind in ("audio_path", "video_path"):
            present = [getattr(s, kind) for s in samples if getattr(s, kind)]
            if not present:
                print(f"  {kind}: 0 populated — skipping existence check")
                continue
            missing = [p for p in present if not (REPO_ROOT / p).exists()]
            print(f"  {kind}: {len(present)} populated, {len(missing)} missing on disk")
            if missing:
                errors += len(missing)
                for p in missing[:3]:
                    print(f"    missing: {p}")
                if len(missing) > 3:
                    print(f"    ... and {len(missing) - 3} more")

    print(f"\nresult: {errors} error(s)")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=(*VALID_DATASETS, "all"),
        default="all",
        help="Dataset to validate (default: all).",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Verify that audio_path / video_path actually exist on disk.",
    )
    args = parser.parse_args()

    targets = VALID_DATASETS if args.dataset == "all" else (args.dataset,)
    total_errors = 0
    for name in targets:
        try:
            total_errors += validate(name, args.check_files)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            total_errors += 1
        print("")

    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
