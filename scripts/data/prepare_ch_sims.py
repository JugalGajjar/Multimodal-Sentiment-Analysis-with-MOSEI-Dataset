"""Prepare CH-SIMS metadata and splits.

CH-SIMS provides per-clip multimodal **and** independent unimodal sentiment
labels — the unimodal labels are critical for X-MoFE's reliability supervision.

Reads ``meta.csv`` and the ``Raw/<video_id>/<clip_id>.mp4`` tree from
``data/raw/ch_sims/`` and writes ``data/interim/ch_sims/{metadata.jsonl,splits.json}``.

Examples
--------
    python scripts/data/prepare_ch_sims.py
    python scripts/data/prepare_ch_sims.py --meta-csv data/raw/ch_sims/meta.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.data import Sample, write_jsonl, write_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_ch_sims")

CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "ch_sims.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-csv", type=Path, default=None, help="Override meta.csv path.")
    parser.add_argument("--clips-root", type=Path, default=None, help="Override clips root.")
    parser.add_argument(
        "--skip-missing-clips",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="Drop samples whose mp4 is missing on disk (default: true).",
    )
    args = parser.parse_args()

    cfg = load_config()
    meta_csv = args.meta_csv or REPO_ROOT / cfg["paths"]["meta_csv"]
    clips_root = args.clips_root or REPO_ROOT / cfg["paths"]["clips_root"]
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    metadata_file = REPO_ROOT / cfg["paths"]["metadata_file"]
    splits_file = REPO_ROOT / cfg["paths"]["splits_file"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    if not meta_csv.exists():
        log.error(
            "meta.csv not found at %s. Run scripts/data/download_datasets.py --dataset ch_sims first.",
            meta_csv,
        )
        sys.exit(2)

    columns = cfg["csv_columns"]
    split_map = cfg["split_map"]
    template = cfg["sample_id_template"]

    samples: list[Sample] = []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    missing_clips = unknown_modes = 0

    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row[columns["mode"]].strip().lower()
            split = split_map.get(mode)
            if split is None:
                unknown_modes += 1
                continue

            video_id = row[columns["video_id"]].strip()
            clip_id = row[columns["clip_id"]].strip()
            sample_id = template.format(video_id=video_id, clip_id=clip_id)

            clip_path = clips_root / video_id / f"{clip_id}.mp4"
            if not clip_path.exists():
                missing_clips += 1
                if args.skip_missing_clips:
                    continue

            sentiment_M = parse_float(row[columns["multimodal_label"]])
            sentiment_T = parse_float(row.get(columns["text_label"], ""))
            sentiment_A = parse_float(row.get(columns["audio_label"], ""))
            sentiment_V = parse_float(row.get(columns["visual_label"], ""))

            if sentiment_M is None:
                log.warning("skipping %s: missing multimodal label", sample_id)
                continue

            video_path_rel = str(clip_path.relative_to(REPO_ROOT)) if clip_path.exists() else None

            sample = Sample(
                sample_id=sample_id,
                dataset="ch_sims",
                split=split,
                transcript=row[columns["text"]].strip(),
                audio_path=None,
                video_path=video_path_rel,
                primary_label=sentiment_M,
                task="regression",
                labels={
                    "sentiment_M": sentiment_M,
                    "sentiment_T": sentiment_T,
                    "sentiment_A": sentiment_A,
                    "sentiment_V": sentiment_V,
                    "annotation": row.get(columns["annotation"], "").strip() or None,
                },
                extra={"video_id": video_id, "clip_id": clip_id},
            )
            samples.append(sample)
            splits[split].append(sample_id)

    written = write_jsonl(metadata_file, samples)
    counts = write_splits(splits_file, splits)
    log.info("wrote %d samples to %s", written, metadata_file)
    log.info("split counts: %s", counts)
    if missing_clips:
        policy = "dropped" if args.skip_missing_clips else "kept (video_path=None)"
        log.warning("%d clip(s) missing on disk — %s", missing_clips, policy)
    if unknown_modes:
        log.warning("%d row(s) had unknown mode values — skipped", unknown_modes)


if __name__ == "__main__":
    main()
