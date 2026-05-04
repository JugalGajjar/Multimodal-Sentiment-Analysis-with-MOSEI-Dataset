"""Prepare MELD metadata and splits.

Reads the official MELD CSVs and clip directories from data/raw/meld/MELD.Raw/
and writes data/interim/meld/{metadata.jsonl,splits.json}.

Each MELD utterance is one mp4 clip named ``dia<DialogueId>_utt<UtteranceId>.mp4``.
The CSV columns we rely on:

    Sr No., Utterance, Speaker, Emotion, Sentiment,
    Dialogue_ID, Utterance_ID, Season, Episode,
    StartTime, EndTime

A handful of clips are missing from MELD's release (a known issue); samples
that point at non-existent mp4s are dropped with a warning.

Examples
--------
    python scripts/data/prepare_meld.py
    python scripts/data/prepare_meld.py --raw-root data/raw/meld/MELD.Raw --skip-missing-clips=false
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
log = logging.getLogger("prepare_meld")

CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "meld.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def emotion_index(label: str, emotions: list[str]) -> int:
    norm = label.strip().lower()
    for i, e in enumerate(emotions):
        if e.lower() == norm:
            return i
    raise ValueError(f"Unknown emotion {label!r}; expected one of {emotions}")


def sentiment_index(label: str, sentiments: list[str]) -> int:
    norm = label.strip().lower()
    for i, s in enumerate(sentiments):
        if s.lower() == norm:
            return i
    raise ValueError(f"Unknown sentiment {label!r}; expected one of {sentiments}")


def parse_time(value: str) -> float | None:
    """MELD start/end times look like ``00:01:23,456``. Return seconds or None."""
    if not value:
        return None
    value = value.strip().replace(",", ".")
    if not value or value.lower() == "nan":
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=None, help="Override MELD.Raw root.")
    parser.add_argument(
        "--skip-missing-clips",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="Drop samples whose mp4 is missing on disk (default: true).",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw_root = args.raw_root or REPO_ROOT / cfg["paths"]["raw_root"]
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    metadata_file = REPO_ROOT / cfg["paths"]["metadata_file"]
    splits_file = REPO_ROOT / cfg["paths"]["splits_file"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        log.error(
            "MELD.Raw not found at %s. Run scripts/data/download_datasets.py --dataset meld first.",
            raw_root,
        )
        sys.exit(2)

    emotions: list[str] = cfg["labels"]["emotions"]
    sentiments: list[str] = cfg["labels"]["sentiments"]
    sample_id_template: str = cfg["sample_id_template"]
    clip_filename_template: str = cfg["clip_filename_template"]

    samples: list[Sample] = []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    missing_clips: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split, spec in cfg["splits"].items():
        csv_path = raw_root / spec["csv"]
        clips_dir = raw_root / spec["clips_dir"]
        if not csv_path.exists():
            log.error("CSV missing: %s", csv_path)
            sys.exit(2)
        if not clips_dir.exists():
            log.error("Clips dir missing: %s", clips_dir)
            sys.exit(2)
        log.info("[%s] csv=%s clips=%s", split, csv_path.name, clips_dir.name)

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = int(row["Dialogue_ID"])
                utterance_id = int(row["Utterance_ID"])
                sample_id = sample_id_template.format(
                    split=split, dialogue_id=dialogue_id, utterance_id=utterance_id
                )
                clip_filename = clip_filename_template.format(
                    dialogue_id=dialogue_id, utterance_id=utterance_id
                )
                clip_path = clips_dir / clip_filename
                if not clip_path.exists():
                    missing_clips[split] += 1
                    if args.skip_missing_clips:
                        continue

                emotion = row["Emotion"]
                sentiment = row["Sentiment"]
                try:
                    emo_id = emotion_index(emotion, emotions)
                    sent_id = sentiment_index(sentiment, sentiments)
                except ValueError as err:
                    log.warning("[%s] skipping %s: %s", split, sample_id, err)
                    continue

                start = parse_time(row.get("StartTime", ""))
                end = parse_time(row.get("EndTime", ""))
                duration = (end - start) if (start is not None and end is not None) else None

                video_path_rel = str(clip_path.relative_to(REPO_ROOT)) if clip_path.exists() else None

                sample = Sample(
                    sample_id=sample_id,
                    dataset="meld",
                    split=split,
                    transcript=row["Utterance"].strip(),
                    audio_path=None,
                    video_path=video_path_rel,
                    start_time=start,
                    end_time=end,
                    duration=duration,
                    primary_label=emo_id,
                    task="classification",
                    labels={
                        "emotion": emotions[emo_id],
                        "emotion_id": emo_id,
                        "sentiment": sentiments[sent_id],
                        "sentiment_id": sent_id,
                    },
                    speaker_id=row.get("Speaker", "").strip() or None,
                    dialogue_id=str(dialogue_id),
                    utterance_index=utterance_id,
                    extra={
                        "season": row.get("Season"),
                        "episode": row.get("Episode"),
                    },
                )
                samples.append(sample)
                splits[split].append(sample_id)

    written = write_jsonl(metadata_file, samples)
    counts = write_splits(splits_file, splits)
    log.info("wrote %d samples to %s", written, metadata_file)
    log.info("split counts: %s", counts)
    for split, n in missing_clips.items():
        if n:
            policy = "dropped" if args.skip_missing_clips else "kept (video_path=None)"
            log.warning("[%s] %d clip(s) missing on disk — %s", split, n, policy)


if __name__ == "__main__":
    main()
