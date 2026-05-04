"""Prepare CMU-MOSEI metadata and splits.

Reads the timestamped-words and labels CSD files from data/raw/mosei/, walks
every aligned segment, joins it with the official train/valid/test folds, and
writes data/interim/mosei/{metadata.jsonl,splits.json}.

The CSD files give us:
  * segment ids of the form "<video_id>[<segment_idx>]"
  * per-segment word lists with start/end timestamps
  * per-segment 7-dim label vectors:
      [sentiment, happy, sad, anger, surprise, disgust, fear]

Raw mp4 video paths are populated only when ``video.root`` is set in the
config and a matching <video_id>.<ext> file exists on disk.

Examples
--------
    python scripts/data/prepare_mosei.py
    python scripts/data/prepare_mosei.py --raw-dir data/raw/mosei --limit 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.data import Sample, write_jsonl, write_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_mosei")

CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "mosei.yaml"

EMOTION_KEYS = ("happy", "sad", "anger", "surprise", "disgust", "fear")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_csd(raw_dir: Path, candidates: list[str]) -> Path:
    """Return the first existing CSD file from ``candidates`` under ``raw_dir``."""
    for name in candidates:
        for path in raw_dir.rglob(name):
            return path
    raise FileNotFoundError(
        f"None of {candidates} found under {raw_dir}. "
        f"Run scripts/data/download_datasets.py --dataset mosei first."
    )


def split_for_video(
    video_id: str,
    train_fold: set[str],
    valid_fold: set[str],
    test_fold: set[str],
) -> str | None:
    """Return our split name for a MOSEI video id, or None if it's unfolded."""
    if video_id in train_fold:
        return "train"
    if video_id in valid_fold:
        return "val"
    if video_id in test_fold:
        return "test"
    return None


def transcript_from_words(words_features) -> str:
    """Decode the timestamped-words feature array into a plain string."""
    parts: list[str] = []
    for word in words_features:
        token = word[0]
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="replace")
        token = str(token).strip()
        if not token or token in {"sp", "<unk>", "<eos>", "<sos>"}:
            continue
        parts.append(token)
    return " ".join(parts)


def build_sample(
    segment_id: str,
    video_id: str,
    split: str,
    transcript: str,
    label_vector,
    intervals,
    video_path_rel: str | None,
) -> Sample:
    sentiment = float(label_vector[0])
    emotions = {key: float(label_vector[i + 1]) for i, key in enumerate(EMOTION_KEYS)}

    if intervals is not None and len(intervals) > 0:
        start_time = float(intervals[0][0])
        end_time = float(intervals[-1][1])
        duration = end_time - start_time
    else:
        start_time = end_time = duration = None

    sentiment_binary = 1 if sentiment > 0 else 0
    sentiment_7class = max(-3, min(3, round(sentiment))) + 3  # shift to 0..6

    return Sample(
        sample_id=segment_id,
        dataset="mosei",
        split=split,
        transcript=transcript,
        audio_path=None,  # extracted from video during Phase 2
        video_path=video_path_rel,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        primary_label=sentiment,
        task="regression",
        labels={
            "sentiment_regression": sentiment,
            "sentiment_binary": sentiment_binary,
            "sentiment_7class": sentiment_7class,
            "emotions": emotions,
        },
        extra={"video_id": video_id},
    )


def discover_video_lookup(video_root: Path | None, extension: str) -> dict[str, Path]:
    """Map video_id -> absolute mp4 path, if a video root is configured."""
    if video_root is None or not video_root.exists():
        return {}
    return {p.stem: p for p in video_root.rglob(f"*{extension}")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw data directory (default from configs/datasets/mosei.yaml).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Optional directory containing CMU-MOSEI raw mp4 videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N segments (debugging).",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = args.raw_dir or REPO_ROOT / cfg["paths"]["raw_dir"]
    interim_dir = REPO_ROOT / cfg["paths"]["interim_dir"]
    metadata_file = REPO_ROOT / cfg["paths"]["metadata_file"]
    splits_file = REPO_ROOT / cfg["paths"]["splits_file"]

    interim_dir.mkdir(parents=True, exist_ok=True)

    # mmsdk is heavy; import lazily so --help works without it installed.
    try:
        from mmsdk import mmdatasdk as md
        from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import (
            cmu_mosei_std_folds as std_folds,
        )
    except ImportError as e:
        log.error("CMU-MultimodalSDK is required: %s", e)
        log.error("Install with: pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git")
        sys.exit(2)

    words_csd = find_csd(raw_dir, cfg["csd_files"]["language"])
    labels_csd = find_csd(raw_dir, cfg["csd_files"]["labels"])
    log.info("words CSD: %s", words_csd)
    log.info("labels CSD: %s", labels_csd)

    dataset = md.mmdataset({"language": str(words_csd), "labels": str(labels_csd)})

    train_fold = set(std_folds.standard_train_fold)
    valid_fold = set(std_folds.standard_valid_fold)
    test_fold = set(std_folds.standard_test_fold)
    log.info(
        "standard folds: train=%d valid=%d test=%d",
        len(train_fold),
        len(valid_fold),
        len(test_fold),
    )

    video_root_cfg = cfg.get("video", {}).get("root")
    video_root = args.video_root or (REPO_ROOT / video_root_cfg if video_root_cfg else None)
    video_extension = cfg.get("video", {}).get("extension", ".mp4")
    video_lookup = discover_video_lookup(video_root, video_extension)
    if video_lookup:
        log.info("found %d raw videos under %s", len(video_lookup), video_root)
    else:
        log.info("no raw videos available; video_path will be None for every sample")

    samples: list[Sample] = []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    skipped = unfolded = 0

    segment_ids = list(dataset["labels"].keys())
    log.info("found %d segments in labels CSD", len(segment_ids))
    if args.limit is not None:
        segment_ids = segment_ids[: args.limit]

    for segment_id in segment_ids:
        video_id = segment_id.split("[")[0]
        split = split_for_video(video_id, train_fold, valid_fold, test_fold)
        if split is None:
            unfolded += 1
            continue

        try:
            label_vector = dataset["labels"][segment_id]["features"][0]
        except (KeyError, IndexError):
            skipped += 1
            continue

        try:
            words_entry = dataset["language"][segment_id]
            transcript = transcript_from_words(words_entry["features"])
            intervals = words_entry["intervals"]
        except KeyError:
            transcript = ""
            intervals = None

        if video_id in video_lookup:
            video_path_rel = str(video_lookup[video_id].relative_to(REPO_ROOT))
        else:
            video_path_rel = None

        sample = build_sample(
            segment_id=segment_id,
            video_id=video_id,
            split=split,
            transcript=transcript,
            label_vector=label_vector,
            intervals=intervals,
            video_path_rel=video_path_rel,
        )
        samples.append(sample)
        splits[split].append(segment_id)

    written = write_jsonl(metadata_file, samples)
    counts = write_splits(splits_file, splits)

    log.info("wrote %d samples to %s", written, metadata_file)
    log.info("split counts: %s", counts)
    if skipped:
        log.warning("skipped %d segments due to missing label features", skipped)
    if unfolded:
        log.warning("skipped %d segments not present in any standard fold", unfolded)


if __name__ == "__main__":
    main()
