"""Prepare CMU-MOSEI metadata and splits.

Reads the timestamped-words and labels CSD files from data/raw/CMU-MOSEI/,
joins each labelled utterance with the official train/valid/test folds, and
writes data/interim/mosei/{metadata.jsonl,splits.json}.

Granularity (``--granularity``):
  * **utterance** (default, ~22,856 samples) — one Sample per labelled
    utterance row. Sample IDs are ``"<video_id>[<utt_idx>]"`` and each
    sample carries the utterance's ``start_time``/``end_time`` so the
    audio/visual feature extractors can slice the per-video CSD sequences.
    This matches the field-standard CMU-MOSEI utterance protocol used by
    Self-MM/MISA/MMIM/MAG-BERT/ALMT (16,326 / 1,871 / 4,659 splits).
  * **video** (~3,225 samples, legacy) — one Sample per video, using only
    the *first* utterance's label/features. Kept for backward compatibility
    with the original pre-fix protocol. **Not recommended for new runs.**

The CSDs we read give us:
  * keys = ``<video_id>``
  * ``labels[vid]["features"]`` shape ``(N_utt, 7)`` —
    ``[sentiment, happy, sad, anger, surprise, disgust, fear]``
  * ``labels[vid]["intervals"]`` shape ``(N_utt, 2)`` — start/end seconds
  * ``language[vid]["features"]``/``intervals`` — per-word tokens + timestamps

Per-utterance transcripts are reconstructed by collecting words whose
midpoint falls within the utterance's ``[start, end]`` interval.

Examples
--------
    python scripts/data/prepare_mosei.py                              # utterance (default)
    python scripts/data/prepare_mosei.py --granularity video          # legacy
    python scripts/data/prepare_mosei.py --limit 50                   # debugging
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
    """Decode a 1-D iterable of word entries into a plain string."""
    parts: list[str] = []
    for word in words_features:
        # Word entry can be either a 1-D row from the words CSD (shape (1,))
        # or a bare string after we sliced by interval.
        token = word[0] if hasattr(word, "__len__") and len(word) > 0 and not isinstance(word, (bytes, str)) else word
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="replace")
        token = str(token).strip()
        if not token or token in {"sp", "<unk>", "<eos>", "<sos>"}:
            continue
        parts.append(token)
    return " ".join(parts)


def slice_words_by_interval(
    word_features,
    word_intervals,
    start_time: float,
    end_time: float,
) -> str:
    """Return the transcript of all words whose midpoint lies within ``[start, end]``.

    The CMU-MOSEI words CSD stores per-word start/end timestamps; this picks
    the subset that belongs to a given utterance interval.
    """
    if word_features is None or word_intervals is None or len(word_intervals) == 0:
        return ""
    import numpy as np
    word_intervals = np.asarray(word_intervals, dtype=float)
    midpoints = word_intervals.mean(axis=1)
    mask = (midpoints >= start_time) & (midpoints <= end_time)
    selected = word_features[mask] if hasattr(word_features, "__getitem__") else \
               [w for w, m in zip(word_features, mask) if m]
    return transcript_from_words(selected)


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
    parser.add_argument(
        "--granularity",
        choices=("utterance", "video"),
        default="utterance",
        help="utterance (default, field-standard ~22k samples) or video (legacy ~3.2k).",
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
    skipped = unfolded = empty_features = 0

    video_ids = list(dataset["labels"].keys())
    log.info(
        "found %d videos in labels CSD; granularity=%s", len(video_ids), args.granularity,
    )

    # Counter so --limit applies at the SAMPLE level (1 sample per utterance
    # in utterance mode, 1 per video in video mode). This keeps debugging
    # runs cheap regardless of granularity.
    n_samples_emitted = 0

    for video_id in video_ids:
        split = split_for_video(video_id, train_fold, valid_fold, test_fold)
        if split is None:
            unfolded += 1
            continue

        try:
            label_features = dataset["labels"][video_id]["features"]   # (N_utt, 7)
            label_intervals = dataset["labels"][video_id]["intervals"] # (N_utt, 2)
        except (KeyError, IndexError):
            skipped += 1
            continue
        if len(label_features) == 0:
            empty_features += 1
            continue

        # Words for this video (whole-video word stream)
        try:
            words_entry = dataset["language"][video_id]
            word_feats = words_entry["features"]
            word_intervals = words_entry["intervals"]
        except KeyError:
            word_feats = None
            word_intervals = None

        if video_id in video_lookup:
            video_path_rel = str(video_lookup[video_id].relative_to(REPO_ROOT))
        else:
            video_path_rel = None

        # Iterate utterances within this video (in utterance mode), or just
        # the first one (legacy video mode).
        utt_indices = range(len(label_features)) if args.granularity == "utterance" else range(1)
        for utt_idx in utt_indices:
            label_vector = label_features[utt_idx]
            utt_start, utt_end = float(label_intervals[utt_idx][0]), float(label_intervals[utt_idx][1])

            if args.granularity == "utterance":
                sample_id = f"{video_id}[{utt_idx}]"
                # Per-utterance transcript: slice the per-video word stream
                # by the utterance interval (word midpoint within [start, end]).
                transcript = slice_words_by_interval(
                    word_feats, word_intervals, utt_start, utt_end,
                )
                intervals_for_sample = [(utt_start, utt_end)]
            else:
                # video mode: keep sample_id = video_id (legacy behaviour)
                sample_id = video_id
                transcript = (
                    transcript_from_words(word_feats) if word_feats is not None else ""
                )
                intervals_for_sample = word_intervals

            sample = build_sample(
                segment_id=sample_id,
                video_id=video_id,
                split=split,
                transcript=transcript,
                label_vector=label_vector,
                intervals=intervals_for_sample,
                video_path_rel=video_path_rel,
            )
            samples.append(sample)
            splits[split].append(sample_id)
            n_samples_emitted += 1
            if args.limit is not None and n_samples_emitted >= args.limit:
                break
        if args.limit is not None and n_samples_emitted >= args.limit:
            break

    written = write_jsonl(metadata_file, samples)
    counts = write_splits(splits_file, splits)

    log.info("wrote %d samples to %s", written, metadata_file)
    log.info("split counts: %s", counts)
    if skipped:
        log.warning("skipped %d videos due to missing label features", skipped)
    if empty_features:
        log.warning("skipped %d videos with empty label feature arrays", empty_features)
    if unfolded:
        log.warning("skipped %d videos not present in any standard fold", unfolded)


if __name__ == "__main__":
    main()
