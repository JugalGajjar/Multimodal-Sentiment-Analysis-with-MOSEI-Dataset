"""Extract visual features for prepared datasets.

Two backends, dispatched by dataset config:

* ``source: video`` (default — MELD, CH-SIMS):
    Sample 16 frames uniformly per clip with PyAV, then run frozen
    ``MCG-NJU/videomae-base`` to produce ``(1568, 768)`` patch sequences.

* ``source: csd`` (CMU-MOSEI):
    Read pre-extracted OpenFace2 sequences from
    ``CMU_MOSEI_VisualOpenFace2.csd`` and slice per ``video_id``.

Both backends emit the standard cache schema (see :mod:`src.data.features`)
so the merger and trainer treat them uniformly.

Examples
--------
    python scripts/features/extract_visual_features.py
    python scripts/features/extract_visual_features.py --dataset meld --batch-size 1
    python scripts/features/extract_visual_features.py --dataset mosei
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data import (  # noqa: E402
    VALID_DATASETS,
    VALID_SPLITS,
    Sample,
    cache_path,
    read_feature_cache,
    read_jsonl,
    write_feature_cache,
)
from src.encoders import (  # noqa: E402
    OpenFace2SequenceReader,
    VideoMAEEncoder,
    load_video_frames,
    resolve_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_visual")

DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
ENCODER_CONFIG_PATH = REPO_ROOT / "configs" / "encoders" / "videomae.yaml"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

CACHE_FRESHNESS_KEYS = (
    "encoder_name",
    "encoder_source",
    "feature_dim",
    "max_length",
    "dtype",
    "num_samples",
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_config(dataset: str) -> dict:
    return load_yaml(DATASET_CONFIG_DIR / f"{dataset}.yaml")


def load_split_samples(dataset_cfg: dict, dataset: str) -> dict[str, list[Sample]]:
    metadata_file = REPO_ROOT / dataset_cfg["paths"]["metadata_file"]
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found. Run scripts/data/prepare_{dataset}.py first."
        )
    by_split: dict[str, list[Sample]] = defaultdict(list)
    for sample in read_jsonl(metadata_file):
        by_split[sample.split].append(sample)
    return by_split


def merge_overrides(encoder_cfg: dict, dataset_cfg: dict) -> dict:
    overrides = (dataset_cfg.get("encoder_overrides") or {}).get("visual") or {}
    merged = {**encoder_cfg, **overrides}
    if overrides:
        log.info("applied visual encoder overrides: %s", overrides)
    return merged


def cache_is_fresh(out_path: Path, expected: dict) -> bool:
    if not out_path.exists():
        return False
    try:
        existing = read_feature_cache(out_path)["metadata"]
    except Exception as e:  # noqa: BLE001
        log.warning("could not read existing cache %s (%s); will re-extract", out_path, e)
        return False
    for key in CACHE_FRESHNESS_KEYS:
        if existing.get(key) != expected.get(key):
            log.info(
                "cache %s stale on %s: existing=%r expected=%r",
                out_path, key, existing.get(key), expected.get(key),
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Backend: VideoMAE over raw mp4 frames (MELD, CH-SIMS)
# ---------------------------------------------------------------------------


def encode_split_videomae(
    encoder: VideoMAEEncoder,
    samples: list[Sample],
    batch_size: int,
    cache_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sample_ids = [s.sample_id for s in samples]
    chunks: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="encode", unit="batch"):
        batch = samples[start : start + batch_size]
        frame_arrays = []
        for s in batch:
            if not s.video_path:
                # Same blank fallback shape as load_video_frames produces on failure
                import numpy as np
                frame_arrays.append(
                    np.zeros((encoder.num_frames, encoder.image_size, encoder.image_size, 3), dtype=np.uint8)
                )
                continue
            frames = load_video_frames(
                REPO_ROOT / s.video_path,
                num_frames=encoder.num_frames,
                image_size=encoder.image_size,
            )
            frame_arrays.append(frames)
        feats, lens = encoder.encode(frame_arrays)
        chunks.append(feats.to(cache_dtype))
        all_lengths.append(lens)

    return sample_ids, torch.cat(chunks, dim=0), torch.cat(all_lengths, dim=0)


# ---------------------------------------------------------------------------
# Backend: OpenFace2 CSD passthrough (MOSEI)
# ---------------------------------------------------------------------------


def encode_split_openface(
    reader: OpenFace2SequenceReader,
    samples: list[Sample],
    batch_size: int,
    cache_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sample_ids = [s.sample_id for s in samples]
    chunks: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="encode", unit="batch"):
        batch = samples[start : start + batch_size]
        # When start/end_time are populated (utterance-level samples), pass
        # them as intervals so the per-video OpenFace2 sequence is sliced
        # to the utterance boundary. Falls through to per-video on absence.
        video_ids = [s.extra.get("video_id") or s.sample_id.split("[")[0] for s in batch]
        intervals = [
            (s.start_time, s.end_time)
            if (s.start_time is not None and s.end_time is not None)
            else None
            for s in batch
        ]
        if any(iv is not None for iv in intervals):
            feats, lens = reader.encode(video_ids, intervals=intervals)
        else:
            feats, lens = reader.encode(video_ids)
        chunks.append(feats.to(cache_dtype))
        all_lengths.append(lens)

    return sample_ids, torch.cat(chunks, dim=0), torch.cat(all_lengths, dim=0)


# ---------------------------------------------------------------------------
# Main per-dataset extraction
# ---------------------------------------------------------------------------


def extract_dataset(
    dataset: str,
    base_encoder_cfg: dict,
    batch_size_override: int | None,
    skip_existing: bool,
    device: str,
    videomae_encoder: VideoMAEEncoder | None = None,
) -> VideoMAEEncoder | None:
    log.info("=== %s ===", dataset)
    dataset_cfg = load_dataset_config(dataset)
    encoder_cfg = merge_overrides(base_encoder_cfg, dataset_cfg)
    cache_dtype = DTYPE_MAP[encoder_cfg["cache_dtype"]]
    batch_size = batch_size_override or encoder_cfg["batch_size"]

    by_split = load_split_samples(dataset_cfg, dataset)

    source = encoder_cfg.get("source", "video")
    if source == "csd":
        csd_path = REPO_ROOT / encoder_cfg["csd_path"]
        if not csd_path.exists():
            raise FileNotFoundError(f"OpenFace2 CSD not found: {csd_path}")
        reader = OpenFace2SequenceReader(
            csd_path=csd_path,
            max_frames=encoder_cfg["max_frames"],
            sampling_rate=encoder_cfg["sampling_rate"],
            feature_dim=encoder_cfg["feature_dim"],
        )
        encoder_name = encoder_cfg.get("encoder_name", "openface2")
        encoder_source = encoder_cfg.get("encoder_source", str(csd_path.name))
        feature_dim = reader.feature_dim
        max_length = encoder_cfg["max_frames"]
    else:  # video → VideoMAE
        if videomae_encoder is None:
            videomae_encoder = VideoMAEEncoder(
                model_name=encoder_cfg["model_name"],
                num_frames=encoder_cfg["num_frames"],
                image_size=encoder_cfg["image_size"],
                device=device,
            )
        encoder_name = encoder_cfg["name"]
        encoder_source = encoder_cfg["model_name"]
        feature_dim = videomae_encoder.feature_dim
        max_length = videomae_encoder.num_patches

    for split in VALID_SPLITS:
        samples = by_split.get(split, [])
        if not samples:
            log.info("[%s/%s] no samples — skip", dataset, split)
            continue

        out_path = cache_path(PROCESSED_ROOT, dataset, "visual", split)
        expected_metadata = {
            "modality": "visual",
            "dataset": dataset,
            "split": split,
            "encoder_name": encoder_name,
            "encoder_source": encoder_source,
            "feature_dim": feature_dim,
            "max_length": max_length,
            "dtype": encoder_cfg["cache_dtype"],
            "num_samples": len(samples),
        }
        if skip_existing and cache_is_fresh(out_path, expected_metadata):
            log.info("[%s/%s] %s up-to-date — skip", dataset, split, out_path)
            continue

        log.info(
            "[%s/%s] encoding %d samples (source=%s, max_length=%d) → %s",
            dataset, split, len(samples), source, max_length, out_path,
        )
        if source == "csd":
            sample_ids, features, lengths = encode_split_openface(
                reader, samples, batch_size, cache_dtype
            )
        else:
            sample_ids, features, lengths = encode_split_videomae(
                videomae_encoder, samples, batch_size, cache_dtype,
            )
        write_feature_cache(out_path, sample_ids, features, lengths, expected_metadata)
        log.info("[%s/%s] wrote %s (%.1f MB)", dataset, split, out_path, out_path.stat().st_size / 1e6)

    return videomae_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=(*VALID_DATASETS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip splits whose cache is already fresh (default).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-extract even when a cache file already exists and matches.",
    )
    args = parser.parse_args()

    encoder_cfg = load_yaml(ENCODER_CONFIG_PATH)
    device = resolve_device(args.device)
    log.info("device=%s default_encoder=%s", device, encoder_cfg["model_name"])

    targets = VALID_DATASETS if args.dataset == "all" else (args.dataset,)
    failures: list[str] = []
    videomae_encoder: VideoMAEEncoder | None = None

    for ds in targets:
        try:
            videomae_encoder = extract_dataset(
                ds, encoder_cfg, args.batch_size, args.skip_existing,
                str(device), videomae_encoder=videomae_encoder,
            )
        except FileNotFoundError as e:
            log.error("[%s] %s", ds, e)
            failures.append(ds)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] failed: %s", ds, e)
            failures.append(ds)

    if failures:
        log.error("failures: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
