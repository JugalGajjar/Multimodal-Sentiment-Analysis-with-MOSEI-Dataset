"""Extract audio features for prepared datasets.

Two backends, dispatched by dataset config:

* ``source: video`` (default — MELD, CH-SIMS):
    Decode audio from each sample's ``video_path`` with PyAV, then run frozen
    ``microsoft/wavlm-base-plus`` over the 16 kHz mono waveform.

* ``source: csd`` (CMU-MOSEI):
    Read pre-extracted COVAREP sequences from ``CMU_MOSEI_COVAREP.csd`` and
    slice per ``video_id``.

Both backends emit the standard cache schema (see :mod:`src.data.features`)
so the downstream merger and trainer treat them uniformly.

Examples
--------
    python scripts/features/extract_audio_features.py
    python scripts/features/extract_audio_features.py --dataset meld --batch-size 8
    python scripts/features/extract_audio_features.py --dataset mosei
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
    COVAREPSequenceReader,
    WavLMEncoder,
    load_audio_from_video,
    resolve_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_audio")

DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
ENCODER_CONFIG_PATH = REPO_ROOT / "configs" / "encoders" / "wavlm.yaml"
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
    overrides = (dataset_cfg.get("encoder_overrides") or {}).get("audio") or {}
    merged = {**encoder_cfg, **overrides}
    if overrides:
        log.info("applied audio encoder overrides: %s", overrides)
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
# Backend: WavLM over raw mp4 audio (MELD, CH-SIMS)
# ---------------------------------------------------------------------------


def encode_split_wavlm(
    encoder: WavLMEncoder,
    samples: list[Sample],
    batch_size: int,
    cache_dtype: torch.dtype,
    sampling_rate: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sample_ids = [s.sample_id for s in samples]
    chunks: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="encode", unit="batch"):
        batch = samples[start : start + batch_size]
        waveforms = []
        for s in batch:
            if not s.video_path:
                waveforms.append(__import__("numpy").zeros(0, dtype="float32"))
                continue
            path = REPO_ROOT / s.video_path
            waveforms.append(load_audio_from_video(path, target_sr=sampling_rate))
        feats, lens = encoder.encode(waveforms)
        chunks.append(feats.to(cache_dtype))
        all_lengths.append(lens)

    return sample_ids, torch.cat(chunks, dim=0), torch.cat(all_lengths, dim=0)


# ---------------------------------------------------------------------------
# Backend: COVAREP CSD passthrough (MOSEI)
# ---------------------------------------------------------------------------


def encode_split_covarep(
    reader: COVAREPSequenceReader,
    samples: list[Sample],
    batch_size: int,
    cache_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sample_ids = [s.sample_id for s in samples]
    chunks: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="encode", unit="batch"):
        batch = samples[start : start + batch_size]
        # MOSEI samples carry video_id in extra (set by prepare_mosei.py).
        # When start/end_time are populated (utterance-level samples), pass
        # them as intervals so the per-video COVAREP sequence is sliced
        # down to the utterance boundary. Falls through to per-video
        # sequences when intervals are absent (legacy video-level path).
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
    wavlm_encoder: WavLMEncoder | None = None,
) -> WavLMEncoder | None:
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
            raise FileNotFoundError(f"COVAREP CSD not found: {csd_path}")
        reader = COVAREPSequenceReader(
            csd_path=csd_path,
            max_frames=encoder_cfg["max_frames"],
            sampling_rate=encoder_cfg["sampling_rate"],
        )
        encoder_name = encoder_cfg.get("encoder_name", "covarep")
        encoder_source = encoder_cfg.get("encoder_source", str(csd_path.name))
        feature_dim = reader.feature_dim
        max_length = encoder_cfg["max_frames"]
    else:  # video → WavLM
        if wavlm_encoder is None:
            wavlm_encoder = WavLMEncoder(
                model_name=encoder_cfg["model_name"],
                sampling_rate=encoder_cfg["sampling_rate"],
                max_duration_seconds=encoder_cfg["max_duration_seconds"],
                device=device,
            )
        encoder_name = encoder_cfg["name"]
        encoder_source = encoder_cfg["model_name"]
        feature_dim = wavlm_encoder.feature_dim
        # WavLM output L_max = _get_feat_extract_output_lengths(max_samples)
        max_input_samples = wavlm_encoder.max_samples
        max_length = int(
            wavlm_encoder.model._get_feat_extract_output_lengths(
                torch.tensor([max_input_samples])
            ).item()
        )

    for split in VALID_SPLITS:
        samples = by_split.get(split, [])
        if not samples:
            log.info("[%s/%s] no samples — skip", dataset, split)
            continue

        out_path = cache_path(PROCESSED_ROOT, dataset, "audio", split)
        expected_metadata = {
            "modality": "audio",
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
            sample_ids, features, lengths = encode_split_covarep(
                reader, samples, batch_size, cache_dtype
            )
        else:
            sample_ids, features, lengths = encode_split_wavlm(
                wavlm_encoder, samples, batch_size, cache_dtype,
                sampling_rate=encoder_cfg["sampling_rate"],
            )
        write_feature_cache(out_path, sample_ids, features, lengths, expected_metadata)
        log.info("[%s/%s] wrote %s (%.1f MB)", dataset, split, out_path, out_path.stat().st_size / 1e6)

    return wavlm_encoder


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
    wavlm_encoder: WavLMEncoder | None = None  # lazy-loaded on first WavLM dataset

    for ds in targets:
        try:
            wavlm_encoder = extract_dataset(
                ds, encoder_cfg, args.batch_size, args.skip_existing,
                str(device), wavlm_encoder=wavlm_encoder,
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
