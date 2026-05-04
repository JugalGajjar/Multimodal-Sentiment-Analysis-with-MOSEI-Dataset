"""Extract ModernBERT text features for prepared datasets.

Reads ``data/interim/<dataset>/{metadata.jsonl,splits.json}``, runs the frozen
ModernBERT encoder over every transcript, and writes per-split caches at
``data/processed/<dataset>/text_features/<split>.pt`` using the schema
defined in :mod:`src.data.features`.

Examples
--------
    python scripts/features/extract_text_features.py
    python scripts/features/extract_text_features.py --dataset mosei
    python scripts/features/extract_text_features.py --dataset meld --batch-size 64 --device cpu
    python scripts/features/extract_text_features.py --dataset all --skip-existing
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
from src.encoders import ModernBERTEncoder, resolve_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_text")

DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
ENCODER_CONFIG_PATH = REPO_ROOT / "configs" / "encoders" / "modernbert.yaml"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_config(dataset: str) -> dict:
    return load_yaml(DATASET_CONFIG_DIR / f"{dataset}.yaml")


def load_split_samples(dataset_cfg: dict, dataset: str) -> dict[str, list[Sample]]:
    """Group prepared samples by split for a dataset."""
    metadata_file = REPO_ROOT / dataset_cfg["paths"]["metadata_file"]
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found. Run scripts/data/prepare_{dataset}.py first."
        )
    by_split: dict[str, list[Sample]] = defaultdict(list)
    for sample in read_jsonl(metadata_file):
        by_split[sample.split].append(sample)
    return by_split


def merge_encoder_overrides(encoder_cfg: dict, dataset_cfg: dict, modality: str) -> dict:
    """Merge dataset-level overrides into the encoder config for a modality."""
    overrides = (dataset_cfg.get("encoder_overrides") or {}).get(modality) or {}
    merged = {**encoder_cfg, **overrides}
    if overrides:
        log.info("applied %s encoder overrides: %s", modality, overrides)
    return merged


CACHE_FRESHNESS_KEYS = (
    "encoder_name",
    "encoder_source",
    "max_length",
    "feature_dim",
    "dtype",
    "num_samples",
)


def cache_is_fresh(out_path: Path, expected: dict) -> bool:
    """True iff an existing cache's metadata matches what we'd write now."""
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


def encode_split(
    encoder: ModernBERTEncoder,
    samples: list[Sample],
    batch_size: int,
    cache_dtype: torch.dtype,
    max_length: int | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Run the encoder over every sample in ``samples`` (in metadata order)."""
    sample_ids = [s.sample_id for s in samples]
    transcripts = [s.transcript for s in samples]

    chunks: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    for start in tqdm(range(0, len(transcripts), batch_size), desc="encode", unit="batch"):
        batch = transcripts[start : start + batch_size]
        feats, lens = encoder.encode(batch, max_length=max_length)
        chunks.append(feats.to(cache_dtype))
        lengths.append(lens)

    features = torch.cat(chunks, dim=0)
    lengths_tensor = torch.cat(lengths, dim=0)
    return sample_ids, features, lengths_tensor


def extract_dataset(
    dataset: str,
    encoder: ModernBERTEncoder,
    base_encoder_cfg: dict,
    batch_size_override: int | None,
    skip_existing: bool,
) -> None:
    log.info("=== %s ===", dataset)
    dataset_cfg = load_dataset_config(dataset)
    encoder_cfg = merge_encoder_overrides(base_encoder_cfg, dataset_cfg, "text")
    cache_dtype = DTYPE_MAP[encoder_cfg["cache_dtype"]]
    max_length = encoder_cfg["max_length"]
    batch_size = batch_size_override or encoder_cfg["batch_size"]

    by_split = load_split_samples(dataset_cfg, dataset)

    for split in VALID_SPLITS:
        samples = by_split.get(split, [])
        if not samples:
            log.info("[%s/%s] no samples — skip", dataset, split)
            continue

        out_path = cache_path(PROCESSED_ROOT, dataset, "text", split)
        expected_metadata = {
            "modality": "text",
            "dataset": dataset,
            "split": split,
            "encoder_name": encoder_cfg["name"],
            "encoder_source": encoder_cfg["model_name"],
            "feature_dim": encoder.feature_dim,
            "max_length": max_length,
            "dtype": encoder_cfg["cache_dtype"],
            "num_samples": len(samples),
        }
        if skip_existing and cache_is_fresh(out_path, expected_metadata):
            log.info("[%s/%s] %s up-to-date — skip", dataset, split, out_path)
            continue

        log.info(
            "[%s/%s] encoding %d samples (max_length=%d) → %s",
            dataset, split, len(samples), max_length, out_path,
        )
        sample_ids, features, lengths = encode_split(
            encoder, samples, batch_size, cache_dtype, max_length=max_length,
        )
        write_feature_cache(out_path, sample_ids, features, lengths, expected_metadata)
        log.info("[%s/%s] wrote %s (%.1f MB)", dataset, split, out_path, out_path.stat().st_size / 1e6)


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
        help="Skip splits whose cache already exists (default).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-extract even when a cache file already exists.",
    )
    args = parser.parse_args()

    encoder_cfg = load_yaml(ENCODER_CONFIG_PATH)
    device = resolve_device(args.device)
    log.info("device=%s encoder=%s", device, encoder_cfg["model_name"])

    encoder = ModernBERTEncoder(
        model_name=encoder_cfg["model_name"],
        max_length=encoder_cfg["max_length"],
        padding=encoder_cfg["padding"],
        device=str(device),
    )

    targets = VALID_DATASETS if args.dataset == "all" else (args.dataset,)
    failures: list[str] = []
    for ds in targets:
        try:
            extract_dataset(ds, encoder, encoder_cfg, args.batch_size, args.skip_existing)
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
