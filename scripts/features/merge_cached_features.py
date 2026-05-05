"""Merge per-modality feature caches into per-split manifests.

For each dataset and split, this script produces:

* ``data/processed/<dataset>/<split>.pt`` — a manifest dict bundling the
  canonical sample order, per-sample labels (primary + rich), and references
  to the per-modality caches for that split. Trainers load this manifest
  plus the underlying modality caches; the manifest itself is small (~1 MB
  even for MELD) so this avoids duplicating the multi-GB feature tensors.

* ``data/processed/<dataset>/metadata.json`` — a top-level dataset summary
  (split sizes, task, encoder dims, label statistics).

Manifest schema::

    {
        "dataset": str,
        "split": str,
        "task": "regression" | "classification",
        "sample_ids": list[str],          # canonical order from metadata.jsonl
        "primary_labels": Tensor (N,),    # float for regression, int for classification
        "labels": list[dict],             # rich Sample.labels per sample
        "modalities": {
            "text":   {"cache_path": "text_features/<split>.pt",   "feature_dim": ..., "max_length": ..., ...},
            "audio":  {"cache_path": "audio_features/<split>.pt",  ...},
            "visual": {"cache_path": "visual_features/<split>.pt", ...},
        },
        "created_at": str,
    }

The script verifies that every modality cache contains exactly the sample
ids in the metadata (no orphans, no missing) and aborts noisily otherwise.

Examples
--------
    python scripts/features/merge_cached_features.py
    python scripts/features/merge_cached_features.py --dataset meld
    python scripts/features/merge_cached_features.py --dataset all --no-skip-existing
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.data import (  # noqa: E402
    VALID_DATASETS,
    VALID_SPLITS,
    Sample,
    cache_path,
    read_feature_cache,
    read_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_features")

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
MODALITIES = ("text", "audio", "visual")

MODALITY_METADATA_KEYS = (
    "encoder_name",
    "encoder_source",
    "feature_dim",
    "max_length",
    "dtype",
    "num_samples",
)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def load_split_samples(dataset: str) -> dict[str, list[Sample]]:
    metadata_file = INTERIM_ROOT / dataset / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found. Run scripts/data/prepare_{dataset}.py first."
        )
    by_split: dict[str, list[Sample]] = defaultdict(list)
    for sample in read_jsonl(metadata_file):
        by_split[sample.split].append(sample)
    return by_split


def primary_label_dtype(task: str) -> torch.dtype:
    return torch.float32 if task == "regression" else torch.int64


def merge_split(dataset: str, split: str, samples: list[Sample], skip_existing: bool) -> dict | None:
    """Build and persist the manifest for one split. Returns None if skipped."""
    out_path = PROCESSED_ROOT / dataset / f"{split}.pt"
    canonical_ids = [s.sample_id for s in samples]

    # Verify modality caches exist and align with the canonical sample order.
    modality_info: dict[str, dict] = {}
    for modality in MODALITIES:
        cpath = cache_path(PROCESSED_ROOT, dataset, modality, split)
        if not cpath.exists():
            raise FileNotFoundError(
                f"Missing {modality} cache: {cpath}. "
                f"Run scripts/features/extract_{modality}_features.py --dataset {dataset} first."
            )
        cache = read_feature_cache(cpath)
        cache_ids = cache["sample_ids"]
        if cache_ids != canonical_ids:
            cache_set = set(cache_ids)
            canonical_set = set(canonical_ids)
            only_cache = cache_set - canonical_set
            only_meta = canonical_set - cache_set
            extras = []
            if only_cache:
                extras.append(f"{len(only_cache)} only in cache (e.g. {sorted(only_cache)[:3]})")
            if only_meta:
                extras.append(f"{len(only_meta)} only in metadata (e.g. {sorted(only_meta)[:3]})")
            if not extras:
                extras.append("ids match but order differs")
            raise ValueError(
                f"[{dataset}/{split}] {modality} cache sample_ids disagree with metadata: "
                + "; ".join(extras)
            )
        meta = {k: cache["metadata"][k] for k in MODALITY_METADATA_KEYS}
        meta["cache_path"] = str(cpath.relative_to(PROCESSED_ROOT / dataset))
        modality_info[modality] = meta

    # Resolve task. All samples in a split share the same task per dataset
    # convention; sanity-check it.
    tasks = {s.task for s in samples}
    if len(tasks) != 1:
        raise ValueError(f"[{dataset}/{split}] mixed tasks within split: {tasks}")
    task = tasks.pop()

    primary_labels = torch.tensor(
        [s.primary_label for s in samples], dtype=primary_label_dtype(task)
    )

    manifest = {
        "dataset": dataset,
        "split": split,
        "task": task,
        "sample_ids": canonical_ids,
        "primary_labels": primary_labels,
        "labels": [s.labels for s in samples],
        "modalities": modality_info,
        "created_at": now_iso(),
    }

    if skip_existing and out_path.exists():
        try:
            existing = torch.load(out_path, map_location="cpu", weights_only=False)
            same = (
                existing.get("sample_ids") == manifest["sample_ids"]
                and existing.get("task") == manifest["task"]
                and torch.equal(existing.get("primary_labels"), manifest["primary_labels"])
                and existing.get("modalities") == manifest["modalities"]
            )
            if same:
                log.info("[%s/%s] %s up-to-date — skip", dataset, split, out_path)
                return manifest
        except Exception as e:  # noqa: BLE001
            log.warning("could not read existing manifest %s (%s); rewriting", out_path, e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, out_path)
    log.info("[%s/%s] wrote %s (%.2f MB)", dataset, split, out_path, out_path.stat().st_size / 1e6)
    return manifest


def write_dataset_metadata(dataset: str, manifests: list[dict]) -> Path:
    """Aggregate per-split manifests into a top-level metadata.json."""
    out_path = PROCESSED_ROOT / dataset / "metadata.json"

    splits_summary = {m["split"]: len(m["sample_ids"]) for m in manifests}
    tasks = {m["task"] for m in manifests}
    if len(tasks) != 1:
        raise ValueError(f"[{dataset}] manifests disagree on task: {tasks}")
    task = tasks.pop()

    # Modality info should match across splits (same encoders).
    modality_summary: dict[str, dict] = {}
    for modality in MODALITIES:
        per_split = [m["modalities"][modality] for m in manifests]
        # Encoder identity must be consistent
        for key in ("encoder_name", "encoder_source", "feature_dim", "dtype"):
            values = {info[key] for info in per_split}
            if len(values) != 1:
                raise ValueError(
                    f"[{dataset}/{modality}] inconsistent {key} across splits: {values}"
                )
        modality_summary[modality] = {
            "encoder_name": per_split[0]["encoder_name"],
            "encoder_source": per_split[0]["encoder_source"],
            "feature_dim": per_split[0]["feature_dim"],
            "dtype": per_split[0]["dtype"],
            "max_length_per_split": {
                m["split"]: m["modalities"][modality]["max_length"] for m in manifests
            },
        }

    primary_stats: dict[str, float] | None = None
    for m in manifests:
        if m["split"] == "train":
            arr = m["primary_labels"].to(torch.float32)
            primary_stats = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std(unbiased=False)),
            }
            break

    payload = {
        "dataset": dataset,
        "task": task,
        "splits": splits_summary,
        "modalities": modality_summary,
        "primary_label_stats_train": primary_stats,
        "created_at": now_iso(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    log.info("[%s] wrote %s", dataset, out_path)
    return out_path


def merge_dataset(dataset: str, skip_existing: bool) -> None:
    log.info("=== %s ===", dataset)
    by_split = load_split_samples(dataset)
    manifests: list[dict] = []
    for split in VALID_SPLITS:
        samples = by_split.get(split, [])
        if not samples:
            log.info("[%s/%s] no samples — skip", dataset, split)
            continue
        manifest = merge_split(dataset, split, samples, skip_existing)
        if manifest is not None:
            manifests.append(manifest)
    if manifests:
        write_dataset_metadata(dataset, manifests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=(*VALID_DATASETS, "all"), default="all")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip manifests that already match the current modality caches (default).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rewrite manifests even when they already match.",
    )
    args = parser.parse_args()

    targets = VALID_DATASETS if args.dataset == "all" else (args.dataset,)
    failures: list[str] = []
    for ds in targets:
        try:
            merge_dataset(ds, args.skip_existing)
        except FileNotFoundError as e:
            log.error("[%s] %s", ds, e)
            failures.append(ds)
        except ValueError as e:
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
