"""Feature cache I/O for X-MoFE.

Each (modality, dataset, split) is stored as a single ``.pt`` file with a
fixed schema so downstream code can load uniformly without per-encoder
branching. The saved object is::

    {
        "sample_ids": list[str],          # length N, parallel to features
        "features": Tensor,               # (N, L_max, D), dtype per metadata
        "lengths":  Tensor,               # (N,) int32, unpadded length per sample
        "metadata": {
            "modality":         "text" | "audio" | "visual",
            "dataset":          "mosei" | "meld" | "ch_sims",
            "split":            "train" | "val" | "test",
            "encoder_name":     str,
            "encoder_source":   str,      # HF model id, "covarep_csd", etc.
            "feature_dim":      int,
            "max_length":       int,
            "dtype":            str,      # "float16" | "float32"
            "num_samples":      int,
            "created_at":       str,      # ISO-8601
        }
    }

Variable-length sequences are stored padded to the per-split L_max; the
``lengths`` tensor lets callers reconstruct the unpadded form.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import torch


REQUIRED_METADATA_KEYS = {
    "modality",
    "dataset",
    "split",
    "encoder_name",
    "encoder_source",
    "feature_dim",
    "max_length",
    "dtype",
    "num_samples",
    "created_at",
}


def _validate_metadata(meta: dict[str, Any]) -> None:
    missing = REQUIRED_METADATA_KEYS - set(meta)
    if missing:
        raise ValueError(f"feature cache metadata missing keys: {sorted(missing)}")


def write_feature_cache(
    path: str | Path,
    sample_ids: list[str],
    features: torch.Tensor,
    lengths: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    """Persist a feature cache to disk.

    The caller is responsible for arranging ``features``, ``lengths``, and
    ``sample_ids`` in matching order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if features.ndim != 3:
        raise ValueError(f"features must be (N, L, D); got shape {tuple(features.shape)}")
    n, l_max, d = features.shape
    if len(sample_ids) != n:
        raise ValueError(f"sample_ids ({len(sample_ids)}) and features rows ({n}) disagree")
    if lengths.shape != (n,):
        raise ValueError(f"lengths must be ({n},); got {tuple(lengths.shape)}")

    meta = dict(metadata)
    meta.setdefault("num_samples", n)
    meta.setdefault("max_length", l_max)
    meta.setdefault("feature_dim", d)
    meta.setdefault("created_at", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    meta.setdefault("dtype", str(features.dtype).removeprefix("torch."))
    _validate_metadata(meta)

    payload = {
        "sample_ids": list(sample_ids),
        "features": features.contiguous(),
        "lengths": lengths.to(dtype=torch.int32).contiguous(),
        "metadata": meta,
    }
    torch.save(payload, path)
    return path


def read_feature_cache(path: str | Path) -> dict[str, Any]:
    """Load a feature cache. Returns the same dict structure that was saved."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_metadata(payload["metadata"])
    return payload


def cache_path(
    processed_root: str | Path,
    dataset: str,
    modality: str,
    split: str,
) -> Path:
    """Canonical location of a per-modality cache.

    ``data/processed/<dataset>/<modality>_features/<split>.pt``
    """
    return Path(processed_root) / dataset / f"{modality}_features" / f"{split}.pt"
