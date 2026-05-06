"""``Dataset`` and ``DataLoader`` factory for X-MoFE training.

Reads the per-split manifests produced by
:mod:`scripts.features.merge_cached_features` plus the per-modality caches
they reference, and yields batches in the shape the model + composite loss
expect:

    {
        "text", "audio", "visual":           Tensor (B, L_m, D_m)  fp32
        "text_length", "audio_length",
        "visual_length":                     Tensor (B,)           long
        "label":                             Tensor (B,)           float|long
        "unimodal_labels":                   Tensor (B, 3)         float
                                             (CH-SIMS only — absent otherwise)
        "sample_ids":                        list[str]             length B
        "rich_labels":                       list[dict]            length B
    }

Caches are stored in fp16; the collate cast lifts batches to fp32 because
that's what the model and the auxiliary losses (faithfulness, stability)
expect. Unimodal labels are pulled from each sample's rich-label dict when
all three of ``sentiment_T``, ``sentiment_A``, ``sentiment_V`` are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.features import read_feature_cache

MODALITIES = ("text", "audio", "visual")
UNIMODAL_LABEL_KEYS = ("sentiment_T", "sentiment_A", "sentiment_V")


class XMoFEDataset(Dataset):
    """Eager-load dataset over a manifest + per-modality caches.

    Memory cost equals the sum of the three modality caches for the split.
    For MELD train that's ≈30 GB in fp16 (text 2 GB + audio 6 GB + visual 22 GB).
    Fits in 48 GB unified memory; switch to a memory-mapped storage format
    only if/when this becomes a bottleneck.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        manifest_path = Path(manifest_path)
        self.manifest_path = manifest_path
        self.dataset_dir = manifest_path.parent

        self.manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        self.caches: dict[str, dict[str, Any]] = {}
        for modality, info in self.manifest["modalities"].items():
            cache = read_feature_cache(self.dataset_dir / info["cache_path"])
            if cache["sample_ids"] != self.manifest["sample_ids"]:
                raise ValueError(
                    f"{modality} cache sample_ids do not align with the "
                    f"manifest at {manifest_path}; rerun merge_cached_features."
                )
            self.caches[modality] = cache

        self.dataset = self.manifest["dataset"]
        self.split = self.manifest["split"]
        self.task = self.manifest["task"]
        self.sample_ids: list[str] = list(self.manifest["sample_ids"])
        self.primary_labels: torch.Tensor = self.manifest["primary_labels"]
        self.rich_labels: list[dict] = list(self.manifest["labels"])

        # Decide once whether this dataset carries unimodal annotations
        # (CH-SIMS does; MELD/MOSEI do not).
        first_label = self.rich_labels[0] if self.rich_labels else {}
        self.has_unimodal_labels = all(k in first_label for k in UNIMODAL_LABEL_KEYS)

    @property
    def feature_dims(self) -> dict[str, int]:
        return {m: int(self.manifest["modalities"][m]["feature_dim"]) for m in MODALITIES}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "sample_id": self.sample_ids[idx],
            "label": self.primary_labels[idx],
            "rich_labels": self.rich_labels[idx],
        }
        for modality in MODALITIES:
            cache = self.caches[modality]
            item[modality] = cache["features"][idx]
            item[f"{modality}_length"] = cache["lengths"][idx]
        if self.has_unimodal_labels:
            rich = self.rich_labels[idx]
            item["unimodal_labels"] = torch.tensor(
                [float(rich[k]) for k in UNIMODAL_LABEL_KEYS],
                dtype=torch.float32,
            )
        return item


def collate_xmofe(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of XMoFEDataset samples into a model-ready batch."""
    out: dict[str, Any] = {
        "sample_ids": [item["sample_id"] for item in batch],
        "rich_labels": [item["rich_labels"] for item in batch],
    }

    for modality in MODALITIES:
        feats = torch.stack([item[modality] for item in batch]).to(torch.float32)
        # Belt-and-suspenders: any NaN/inf that slipped through the cache
        # (e.g. fp16 overflow on some legacy file) gets zeroed here so
        # training doesn't poison from one bad sample.
        out[modality] = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        out[f"{modality}_length"] = torch.stack([item[f"{modality}_length"] for item in batch]).long()

    label_first = batch[0]["label"]
    out["label"] = torch.stack([item["label"] for item in batch]) if isinstance(label_first, torch.Tensor) else torch.tensor([item["label"] for item in batch])

    if "unimodal_labels" in batch[0]:
        out["unimodal_labels"] = torch.stack([item["unimodal_labels"] for item in batch])

    return out


def make_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dataset: XMoFEDataset | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` for a ``<split>.pt`` manifest.

    Pass an existing ``dataset`` to share cached tensors across train/val
    loaders if needed; otherwise the constructor loads them fresh.
    """
    if dataset is None:
        dataset = XMoFEDataset(manifest_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_xmofe,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
