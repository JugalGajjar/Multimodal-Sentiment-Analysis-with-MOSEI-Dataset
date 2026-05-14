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
        "transcripts":                       list[str]             length B
                                             (only when manifest carries the
                                             post-patch_manifests_with_text
                                             ``transcripts`` field — needed
                                             for end-to-end fine-tuning)
        "speaker_ids", "dialogue_ids",
        "utterance_indices":                 list[str|int|None]
                                             (Phase-3 dialogue context;
                                             present iff manifest carries them)
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

    def __init__(
        self,
        manifest_path: str | Path,
        context_window: int = 0,
        context_separator: str = " [SEP] ",
        context_speaker_format: str = "{speaker}: {text}",
    ) -> None:
        """Build the dataset.

        Args:
            manifest_path: Phase-2 ``<split>.pt`` manifest.
            context_window: when > 0 AND the manifest carries ``dialogue_ids`` +
                ``utterance_indices`` (post-``patch_manifests_with_text``), each
                sample's ``transcript`` is prefixed with the most-recent ``N``
                preceding utterances from the same dialogue (same split),
                joined by ``context_separator``. Required for MELD's
                dialogue-aware modeling (Lever-2). 0 (default) → no prefix.
            context_separator: token-encoder-friendly separator between
                contextual utterances. Use a string the tokenizer treats as
                a real boundary (e.g. ``" [SEP] "`` for ModernBERT / BERT).
            context_speaker_format: format string for each contextual line.
                ``{speaker}`` and ``{text}`` are substituted. When a sample
                lacks a speaker id, the format falls back to ``"{text}"``.
        """
        manifest_path = Path(manifest_path)
        self.manifest_path = manifest_path
        self.dataset_dir = manifest_path.parent
        self.context_window = int(context_window)
        self.context_separator = context_separator
        self.context_speaker_format = context_speaker_format

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

        # Optional fields added by ``scripts/data/patch_manifests_with_text.py``.
        # Present when the manifest has been patched for fine-tuning / Phase 3
        # dialogue modelling; absent on un-patched legacy manifests, in which
        # case downstream code falls back to the cached-feature path.
        self.transcripts: list[str] | None = self.manifest.get("transcripts")
        self.speaker_ids: list | None = self.manifest.get("speaker_ids")
        self.dialogue_ids: list | None = self.manifest.get("dialogue_ids")
        self.utterance_indices: list | None = self.manifest.get("utterance_indices")
        self.has_transcripts = (
            self.transcripts is not None and len(self.transcripts) == len(self.sample_ids)
        )

        # Decide once whether this dataset carries unimodal annotations
        # (CH-SIMS does; MELD/MOSEI do not).
        first_label = self.rich_labels[0] if self.rich_labels else {}
        self.has_unimodal_labels = all(k in first_label for k in UNIMODAL_LABEL_KEYS)

        # Dialogue-context lookup. When the manifest carries dialogue_ids +
        # utterance_indices (post-patch), pre-build a sorted index per dialogue
        # so __getitem__ can fetch the preceding N utterances cheaply. Only
        # built when ``context_window > 0`` to avoid the overhead for the
        # frozen-feature path or non-dialogue datasets.
        self._dialogue_index: dict[str, list[tuple[int, int]]] = {}
        if (
            self.context_window > 0
            and self.has_transcripts
            and self.dialogue_ids is not None
            and self.utterance_indices is not None
        ):
            for i in range(len(self.sample_ids)):
                did = self.dialogue_ids[i]
                uidx = self.utterance_indices[i]
                if did is None or uidx is None:
                    continue
                self._dialogue_index.setdefault(str(did), []).append((int(uidx), i))
            for did in self._dialogue_index:
                self._dialogue_index[did].sort()

    def _build_context_prefix(self, sample_idx: int) -> str:
        """Return the dialogue-context string for the sample at ``sample_idx``.

        Returns an empty string when ``context_window == 0``, no dialogue index
        is available, or the sample has no prior utterances in its dialogue.
        """
        if self.context_window <= 0 or not self._dialogue_index:
            return ""
        did = self.dialogue_ids[sample_idx]
        cur_uidx = self.utterance_indices[sample_idx]
        if did is None or cur_uidx is None:
            return ""
        ordered = self._dialogue_index.get(str(did), ())
        # All utterances strictly before this one in the same dialogue.
        prior = [(u, idx) for (u, idx) in ordered if u < int(cur_uidx)]
        if not prior:
            return ""
        window = prior[-self.context_window:]
        lines: list[str] = []
        for _, j in window:
            text = self.transcripts[j] if self.transcripts else ""
            speaker = (self.speaker_ids[j] if self.speaker_ids else None)
            if speaker:
                lines.append(self.context_speaker_format.format(speaker=speaker, text=text))
            else:
                lines.append(text)
        # Append a trailing separator so the current utterance reads as the
        # last turn of the dialogue when the tokenizer concatenates.
        return self.context_separator.join(lines) + self.context_separator

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
        if self.has_transcripts:
            raw_transcript = self.transcripts[idx]
            speaker = self.speaker_ids[idx] if self.speaker_ids is not None else None
            if self.context_window > 0 and self._dialogue_index:
                prefix = self._build_context_prefix(idx)
                if speaker and prefix:
                    # Tag the current utterance with the speaker so the model
                    # can attribute the response to the right participant.
                    current = self.context_speaker_format.format(
                        speaker=speaker, text=raw_transcript,
                    )
                    item["transcript"] = prefix + current
                else:
                    item["transcript"] = prefix + raw_transcript
            else:
                item["transcript"] = raw_transcript

            if self.speaker_ids is not None:
                item["speaker_id"] = speaker
            if self.dialogue_ids is not None:
                item["dialogue_id"] = self.dialogue_ids[idx]
            if self.utterance_indices is not None:
                item["utterance_index"] = self.utterance_indices[idx]
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

    # Optional fields populated only when the manifest carries them
    # (post-``patch_manifests_with_text``). Lists rather than stacked tensors
    # because they're heterogeneous text/identifiers, not features.
    if "transcript" in batch[0]:
        out["transcripts"] = [item["transcript"] for item in batch]
    if "speaker_id" in batch[0]:
        out["speaker_ids"] = [item["speaker_id"] for item in batch]
    if "dialogue_id" in batch[0]:
        out["dialogue_ids"] = [item["dialogue_id"] for item in batch]
    if "utterance_index" in batch[0]:
        out["utterance_indices"] = [item["utterance_index"] for item in batch]

    return out


def make_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dataset: XMoFEDataset | None = None,
    context_window: int = 0,
) -> DataLoader:
    """Build a ``DataLoader`` for a ``<split>.pt`` manifest.

    Pass an existing ``dataset`` to share cached tensors across train/val
    loaders if needed; otherwise the constructor loads them fresh.

    ``context_window > 0`` prepends each sample's transcript with the most
    recent N preceding utterances from the same dialogue (Lever-2 MELD
    dialogue context). Quietly a no-op when the manifest lacks the
    dialogue/utterance index metadata.
    """
    if dataset is None:
        dataset = XMoFEDataset(manifest_path, context_window=context_window)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_xmofe,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
