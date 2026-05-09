"""Missing-modality evaluation conditions (spec §16.1).

Sets the corresponding modality's per-sample length to zero; the model's
``AttentionPool.missing_token`` then takes over for those rows. The
reliability estimator continues to receive the missing-token-derived
pooled vector, so reliability scores naturally adapt to absent modalities
— this is the H2 behaviour we evaluate against in the report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

# Spec §16.1 evaluation matrix: clean baseline + 3 single-missing + 3 pair-missing.
MISSING_CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("clean", ()),
    ("text_missing", ("text",)),
    ("audio_missing", ("audio",)),
    ("visual_missing", ("visual",)),
    ("text_audio_missing", ("text", "audio")),
    ("text_visual_missing", ("text", "visual")),
    ("audio_visual_missing", ("audio", "visual")),
)

VALID_MODALITIES = ("text", "audio", "visual")


def apply_missing(
    batch: Mapping[str, Any],
    modalities_to_drop: Sequence[str],
) -> dict[str, Any]:
    """Return a shallow-copied batch with the listed modalities marked missing.

    Marking == zeroing the per-sample length tensor for that modality.
    """
    out: dict[str, Any] = dict(batch)
    for m in modalities_to_drop:
        if m not in VALID_MODALITIES:
            raise ValueError(f"unknown modality {m!r}; expected one of {VALID_MODALITIES}")
        length_key = f"{m}_length"
        if length_key not in batch:
            raise KeyError(f"batch missing {length_key!r}")
        out[length_key] = torch.zeros_like(batch[length_key])
        # If text is being marked missing AND the batch carries raw
        # transcripts (Phase-1 fine-tuning path), we must also empty the
        # transcripts. Otherwise an in-graph encoder would still encode the
        # original text and re-introduce it through the encoder pathway,
        # silently defeating the missing-modality condition.
        if m == "text" and "transcripts" in batch:
            out["transcripts"] = ["" for _ in batch["transcripts"]]
    return out
