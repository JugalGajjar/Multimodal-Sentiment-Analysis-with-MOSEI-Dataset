"""Read/write canonical train/val/test split files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from src.data.schemas import VALID_SPLITS


def write_splits(path: str | Path, splits: dict[str, Iterable[str]]) -> dict[str, int]:
    """Write a splits.json mapping split_name -> sorted list of sample_ids.

    Returns the per-split counts.
    """
    unknown = set(splits) - set(VALID_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}; expected {VALID_SPLITS}")

    payload: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for split in VALID_SPLITS:
        ids = sorted(set(splits.get(split, [])))
        payload[split] = ids
        counts[split] = len(ids)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return counts


def read_splits(path: str | Path) -> dict[str, list[str]]:
    """Read a splits.json file and return a {split: [sample_ids]} mapping."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {split: list(data.get(split, [])) for split in VALID_SPLITS}
