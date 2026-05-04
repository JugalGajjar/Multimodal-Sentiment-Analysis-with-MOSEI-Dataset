"""Standardized sample schema shared by all datasets in the X-MoFE suite.

Every prepare_<dataset>.py script normalizes its raw layout into a stream of
``Sample`` records, serialized as JSON Lines. Downstream feature extraction,
training, evaluation, and validation read this single schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


VALID_DATASETS = ("mosei", "meld", "ch_sims")
VALID_SPLITS = ("train", "val", "test")
VALID_TASKS = ("regression", "classification")


@dataclass
class Sample:
    """A single utterance/segment with text, audio, video pointers and labels.

    Required fields:
        sample_id: dataset-unique identifier
        dataset:   one of VALID_DATASETS
        split:     one of VALID_SPLITS
        transcript: utterance text (may be empty for non-speech segments)

    Media paths are repo-relative (under ``data/raw/<dataset>/``) when possible.
    Time bounds apply when audio_path / video_path point at a longer recording
    that needs to be sliced; for already-segmented clips, leave them as None.

    The ``labels`` dict carries dataset-specific structured labels beyond the
    primary scalar (e.g. CH-SIMS unimodal y_T/y_A/y_V, MELD sentiment+emotion,
    MOSEI emotion intensities). ``primary_label`` is the headline target used
    by the task loss.
    """

    sample_id: str
    dataset: str
    split: str
    transcript: str

    audio_path: str | None = None
    video_path: str | None = None

    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None

    primary_label: float | int | None = None
    task: str = "regression"
    labels: dict[str, Any] = field(default_factory=dict)

    speaker_id: str | None = None
    dialogue_id: str | None = None
    utterance_index: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"dataset={self.dataset!r} not in {VALID_DATASETS}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split={self.split!r} not in {VALID_SPLITS}")
        if self.task not in VALID_TASKS:
            raise ValueError(f"task={self.task!r} not in {VALID_TASKS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        return cls(**data)


def write_jsonl(path: str | Path, samples: Iterable[Sample]) -> int:
    """Write samples to ``path`` as JSON Lines. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[Sample]:
    """Yield ``Sample`` records from a JSON Lines file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Sample.from_dict(json.loads(line))
