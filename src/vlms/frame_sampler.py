"""Adaptive uniform frame sampling for VLM inputs (spec §18.1).

Spec: 4 frames for short clips, 8 frames for longer clips, uniform sampling.
We reuse :func:`src.encoders.visual.load_video_frames`'s PyAV-based sampler
under the hood and only change the frame count + return type.

Returned frames are H×W×3 uint8 numpy arrays (one per sampled timestep).
VLM image processors handle resizing internally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.encoders.visual import load_video_frames

log = logging.getLogger(__name__)

DEFAULT_SHORT_FRAMES = 4
DEFAULT_LONG_FRAMES = 8
DEFAULT_SHORT_THRESHOLD_SEC = 4.0


def _video_duration_seconds(path: str | Path) -> float | None:
    """Return container duration in seconds, or None on failure."""
    import av

    try:
        with av.open(str(path)) as container:
            if container.duration is None:
                return None
            return float(container.duration) / 1_000_000.0
    except Exception as e:  # noqa: BLE001 — corrupted clip is the typical failure
        log.warning("could not probe duration for %s: %s", path, e)
        return None


def sample_frames(
    path: str | Path,
    num_frames: int,
    image_size: int = 224,
) -> list[np.ndarray]:
    """Uniformly sample ``num_frames`` frames from ``path``.

    Returns a list of (H, W, 3) uint8 arrays. On decode failure returns
    ``num_frames`` blank frames at ``image_size`` so the calling pipeline
    doesn't crash on a single bad video.
    """
    arr = load_video_frames(path, num_frames=num_frames, image_size=image_size)
    return [arr[i] for i in range(arr.shape[0])]


def sample_frames_adaptive(
    path: str | Path,
    short_threshold_sec: float = DEFAULT_SHORT_THRESHOLD_SEC,
    num_frames_short: int = DEFAULT_SHORT_FRAMES,
    num_frames_long: int = DEFAULT_LONG_FRAMES,
    image_size: int = 224,
) -> list[np.ndarray]:
    """Spec §18.1 adaptive sampling: 4 frames if duration < threshold, else 8."""
    duration = _video_duration_seconds(path)
    n = num_frames_short if (duration is not None and duration < short_threshold_sec) else num_frames_long
    return sample_frames(path, num_frames=n, image_size=image_size)
