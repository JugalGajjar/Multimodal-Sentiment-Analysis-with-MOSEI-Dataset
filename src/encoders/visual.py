"""Visual extraction backends for X-MoFE.

Two implementations share the same encode signature so the extraction script
can dispatch by dataset:

* :class:`VideoMAEEncoder` — frozen ``MCG-NJU/videomae-base`` over a uniformly
  sampled 16-frame clip; returns ``(B, 1568, 768)`` patches per sample
  (8 temporal × 14 × 14 spatial). VideoMAEv2-base is in the spec but its
  HF checkpoint requires ``trust_remote_code=True``; V1 is the canonical
  baseline in multimodal-sentiment papers and loads via standard transformers.
* :class:`OpenFace2SequenceReader` — passthrough that pulls per-video
  OpenFace2 sequences from CMU-MOSEI's CSD (713-dim, ~30 Hz).

Frame loading from mp4 uses PyAV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.encoders.text import resolve_device

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame loading from mp4
# ---------------------------------------------------------------------------


def load_video_frames(
    path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """Decode and uniformly sample ``num_frames`` from an mp4 as an RGB
    numpy array of shape ``(num_frames, H, W, 3)`` uint8.

    On any decode failure (corrupted file, missing video stream, etc.) returns
    a blank black-frame stack at the requested size so downstream batching
    never breaks. The ``image_size`` argument is *only* used for the failure
    case — successful decodes return frames at native resolution and the
    image processor handles resizing.
    """
    import av

    blank = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

    try:
        container = av.open(str(path))
    except Exception as e:  # noqa: BLE001
        log.warning("av.open failed for %s: %s -- returning blank frames", path, e)
        return blank

    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        container.close()
        log.warning("no video stream in %s -- returning blank frames", path)
        return blank

    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(video_stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as e:  # noqa: BLE001
        log.warning("decode failed for %s: %s -- returning blank frames", path, e)
        container.close()
        return blank
    finally:
        container.close()

    if not frames:
        log.warning("no frames decoded from %s -- returning blank frames", path)
        return blank

    if len(frames) >= num_frames:
        # Uniform sampling across the full duration
        indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        sampled = [frames[i] for i in indices]
    else:
        # Repeat the last frame to pad to the required count
        sampled = list(frames) + [frames[-1]] * (num_frames - len(frames))

    return np.stack(sampled)


# ---------------------------------------------------------------------------
# VideoMAE encoder
# ---------------------------------------------------------------------------


class VideoMAEEncoder:
    """Frozen VideoMAE-base wrapper that emits last-hidden-state patch sequences.

    Args:
        model_name: HF model id, default ``MCG-NJU/videomae-base``.
        num_frames: frames per clip, default 16 (model's pretraining setting).
        image_size: spatial size for the image processor, default 224.
        device: ``"mps"`` / ``"cuda"`` / ``"cpu"`` / ``"auto"``.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        num_frames: int = 16,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.model_name = model_name
        self.num_frames = num_frames
        self.image_size = image_size
        self.device = resolve_device(device)

        log.info("loading videomae image processor: %s", model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        log.info("loading videomae model: %s -> %s", model_name, self.device)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_dim = int(self.model.config.hidden_size)
        # 1568 for default config: (16 / 2) * (224 / 16)^2 = 8 * 14 * 14
        tubelet = int(self.model.config.tubelet_size)
        patch = int(self.model.config.patch_size)
        self.num_patches = (num_frames // tubelet) * (image_size // patch) ** 2

    @torch.no_grad()
    def encode(
        self,
        video_arrays: Sequence[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of (num_frames, H, W, 3) uint8 video arrays.

        Returns:
            features: ``(B, num_patches, feature_dim)`` float tensor on CPU.
            lengths: ``(B,)`` int tensor — always ``num_patches`` since every
                clip is padded/sampled to ``num_frames`` upstream.
        """
        # The processor accepts a list of "videos", each a list/array of frames.
        videos = [list(v) for v in video_arrays]
        inputs = self.processor(videos, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.detach().to("cpu")

        b = features.shape[0]
        lengths = torch.full((b,), self.num_patches, dtype=torch.int32)
        return features, lengths


# ---------------------------------------------------------------------------
# OpenFace2 CSD passthrough (CMU-MOSEI)
# ---------------------------------------------------------------------------


class OpenFace2SequenceReader:
    """Read per-video OpenFace2 sequences from CMU-MOSEI's CSD.

    For each ``video_id``, returns the entire 713-dim OpenFace2 sequence
    (facial landmarks + action units + gaze + head pose + HoG) truncated
    to ``max_frames`` and zero-padded so the cache stacks cleanly. NaN/inf
    values are replaced with zeros — common in OpenFace2 features when face
    detection drops out mid-clip.
    """

    def __init__(
        self,
        csd_path: str | Path,
        max_frames: int = 3600,
        sampling_rate: int = 30,
        feature_dim: int | None = None,
    ) -> None:
        from mmsdk import mmdatasdk as md

        self.csd_path = str(csd_path)
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate
        self._explicit_dim = feature_dim

        log.info("loading OpenFace2 CSD: %s", self.csd_path)
        self._dataset = md.mmdataset({"visual": self.csd_path})

        # Probe the first key to discover the feature dim if not declared.
        first_key = next(iter(self._dataset["visual"].keys()))
        sample_feats = np.asarray(
            self._dataset["visual"][first_key]["features"], dtype=np.float32
        )
        probed_dim = sample_feats.shape[1] if sample_feats.ndim == 2 else 0
        self.feature_dim = feature_dim or probed_dim
        if feature_dim and feature_dim != probed_dim:
            log.warning(
                "configured feature_dim=%d but CSD has %d; using configured value",
                feature_dim, probed_dim,
            )

    def encode(
        self,
        video_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice per-video OpenFace2 sequences keyed by ``video_id``.

        Returns:
            features: ``(B, max_frames, feature_dim)`` float tensor.
            lengths: ``(B,)`` int tensor of unpadded frame counts.
        """
        chunks: list[np.ndarray] = []
        lengths: list[int] = []
        for vid in video_ids:
            try:
                feats = np.asarray(self._dataset["visual"][vid]["features"], dtype=np.float32)
            except KeyError:
                log.warning("video_id %s missing from OpenFace2 CSD", vid)
                feats = np.zeros((0, self.feature_dim), dtype=np.float32)

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            if feats.ndim == 2 and feats.shape[1] != self.feature_dim:
                # Pad/truncate width dimension to the configured feature_dim
                fixed = np.zeros((feats.shape[0], self.feature_dim), dtype=np.float32)
                d = min(feats.shape[1], self.feature_dim)
                if d > 0:
                    fixed[:, :d] = feats[:, :d]
                feats = fixed

            t = min(feats.shape[0], self.max_frames) if feats.ndim == 2 else 0
            padded = np.zeros((self.max_frames, self.feature_dim), dtype=np.float32)
            if t > 0:
                padded[:t] = feats[:t]
            chunks.append(padded)
            lengths.append(t)

        features = torch.from_numpy(np.stack(chunks))
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
        return features, lengths_tensor
