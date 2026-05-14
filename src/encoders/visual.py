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
    to ``max_frames`` and zero-padded so the cache stacks cleanly.

    OpenFace2's HoG dimensions can carry extreme outliers (we observed
    magnitudes up to 5×10⁷ in the CMU-MOSEI distribution) that overflow
    fp16 to ±inf when cached. This reader therefore:

      1. Replaces NaN/inf with zeros.
      2. Clips raw values to a configurable outlier band before stats.
      3. Standardizes per-dimension by streaming-pass mean/std stats
         computed once over the entire CSD on init.
      4. Clips standardized values to ±10σ to drop residual outliers.

    Values cached after standardization are well within fp16 range and
    centered on zero, which the downstream ModalityProjection's input
    LayerNorm leaves close to its no-op fixed point.
    """

    def __init__(
        self,
        csd_path: str | Path,
        max_frames: int = 3600,
        sampling_rate: int = 30,
        feature_dim: int | None = None,
        normalize: bool = True,
        outlier_clip: float = 1.0e4,
        standardized_clip: float = 10.0,
    ) -> None:
        from mmsdk import mmdatasdk as md

        self.csd_path = str(csd_path)
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.outlier_clip = float(outlier_clip)
        self.standardized_clip = float(standardized_clip)

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

        if self.normalize:
            self.mean, self.std = self._compute_stats()
        else:
            self.mean = np.zeros(self.feature_dim, dtype=np.float32)
            self.std = np.ones(self.feature_dim, dtype=np.float32)

    def _compute_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature mean and std, computed once over the entire CSD.

        Uses outlier clipping before accumulation so a handful of HoG
        explosions don't dominate the normalization parameters.
        """
        log.info("computing OpenFace2 normalization stats over CSD...")
        total = np.zeros(self.feature_dim, dtype=np.float64)
        sq_total = np.zeros(self.feature_dim, dtype=np.float64)
        n_frames = 0
        for vid in self._dataset["visual"].keys():
            feats = np.asarray(self._dataset["visual"][vid]["features"], dtype=np.float64)
            if feats.ndim != 2 or feats.shape[1] != self.feature_dim:
                continue
            feats = np.nan_to_num(
                feats, nan=0.0,
                posinf=self.outlier_clip, neginf=-self.outlier_clip,
            )
            feats = np.clip(feats, -self.outlier_clip, self.outlier_clip)
            n_frames += feats.shape[0]
            total += feats.sum(axis=0)
            sq_total += (feats ** 2).sum(axis=0)

        if n_frames == 0:
            log.warning("no frames found while computing OpenFace2 stats")
            return (
                np.zeros(self.feature_dim, dtype=np.float32),
                np.ones(self.feature_dim, dtype=np.float32),
            )

        mean = total / n_frames
        var = (sq_total / n_frames) - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        # Avoid divide-by-zero for constant dimensions.
        std = np.where(std < 1e-6, 1.0, std)
        log.info(
            "  stats over %d frames: |mean|≈%.3f mean(std)≈%.3f",
            n_frames, float(np.abs(mean).mean()), float(std.mean()),
        )
        return mean.astype(np.float32), std.astype(np.float32)

    def encode(
        self,
        video_ids: Sequence[str],
        intervals: Sequence[tuple[float, float] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice per-video OpenFace2 sequences keyed by ``video_id``.

        Args:
            video_ids: per-sample video identifiers.
            intervals: optional per-sample ``(start_seconds, end_seconds)`` for
                utterance-level slicing. ``None`` returns the full video.
                Used for MOSEI utterance-level (~22,856 samples).

        Returns:
            features: ``(B, max_frames, feature_dim)`` float tensor.
            lengths: ``(B,)`` int tensor of unpadded frame counts.
        """
        chunks: list[np.ndarray] = []
        lengths: list[int] = []
        for i, vid in enumerate(video_ids):
            try:
                feats = np.asarray(self._dataset["visual"][vid]["features"], dtype=np.float32)
            except KeyError:
                log.warning("video_id %s missing from OpenFace2 CSD", vid)
                feats = np.zeros((0, self.feature_dim), dtype=np.float32)

            feats = np.nan_to_num(
                feats, nan=0.0,
                posinf=self.outlier_clip, neginf=-self.outlier_clip,
            )
            feats = np.clip(feats, -self.outlier_clip, self.outlier_clip)

            if feats.ndim == 2 and feats.shape[1] != self.feature_dim:
                fixed = np.zeros((feats.shape[0], self.feature_dim), dtype=np.float32)
                d = min(feats.shape[1], self.feature_dim)
                if d > 0:
                    fixed[:, :d] = feats[:, :d]
                feats = fixed

            # Utterance-level slicing in seconds → frames at ``sampling_rate``.
            iv = intervals[i] if intervals is not None else None
            if iv is not None and feats.ndim == 2:
                start_s, end_s = float(iv[0]), float(iv[1])
                start_f = max(0, int(round(start_s * self.sampling_rate)))
                end_f = min(feats.shape[0], int(round(end_s * self.sampling_rate)))
                if end_f > start_f:
                    feats = feats[start_f:end_f]
                else:
                    feats = feats[:0]

            if self.normalize and feats.shape[0] > 0:
                feats = (feats - self.mean) / self.std
                feats = np.clip(feats, -self.standardized_clip, self.standardized_clip)

            t = min(feats.shape[0], self.max_frames) if feats.ndim == 2 else 0
            padded = np.zeros((self.max_frames, self.feature_dim), dtype=np.float32)
            if t > 0:
                padded[:t] = feats[:t]
            chunks.append(padded)
            lengths.append(t)

        features = torch.from_numpy(np.stack(chunks))
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
        return features, lengths_tensor
