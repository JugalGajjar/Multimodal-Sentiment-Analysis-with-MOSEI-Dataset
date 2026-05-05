"""Audio extraction backends for X-MoFE.

Two implementations share the same encode signature so the extraction script
can dispatch by dataset:

* :class:`WavLMEncoder` — runs frozen ``microsoft/wavlm-base-plus`` on raw
  16 kHz mono waveforms and returns ``(B, L, 768)`` features at ~50 Hz.
* :class:`COVAREPSequenceReader` — passthrough for CMU-MOSEI: opens
  ``CMU_MOSEI_COVAREP.csd`` once and slices the per-video 74-dim sequence at
  100 Hz, truncating to ``max_frames``.

Audio loading from mp4 uses PyAV (bundles its own ffmpeg libs, so no system
``ffmpeg`` install is required).
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
# Audio loading from mp4
# ---------------------------------------------------------------------------


def load_audio_from_video(
    path: str | Path,
    target_sr: int = 16000,
    silence_seconds_on_failure: float = 0.1,
) -> np.ndarray:
    """Decode the audio track of an mp4/mov/wav/flac file as mono float32 at ``target_sr``.

    On any failure (corrupted moov atom, missing audio stream, decode error)
    returns a short silence buffer instead of an empty array. WavLM's CNN
    front-end produces a *negative* output length for zero-length input, so
    a small silence pad keeps downstream lengths well-defined while still
    signalling "no useful audio" via near-zero feature energy.
    """
    import av

    silence = np.zeros(int(target_sr * silence_seconds_on_failure), dtype=np.float32)

    try:
        container = av.open(str(path))
    except Exception as e:  # noqa: BLE001
        log.warning("av.open failed for %s: %s -- returning silence", path, e)
        return silence

    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        log.warning("no audio stream in %s -- returning silence", path)
        return silence

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().flatten())
        # Flush the resampler buffer
        for resampled in resampler.resample(None) or []:
            chunks.append(resampled.to_ndarray().flatten())
    except Exception as e:  # noqa: BLE001
        log.warning("decode failed for %s: %s -- returning silence", path, e)
        container.close()
        return silence
    finally:
        container.close()

    if not chunks:
        log.warning("no audio frames decoded from %s -- returning silence", path)
        return silence
    return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# WavLM encoder
# ---------------------------------------------------------------------------


class WavLMEncoder:
    """Frozen WavLM Base+ wrapper that emits last-hidden-state sequences.

    Args:
        model_name: HF model id, default ``microsoft/wavlm-base-plus``.
        sampling_rate: WavLM expects 16 kHz waveforms.
        max_duration_seconds: longer waveforms are truncated, shorter padded.
        device: ``"mps"`` / ``"cuda"`` / ``"cpu"`` / ``"auto"``.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        sampling_rate: int = 16000,
        max_duration_seconds: float = 8.0,
        device: str | None = None,
    ) -> None:
        from transformers import AutoFeatureExtractor, WavLMModel

        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.max_duration_seconds = max_duration_seconds
        self.max_samples = int(round(sampling_rate * max_duration_seconds))
        self.device = resolve_device(device)

        log.info("loading wavlm feature extractor: %s", model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        log.info("loading wavlm model: %s -> %s", model_name, self.device)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self,
        waveforms: Sequence[np.ndarray],
        max_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of mono float32 waveforms.

        Returns:
            features: ``(B, T_out_max, feature_dim)`` float tensor on CPU.
            lengths: ``(B,)`` int tensor of unpadded output frame counts.
        """
        cap = max_samples if max_samples is not None else self.max_samples
        # Coerce inputs: feature_extractor expects 1-D float arrays; clip to cap.
        prepared = [np.asarray(w, dtype=np.float32)[:cap] for w in waveforms]
        # Pad short clips with zeros so the feature extractor builds an
        # attention_mask reflecting the true content length per sample.
        inputs = self.feature_extractor(
            prepared,
            sampling_rate=self.sampling_rate,
            padding="max_length",
            max_length=cap,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.detach().to("cpu")

        # WavLM's CNN front-end downsamples the input; recover per-sample
        # output frame counts from the input attention mask. The convolution
        # arithmetic can yield a *negative* output length for very-short or
        # zero-length inputs (corrupted clips), so clamp at zero.
        input_lengths = inputs["attention_mask"].sum(dim=-1)
        output_lengths = self.model._get_feat_extract_output_lengths(input_lengths)
        output_lengths = output_lengths.clamp(min=0).to(dtype=torch.int32, device="cpu")
        return features, output_lengths


# ---------------------------------------------------------------------------
# COVAREP CSD passthrough (CMU-MOSEI)
# ---------------------------------------------------------------------------


class COVAREPSequenceReader:
    """Read per-video COVAREP sequences from CMU-MOSEI's CSD.

    For each ``video_id``, returns the entire 74-dim COVAREP sequence
    truncated to ``max_frames`` and padded to that length so the cache stacks
    cleanly. NaN/inf values (common in COVAREP's voicing-related features
    during silence) are replaced with zeros before caching.
    """

    feature_dim = 74

    def __init__(
        self,
        csd_path: str | Path,
        max_frames: int = 6000,
        sampling_rate: int = 100,
    ) -> None:
        from mmsdk import mmdatasdk as md

        self.csd_path = str(csd_path)
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate

        log.info("loading COVAREP CSD: %s", self.csd_path)
        self._dataset = md.mmdataset({"audio": self.csd_path})

    def encode(
        self,
        video_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice per-video COVAREP sequences keyed by ``video_id``.

        Returns:
            features: ``(B, max_frames, 74)`` float tensor.
            lengths: ``(B,)`` int tensor of unpadded frame counts.
        """
        chunks: list[np.ndarray] = []
        lengths: list[int] = []
        for vid in video_ids:
            try:
                feats = np.asarray(self._dataset["audio"][vid]["features"], dtype=np.float32)
            except KeyError:
                log.warning("video_id %s missing from COVAREP CSD", vid)
                feats = np.zeros((0, self.feature_dim), dtype=np.float32)

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            if feats.shape[1:] != (self.feature_dim,):
                # Some MOSEI distributions ship slightly different COVAREP
                # variants; trim or pad to the canonical 74-dim shape.
                fixed = np.zeros((feats.shape[0], self.feature_dim), dtype=np.float32)
                d = min(feats.shape[1] if feats.ndim == 2 else 0, self.feature_dim)
                if d > 0:
                    fixed[:, :d] = feats[:, :d]
                feats = fixed

            t = min(feats.shape[0], self.max_frames)
            padded = np.zeros((self.max_frames, self.feature_dim), dtype=np.float32)
            if t > 0:
                padded[:t] = feats[:t]
            chunks.append(padded)
            lengths.append(t)

        features = torch.from_numpy(np.stack(chunks))
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
        return features, lengths_tensor
