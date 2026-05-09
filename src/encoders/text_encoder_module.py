"""Trainable text encoder for end-to-end X-MoFE training.

Companion to :mod:`src.encoders.text` (which is a frozen-feature extractor
used in the Phase-2 caching pipeline). This module is a ``torch.nn.Module``
that lives *inside* the model graph at training time and supports three
fine-tuning regimes:

* ``trainable_layers="full"``    — all encoder parameters trainable.
* ``trainable_layers="last_n"``  — only the top ``n`` transformer layers + the
  final layer-norm / pooler trainable. Cheaper memory than full fine-tune
  while still recovering most of the accuracy gain (Houlsby et al., 2019).
* ``trainable_layers="none"``    — encoder fully frozen; identical to the
  cached-feature path but recomputed each step. Mainly for sanity-tests.

Forward signature mirrors what the rest of X-MoFE expects: takes a list of
raw transcripts, tokenises + encodes inside the module, returns
``(features (B, L_max, D), lengths (B,))`` with the SAME dtype/shape contract
as the cached features so downstream code is agnostic to which path produced
them.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

VALID_TRAINABLE = ("full", "last_n", "none")


class TextEncoderModule(nn.Module):
    """Trainable wrapper around an encoder-only HF model (e.g. ModernBERT).

    Args:
        model_name: HF model id. Default matches the frozen encoder used at
            cache-extraction time so the trained model can resume from the
            cached-features checkpoint cleanly if desired.
        max_length: token cap; longer transcripts are truncated.
        trainable_layers: ``"full"``, ``"last_n"``, or ``"none"``.
        last_n: when ``trainable_layers == "last_n"``, the number of top
            transformer layers to keep trainable. Final layer-norm / pooler
            (if present) are always also trainable in that mode.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 128,
        trainable_layers: str = "last_n",
        last_n: int = 4,
    ) -> None:
        super().__init__()
        if trainable_layers not in VALID_TRAINABLE:
            raise ValueError(
                f"trainable_layers must be one of {VALID_TRAINABLE}; got {trainable_layers!r}"
            )
        if trainable_layers == "last_n" and last_n <= 0:
            raise ValueError(f"last_n must be > 0; got {last_n}")

        self.model_name = model_name
        self.max_length = int(max_length)
        self.trainable_layers = trainable_layers
        self.last_n = int(last_n)

        log.info("loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        log.info("loading encoder: %s (trainable=%s, last_n=%d)",
                 model_name, trainable_layers, last_n)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = int(self.encoder.config.hidden_size)

        self._configure_trainability()

    # ------------------------------------------------------------------
    # Trainability control
    # ------------------------------------------------------------------

    def _configure_trainability(self) -> None:
        """Toggle ``requires_grad`` on encoder parameters per ``trainable_layers``."""
        if self.trainable_layers == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            return
        if self.trainable_layers == "none":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            return

        # last_n: freeze everything, then unfreeze the last n transformer layers
        # plus the final layer norm / pooler.
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Try to locate the encoder's transformer layer list. Common attrs:
        #   * ``self.encoder.encoder.layer``    (BERT, RoBERTa)
        #   * ``self.encoder.encoder.layers``   (some variants)
        #   * ``self.encoder.layers``           (decoder-style)
        layer_list = self._find_layer_list()
        if layer_list is None:
            log.warning(
                "could not locate transformer layers for %s; falling back to "
                "full fine-tune to be safe", self.model_name,
            )
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            return

        n = min(self.last_n, len(layer_list))
        for layer in list(layer_list)[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # Also unfreeze the final layer norm if there is one.
        for attr in ("final_layer_norm", "norm", "ln_f"):
            mod = getattr(self.encoder, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)

    def _find_layer_list(self) -> nn.ModuleList | None:
        """Best-effort lookup of the transformer-layer ``ModuleList``."""
        candidates = (
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("layers",),
            ("transformer", "layer"),
            ("transformer", "layers"),
        )
        for path in candidates:
            obj = self.encoder
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        return None

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        transcripts: Sequence[str],
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise + encode a batch of raw transcripts.

        Returns
        -------
        features : ``(B, L_max, D)`` — last hidden state, on the encoder's device.
        lengths  : ``(B,)`` long — unpadded token counts (== sum(attention_mask)).

        The returned tensors live on the encoder's parameter device by
        default. Pass ``device`` to move them somewhere else (rare —
        ordinarily the X-MoFE forward keeps everything on one device).
        """
        if device is None:
            device = next(self.encoder.parameters()).device

        inputs = self.tokenizer(
            list(transcripts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        features = outputs.last_hidden_state
        lengths = inputs["attention_mask"].sum(dim=1).long()
        return features, lengths
