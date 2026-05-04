"""ModernBERT text encoder wrapper.

Loads a frozen ``answerdotai/ModernBERT-base`` (or any compatible HF
encoder-only model) and exposes a batched ``encode`` method that returns the
final hidden-state sequence plus the corresponding attention mask. The
attention mask is the source of truth for per-sample sequence lengths so
downstream attention pooling ignores padded positions.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick mps > cuda > cpu unless caller forces a specific device."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModernBERTEncoder:
    """Frozen ModernBERT wrapper that emits last-hidden-state sequences.

    Args:
        model_name: HF model id (default: ``answerdotai/ModernBERT-base``).
        max_length: token cap; longer transcripts are truncated.
        padding: ``"max_length"`` produces a uniform L_max across batches so
            cached tensors stack cleanly into a single (N, L_max, D) blob.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 128,
        padding: str = "max_length",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.device = resolve_device(device)

        log.info("loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        log.info("loading model: %s -> %s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.feature_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self,
        transcripts: Sequence[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of transcripts.

        ``max_length`` overrides the default set at construction time so the
        same encoder instance can serve datasets with different caps (MOSEI
        wants 512 for full-video transcripts; MELD/CH-SIMS want 128).

        Returns:
            features: ``(B, L_max, feature_dim)`` float tensor on CPU.
            lengths: ``(B,)`` int tensor of unpadded token counts.
        """
        cap = max_length if max_length is not None else self.max_length
        inputs = self.tokenizer(
            list(transcripts),
            padding=self.padding,
            truncation=True,
            max_length=cap,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.detach().to("cpu")
        lengths = inputs["attention_mask"].sum(dim=1).to(dtype=torch.int32, device="cpu")
        return features, lengths
