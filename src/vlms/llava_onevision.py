"""LLaVA-OneVision inference wrapper.

Mirrors :class:`Qwen25VL` so both VLMs expose the same ``generate(frames,
prompt)`` interface — the run script doesn't branch on model identity past
construction.

Default checkpoint: ``llava-hf/llava-onevision-qwen2-7b-ov-hf`` (~14 GB).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)


class LLaVAOneVision:
    """Frozen LLaVA-OneVision wrapper for inference-only evaluation."""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.model_name = model_name
        self.dtype = dtype

        log.info("loading LLaVA-OneVision processor: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        log.info("loading LLaVA-OneVision model (%s, dtype=%s)...", model_name, dtype)
        device_arg: str | None = None if device == "auto" else str(device)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_arg or "auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        frames: Sequence[np.ndarray],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        from PIL import Image

        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images]
                       + [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text, images=images, return_tensors="pt",
        ).to(self.device)

        do_sample = temperature > 0.0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-7) if do_sample else None,
        )
        input_len = inputs.input_ids.shape[1]
        gen_only = outputs[:, input_len:]
        return self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
