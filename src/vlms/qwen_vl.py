"""Qwen2.5-VL inference wrapper.

The HF checkpoint is ``Qwen/Qwen2.5-VL-7B-Instruct`` (~14 GB in fp16).
Loading is lazy — instantiating this class triggers the download/load,
which is the dominant compute cost. After construction, :meth:`generate`
is a single call per sample that emits a string response.

The wrapper is deliberately minimal: image preprocessing and chat
templating live in the HF processor, and we only expose a generate
interface. The run script handles JSON parsing downstream via
:mod:`src.vlms.output_parser`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)


class Qwen25VL:
    """Frozen Qwen2.5-VL wrapper for inference-only evaluation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name = model_name
        self.dtype = dtype

        log.info("loading Qwen2.5-VL processor: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        log.info("loading Qwen2.5-VL model (%s, dtype=%s)...", model_name, dtype)
        device_arg: str | None = None if device == "auto" else str(device)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            "content": [{"type": "image", "image": img} for img in images]
                       + [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=images, padding=True, return_tensors="pt",
        ).to(self.device)

        do_sample = temperature > 0.0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-7) if do_sample else None,
        )
        # Trim the prompt tokens off so we keep only the model's response.
        input_len = inputs.input_ids.shape[1]
        gen_only = outputs[:, input_len:]
        return self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
