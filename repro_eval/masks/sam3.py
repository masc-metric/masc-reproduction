"""SAM3 text-prompted segmentation.

Thin wrapper around `sam3.Sam3Processor` that exposes a single
`segment(image, prompt) -> binary uint8 mask` call. The model stays
loaded on GPU between calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _default_bpe_path() -> str:
    return os.path.join(
        os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"
    )


@dataclass
class Sam3Segmenter:
    processor: Sam3Processor
    union_threshold: float = 0.5

    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        """Return a uint8 {0, 255} mask with the same (H, W) as the image."""
        image = image.convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            self.processor.reset_all_prompts(state)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = out.get("masks")
        w, h = image.size
        if masks is None or (hasattr(masks, "numel") and masks.numel() == 0):
            return np.zeros((h, w), dtype=np.uint8)

        m = masks.detach().cpu().float().numpy()
        flat = m.reshape(-1, m.shape[-2], m.shape[-1])
        return (flat.max(axis=0) > self.union_threshold).astype(np.uint8) * 255


def load_segmenter(
    confidence_threshold: float = 0.5,
    union_threshold: float = 0.5,
    bpe_path: str | None = None,
) -> Sam3Segmenter:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(bpe_path=bpe_path or _default_bpe_path())
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    return Sam3Segmenter(processor=processor, union_threshold=union_threshold)
