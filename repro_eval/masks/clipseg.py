"""CLIPSeg text-prompted segmentation.

Thin wrapper around `CIDAS/clipseg-rd64-refined`. Same `segment` interface
as `Sam3Segmenter`: returns a uint8 {0, 255} mask at the input image's
native (H, W). The model stays loaded on GPU between calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


@dataclass
class CLIPSegSegmenter:
    processor: CLIPSegProcessor
    model: CLIPSegForImageSegmentation
    device: str
    threshold: float = 0.5

    @torch.inference_mode()
    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        image = image.convert("RGB")
        w, h = image.size
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        ).to(self.device)
        logits = self.model(**inputs).logits  # (1, H', W') at 352x352
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits.float()).unsqueeze(1)  # (1, 1, H', W')
        probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
        m = probs[0, 0].cpu().numpy()
        return ((m > self.threshold).astype(np.uint8)) * 255


def load_segmenter(
    model_id: str = "CIDAS/clipseg-rd64-refined",
    threshold: float = 0.5,
    device: str | None = None,
) -> CLIPSegSegmenter:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device).eval()
    return CLIPSegSegmenter(
        processor=processor, model=model, device=device, threshold=threshold
    )
