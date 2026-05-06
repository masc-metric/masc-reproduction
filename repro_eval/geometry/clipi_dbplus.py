"""DreamBench++ CLIP-I baseline.

Global pooled cosine between `CLIPModel.get_image_features(...)` outputs.
Matches DB++'s `CLIPScore.clipi_score` recipe:
    model_id = "openai/clip-vit-base-patch32"
    score = 100 * cos(normalize(f_ref), normalize(f_out))

No mask, no per-patch aggregation. `fg_match()` ignores both masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


@dataclass
class ClipIDbplusMatcher:
    model: CLIPModel
    processor: object
    device: str

    @torch.inference_mode()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        # Bypass `get_image_features` — in transformers 5.5.4 it returns a
        # `BaseModelOutputWithPooling` (vision-model output) rather than the
        # projected tensor. Replicate DB++'s recipe manually:
        #   pooled = vision_model(pixel_values)[1]
        #   feats  = visual_projection(pooled)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device)
        vision_out = self.model.vision_model(pixel_values=pix)
        pooled = vision_out.pooler_output  # (1, D)
        feats = self.model.visual_projection(pooled)
        return F.normalize(feats, dim=-1)[0]

    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        # masks unused
        ref = self._embed(ref_image)
        out = self._embed(out_image)
        cos = float((ref * out).sum().item())
        return {
            "score": cos,  # unscaled cosine in [-1, 1]; DB++ scales ×100
            "extras": {
                "score_type": "pooled_cosine",
                "cos_x100": cos * 100.0,
            },
        }


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
) -> ClipIDbplusMatcher:
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return ClipIDbplusMatcher(model=model, processor=processor, device=device)
