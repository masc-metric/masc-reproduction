"""SigLIP2 global pooled image-image cosine baseline.

Same-backbone, same-checkpoint counterpart to our masked-maxcos recipe in
`siglip2.py` — uses the model's trained `SiglipMultiheadAttentionPoolingHead`
output (the same vector that goes into the joint image-text contrastive
space) as a global descriptor and reports cosine. Direct apples-to-apples
analog of how DB++ uses CLIP-I (`visual_projection(pooler_output)` cosine)
and DINO-I (`dino_vits8` CLS cosine), and the missing same-encoder ablation
that isolates "what does masked-maxcos buy us over a global pool on the
same patch features?"

Supports both fixed-size SigLIP2 checkpoints and NaFlex variants. NaFlex
path threads `pixel_attention_mask` + `spatial_shapes` through the vision
tower so the trained pool sees a variable-resolution patch sequence.

Implementation note: bypasses `model.get_image_features(...)` because in
transformers 5.5.4 that returns a `BaseModelOutputWithPooling` wrapper
rather than the projected tensor (same gotcha as `clipi_dbplus.py`). We
call `model.vision_model(...)` directly and read `pooler_output` — this
IS the joint-space image embedding, since SigLIP2's contrastive loss is
between `vision_model.pooler_output` and `text_model.pooler_output`.

NOTE: Distinct from `siglip2.py`'s patch-cosine recipe (PatchFgMatcher);
this module ignores both masks and emits a single per-pair cosine.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "google/siglip2-so400m-patch16-naflex"
DEFAULT_NAFLEX_MAX_PATCHES = 1024


@dataclass
class Siglip2GlobalMatcher:
    model: torch.nn.Module
    processor: object
    device: str
    is_naflex: bool
    naflex_max_patches: int

    @torch.inference_mode()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        kwargs = dict(images=[image.convert("RGB")], return_tensors="pt")
        if self.is_naflex:
            kwargs["max_num_patches"] = self.naflex_max_patches
        try:
            inputs = self.processor(**kwargs)
        except TypeError:
            kwargs.pop("max_num_patches", None)
            inputs = self.processor(**kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run the vision tower directly. NaFlex needs `attention_mask` and
        # `spatial_shapes` (the inner Siglip2VisionTransformer takes
        # `attention_mask`, not `pixel_attention_mask`; same kwarg-name
        # caveat as `_patch_match.py:119-126`). Non-NaFlex takes only
        # `pixel_values`.
        if self.is_naflex:
            vis_out = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )
        else:
            vis_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        # `pooler_output` is the trained MultiheadAttentionPoolingHead
        # output — the joint-space image embedding (cosine-comparable to
        # `text_model(...).pooler_output`).
        pooled = vis_out.pooler_output  # (1, D)
        return F.normalize(pooled, dim=-1)[0]

    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        # Masks unused — global pooled cosine.
        ref = self._embed(ref_image)
        out = self._embed(out_image)
        cos = float((ref * out).sum().item())
        return {
            "score": cos,
            "extras": {
                "score_type": "siglip2_global_cosine",
                "cos_x100": cos * 100.0,
            },
        }


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    naflex_max_patches: int = DEFAULT_NAFLEX_MAX_PATCHES,
    device: str = "cuda",
) -> Siglip2GlobalMatcher:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    is_naflex = "naflex" in model_id.lower()
    return Siglip2GlobalMatcher(
        model=model,
        processor=processor,
        device=device,
        is_naflex=is_naflex,
        naflex_max_patches=naflex_max_patches,
    )
