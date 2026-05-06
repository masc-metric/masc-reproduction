"""Geometry control via CLIP ViT patch-cosine matching.

Vision tower of `openai/clip-vit-base-patch16`. CLIP position
embeddings are fixed at 224×224 in the `transformers` implementation we
use, so `input_size` is locked to 224 → 14×14 patch grid.
"""
from __future__ import annotations

from transformers import AutoImageProcessor, AutoModel

from ._patch_match import PatchFgMatcher

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch16"
DEFAULT_INPUT_SIZE = 224


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    fg_threshold: float = 0.5,
    device: str = "cuda",
) -> PatchFgMatcher:
    proc = AutoImageProcessor.from_pretrained(model_id)
    raw = AutoModel.from_pretrained(model_id).eval().to(device)
    vision = raw.vision_model

    patch_size = vision.config.patch_size
    if input_size % patch_size != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by patch_size={patch_size}"
        )
    return PatchFgMatcher(
        model=vision,
        input_size=input_size,
        patch_size=patch_size,
        n_prefix_tokens=1,  # CLIP: CLS only, no registers
        image_mean=list(proc.image_mean),
        image_std=list(proc.image_std),
        device=device,
        fg_threshold=fg_threshold,
    )
