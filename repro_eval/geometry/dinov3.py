"""Geometry control via DINOv3 patch-cosine matching."""
from __future__ import annotations

from transformers import AutoImageProcessor, AutoModel

from ._patch_match import PatchFgMatcher

DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_INPUT_SIZE = 512  # must be divisible by model patch size (16)


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    fg_threshold: float = 0.5,
    score_type: str = "recall_fg",
    device: str = "cuda",
) -> PatchFgMatcher:
    proc = AutoImageProcessor.from_pretrained(model_id)
    raw = AutoModel.from_pretrained(model_id).eval().to(device)
    vision = getattr(raw, "vision_model", raw)  # DINOv3 AutoModel IS the vision tower

    patch_size = vision.config.patch_size
    if input_size % patch_size != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by patch_size={patch_size}"
        )
    n_reg = getattr(vision.config, "num_register_tokens", 0)
    return PatchFgMatcher(
        model=vision,
        input_size=input_size,
        patch_size=patch_size,
        n_prefix_tokens=1 + n_reg,  # CLS + registers
        image_mean=list(proc.image_mean),
        image_std=list(proc.image_std),
        device=device,
        fg_threshold=fg_threshold,
        score_type=score_type,
    )
