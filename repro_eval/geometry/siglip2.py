"""Geometry control via SigLIP2 patch-cosine matching.

Supports every SigLIP2 variant on HF:
- Standard: base / large / so400m / giant-opt at patch16-{256,384,512}.
  Path: fixed-size resize → ViT forward → square patch grid.
- NaFlex: `*-patch16-naflex`. Path: processor pre-patchification →
  (pixel_values, pixel_attention_mask, spatial_shapes) → forward.
  `naflex_max_patches` budgets the patch count (default 1024 = 32×32
  grid for square DB++ inputs, matching patch16-512 baseline).

SigLIP2 has no CLS token; `last_hidden_state` is all patch tokens.
"""
from __future__ import annotations

from transformers import AutoImageProcessor, AutoModel

from ._patch_match import PatchFgMatcher

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-512"
DEFAULT_INPUT_SIZE = 512  # native for this checkpoint
DEFAULT_NAFLEX_MAX_PATCHES = 1024


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    fg_threshold: float = 0.5,
    score_type: str = "recall_fg",
    naflex_max_patches: int = DEFAULT_NAFLEX_MAX_PATCHES,
    mask_blur_sigma: float = 0.0,
    device: str = "cuda",
) -> PatchFgMatcher:
    proc = AutoImageProcessor.from_pretrained(model_id)
    raw = AutoModel.from_pretrained(model_id).eval().to(device)
    vision = raw.vision_model

    patch_size = vision.config.patch_size
    is_naflex = "naflex" in model_id.lower()

    # For non-NaFlex, input_size must be divisible by patch_size (fixed grid).
    # For NaFlex, input_size / patch_size is only used as a nominal hint; the
    # actual grid is set by max_num_patches at processor time.
    if not is_naflex and input_size % patch_size != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by patch_size={patch_size}"
        )

    return PatchFgMatcher(
        model=vision,
        input_size=input_size,
        patch_size=patch_size,
        n_prefix_tokens=0,  # SigLIP2: no CLS
        image_mean=list(proc.image_mean),
        image_std=list(proc.image_std),
        device=device,
        fg_threshold=fg_threshold,
        score_type=score_type,
        naflex_processor=(proc if is_naflex else None),
        naflex_max_patches=(naflex_max_patches if is_naflex else 0),
        mask_blur_sigma=mask_blur_sigma,
    )
