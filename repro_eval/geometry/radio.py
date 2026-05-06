"""Geometry control via NVIDIA C-RADIOv4 patch-cosine matching.

C-RADIOv4 is an agglomerative multi-teacher distillation: a single ViT
trained to reproduce the patch features of SigLIP2-g, DINOv3-7B, and
SAM3 simultaneously. The SO400M-size variant (431M params) is the
direct architectural counterpart to our SigLIP2 SO400M-NaFlex champion
backbone — same parameter footprint, same patch_size=16 grid, but with
SAM3 features baked into the patch manifold rather than applied as a
post-hoc segmentation mask.

The model returns `(summary, features)` where `features` is already a
pure patch sequence `(B, T, D)` with no prefix tokens. We therefore set
`n_prefix_tokens=0` and adapt the call site to expose `.last_hidden_state`
for `_patch_match.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from ._patch_match import PatchFgMatcher

DEFAULT_MODEL_ID = "nvidia/C-RADIOv4-SO400M"
DEFAULT_INPUT_SIZE = 512  # 32x32 patch grid at patch_size=16


class _RadioVisionAdapter(nn.Module):
    """Wraps RADIO so `model(pixel_values).last_hidden_state` returns the
    patch features, mirroring the HF ViT interface that `_patch_match.py`
    expects. RADIO's native return is `(summary, features)` — we discard
    the summary (the CLS-equivalent global vector is irrelevant for our
    dense patch-cosine algorithm)."""

    def __init__(self, raw: nn.Module) -> None:
        super().__init__()
        self.raw = raw

    def forward(self, pixel_values: torch.Tensor):
        # RADIO trained with bf16 amp (per HF config); use the same dtype at
        # inference. ~3x faster than fp32 on Ampere+ and matches training
        # numerics. Cast the output back to fp32 so downstream cosine math
        # (in `_patch_match.py`) stays in single precision.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.raw(pixel_values)
        # RADIO emits a `RadioOutput` namedtuple with `.summary` and
        # `.features`. The `features` tensor is `(B, Hp*Wp, D)` and is
        # already pure patches (no CLS / register tokens prepended).
        features = out.features if hasattr(out, "features") else out[1]
        return type("R", (), {"last_hidden_state": features.float()})()


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    fg_threshold: float = 0.5,
    score_type: str = "masked_maxcos",
    device: str = "cuda",
) -> PatchFgMatcher:
    raw = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(device)
    vision = _RadioVisionAdapter(raw)

    patch_size = 16
    if input_size % patch_size != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by patch_size={patch_size}"
        )

    # CRITICAL: do NOT apply CLIP mean/std externally. RADIO's `RADIOModel`
    # documents that input must be [0, 1]; its internal `InputConditioner`
    # (initialized with OpenAI CLIP mean/std by default) applies normalization
    # inside the model. External normalization here would silently double-
    # normalize and compress feature scores. Pass identity-normalize stats
    # (mean=0, std=1) so `_patch_match.py`'s `_preprocess_image` becomes a
    # no-op for normalization while keeping the same resize+ToTensor path.
    return PatchFgMatcher(
        model=vision,
        input_size=input_size,
        patch_size=patch_size,
        n_prefix_tokens=0,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        device=device,
        fg_threshold=fg_threshold,
        score_type=score_type,
    )
