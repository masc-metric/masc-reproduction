"""DINOv3 CLS-token global cosine baseline (modern DINO).

Modern analog of DB++'s shipped DINO-I baseline. DB++ uses `dino_vits8`
(DINOv1, ICCV 2021) which 2024-2026 personalization papers (IP-Adapter,
InstantID, PuLID, MS-Diffusion, OmniGen) have moved past in favor of
DINOv2 / DINOv3. This module reproduces the same CLS-cosine recipe but
on a modern DINOv3 checkpoint.

Default: `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, ~300M params,
LVD-1689M pretraining). DINOv3 prefixes the patch sequence with CLS plus
N register tokens; the CLS token sits at position 0 and is what we cosine.

Preprocessing follows the model card's canonical recipe (per DINOv3 HF
docs): resize to a multiple of 16, ImageNet normalize. We use
`Resize(256, BICUBIC) → CenterCrop(224)` for parity with `dinoi_dbplus.py`'s
DB++ recipe — keeps the only changed axis the backbone itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_INPUT_SIZE = 224  # match DB++'s DINO-I crop


@dataclass
class Dinov3DbplusMatcher:
    model: torch.nn.Module
    transform: T.Compose
    device: str

    @torch.inference_mode()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        out = self.model(t)
        # DINOv3's HF AutoModel exposes the full token sequence as
        # `last_hidden_state` (CLS + N registers + patches). CLS is index 0.
        cls = out.last_hidden_state[:, 0, :]  # (1, D)
        return F.normalize(cls, dim=-1)[0]

    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        # Masks unused — CLS-token global cosine.
        ref = self._embed(ref_image)
        out = self._embed(out_image)
        cos = float((ref * out).sum().item())
        return {
            "score": cos,
            "extras": {
                "score_type": "dinov3_cls_cosine",
                "cos_x100": cos * 100.0,
            },
        }


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: str = "cuda",
) -> Dinov3DbplusMatcher:
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    # Mirror dino_vits8's preprocessing (Resize 256 BICUBIC → CenterCrop 224
    # → Normalize). This keeps the backbone-swap clean: only the model
    # changes, preprocessing stays at DB++'s DINO-I convention. Use the
    # model's own ImageNet mean/std (matches DINOv3 HF defaults).
    resize_short = max(256, int(input_size * 256 / 224))  # preserve 256/224 ratio
    transform = T.Compose([
        T.Resize(resize_short, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=list(proc.image_mean), std=list(proc.image_std)),
    ])
    return Dinov3DbplusMatcher(model=model, transform=transform, device=device)
