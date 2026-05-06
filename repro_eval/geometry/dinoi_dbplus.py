"""DreamBench++ DINO-I baseline.

DINO-v1 ViT-S/8 (`dino_vits8`) from torch.hub, CLS-token cosine. Matches
DB++'s `DinoScore` recipe:
    model   = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    preproc = Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor
              -> Normalize(ImageNet mean/std)
    score   = 100 * cos(normalize(f_ref), normalize(f_out))

`f_*` is the raw forward output of the hub model for `dino_vits8`, which
is the CLS token. No mask, no per-patch aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

DEFAULT_HUB_REPO = "facebookresearch/dino:main"
DEFAULT_HUB_MODEL = "dino_vits8"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DinoIDbplusMatcher:
    model: torch.nn.Module
    transform: T.Compose
    device: str

    @torch.inference_mode()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        feats = self.model(t)  # (1, D) CLS token
        return F.normalize(feats, dim=-1)[0]

    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        ref = self._embed(ref_image)
        out = self._embed(out_image)
        cos = float((ref * out).sum().item())
        return {
            "score": cos,
            "extras": {
                "score_type": "pooled_cosine",
                "cos_x100": cos * 100.0,
            },
        }


def load_matcher(
    hub_repo: str = DEFAULT_HUB_REPO,
    hub_model: str = DEFAULT_HUB_MODEL,
    device: str = "cuda",
) -> DinoIDbplusMatcher:
    model = torch.hub.load(hub_repo, hub_model).eval().to(device)
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return DinoIDbplusMatcher(model=model, transform=transform, device=device)
