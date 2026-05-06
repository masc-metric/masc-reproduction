"""DreamSim perceptual-similarity baseline.

DreamSim (Fu et al., NeurIPS 2023; arXiv:2306.09344) — ensemble of fine-tuned
DINO + CLIP + OpenCLIP ViTs trained on the NIGHTS dataset of human triplet
similarity judgments. Returns a distance d ∈ [0, 1]; we report similarity
= 1 − d so higher = more similar, matching our other CP baselines.

NOTE: DreamSim *is* trained on human similarity labels — but on perceptual
similarity triplets, not T2I generation preferences. It's a perceptual
baseline, not a personalization-specific reward model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

DEFAULT_TYPE = "ensemble"
DEFAULT_CACHE_DIR = ".cache/dreamsim"


@dataclass
class DreamSimMatcher:
    model: torch.nn.Module
    preprocess: object
    device: str

    @torch.inference_mode()
    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        # masks unused — DreamSim is a global-image perceptual metric
        ref_t = self.preprocess(ref_image.convert("RGB")).to(self.device)
        out_t = self.preprocess(out_image.convert("RGB")).to(self.device)
        distance = float(self.model(ref_t, out_t).item())
        return {
            "score": 1.0 - distance,
            "extras": {
                "score_type": "dreamsim_similarity",
                "distance": distance,
            },
        }


def load_matcher(
    dreamsim_type: str = DEFAULT_TYPE,
    cache_dir: str = DEFAULT_CACHE_DIR,
    device: str = "cuda",
) -> DreamSimMatcher:
    from dreamsim import dreamsim

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    model, preprocess = dreamsim(
        pretrained=True,
        dreamsim_type=dreamsim_type,
        device=device,
        cache_dir=cache_dir,
    )
    return DreamSimMatcher(model=model, preprocess=preprocess, device=device)
