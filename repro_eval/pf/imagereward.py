"""ImageReward — Xu et al., NeurIPS 2023 (arXiv:2304.05977).

Recipe: BLIP backbone fine-tuned on 137k human pairwise preferences over
generic Stable Diffusion outputs. Returns a scalar reward (signed,
unbounded; typical range ~[-2, 2]).

Install: `pip install image-reward` (https://github.com/THUDM/ImageReward).

NOTE: ImageReward was trained on generic T2I distributions, not
personalization specifically. There's a real distribution shift on DB++
(personalization outputs); we report it as one of several
preference-trained reward baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_MODEL = "ImageReward-v1.0"


@dataclass
class ImageRewardScorer:
    model: object
    model_id: str

    def score(self, image: Image.Image, prompt_text: str) -> dict:
        # The PyPI package's `score(prompt, image)` accepts a PIL image
        # or a path. We pass PIL directly.
        s = float(self.model.score(prompt_text, image))
        return {
            "score": s,
            "extras": {
                "score_type": "imagereward_scalar",
                "model_id": self.model_id,
            },
        }


def load_scorer(
    model_id: str = DEFAULT_MODEL,
    device: str = "cuda",
) -> ImageRewardScorer:
    import ImageReward as RM  # noqa: N814 — package name
    model = RM.load(model_id, device=device)
    return ImageRewardScorer(model=model, model_id=model_id)
