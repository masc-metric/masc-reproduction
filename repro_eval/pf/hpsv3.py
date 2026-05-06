"""HPSv3 — Ma et al., 2025 (arXiv:2503.10628).

Recipe: VLM (Qwen2-VL backbone) trained on HPDv3 human preference data;
outputs a (mu, sigma) reward; we report mu.

Install: `pip install hpsv3`. The package exposes
`hpsv3.HPSv3RewardInferencer`; its `.reward(image_paths, prompts)` takes
file paths (not PIL images) and returns a `(N, 2)` tensor where col 0 is
mu and col 1 is sigma. Default checkpoint is auto-downloaded from
`MizzenAI/HPSv3` on first use.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from PIL import Image

DEFAULT_MODEL = "MizzenAI/HPSv3"


@dataclass
class HPSv3Scorer:
    model: object  # HPSv3RewardInferencer
    model_id: str

    def score(self, image: Image.Image, prompt_text: str) -> dict:
        # The HPSv3 API takes file paths, so we write a temp PNG.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            image.save(tf, format="PNG")
            path = tf.name
        try:
            rewards = self.model.reward([path], [prompt_text])
        finally:
            os.unlink(path)
        # rewards: (1, 2) tensor — [mu, sigma]; we report mu.
        mu = float(rewards[0][0].item())
        sigma = float(rewards[0][1].item()) if rewards.shape[-1] >= 2 else 0.0
        return {
            "score": mu,
            "extras": {
                "score_type": "hpsv3_mu",
                "sigma": sigma,
                "model_id": self.model_id,
            },
        }


def load_scorer(
    model_id: str = DEFAULT_MODEL,
    device: str = "cuda",
) -> HPSv3Scorer:
    import torch
    from hpsv3 import HPSv3RewardInferencer
    # `model_id` is informational; HPSv3RewardInferencer auto-downloads the
    # MizzenAI/HPSv3 checkpoint when checkpoint_path is None.
    inferencer = HPSv3RewardInferencer(device=device)
    # The Qwen2-VL 7B backbone loads in fp32 (~28 GB) and OOMs a 24 GB GPU.
    # Cast to bf16 — matches HPSv3's own batched-inference example at
    # `hpsv3/inference.py` (`dtype = torch.bfloat16`).
    inferencer.model = inferencer.model.to(dtype=torch.bfloat16)
    return HPSv3Scorer(model=inferencer, model_id=model_id)
