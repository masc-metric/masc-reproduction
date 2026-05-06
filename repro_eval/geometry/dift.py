"""Geometry matcher via DIFT on SDXL.

Extracts intermediate U-Net features from a frozen SDXL model at a
specific timestep + up-block, then reuses the patch-cosine mutual-NN
matcher in `_patch_match.py` — DIFT plugs in wherever a ViT backbone
would. Paper: "Emergent Correspondence from Image Diffusion" (Tang et
al., NeurIPS 2023).

Extra dep: `pip install diffusers accelerate`.

Hyperparameters are not tuned — DIFT paper reports `timestep=261` for
SD1.5 and `up_ft_index=1`; SDXL conventions vary in practice, so the
defaults are a reasonable first pass (null prompt, mid-early timestep,
second up-block). Tune via the YAML config.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from ._patch_match import PatchFgMatcher

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_INPUT_SIZE = 1024     # SDXL native training resolution
DEFAULT_TIMESTEP = 101        # early-mid denoising for semantic features
DEFAULT_UP_FT_INDEX = 1       # output of second up-block
DEFAULT_PROMPT = ""           # null prompt


class SDXLDIFTModel(torch.nn.Module):
    """Wraps SDXL VAE + U-Net (+ null-prompt text embeddings) to expose a
    `(pixel_values) -> SimpleNamespace(last_hidden_state=(1, Hp*Wp, C))` API
    so `PatchFgMatcher` can treat DIFT like any other ViT backbone.
    """

    def __init__(
        self,
        model_id: str,
        timestep: int,
        up_ft_index: int,
        input_size: int,
        prompt: str,
        device: str,
        dtype: torch.dtype,
        seed: int,
    ):
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.vae = (
            AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            .to(device=device, dtype=dtype).eval()
        )
        self.unet = (
            UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            .to(device=device, dtype=dtype).eval()
        )
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        te = (
            CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
            .to(device=device, dtype=dtype).eval()
        )
        te2 = (
            CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
            .to(device=device, dtype=dtype).eval()
        )

        prompt_embeds, pooled = self._encode_prompt(prompt, tok, tok2, te, te2, device, dtype)
        self.register_buffer("prompt_embeds", prompt_embeds, persistent=False)
        self.register_buffer("pooled_embeds", pooled, persistent=False)

        # SDXL conditioning: [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
        # Eval: no crops, target = input_size.
        add_time_ids = torch.tensor(
            [[input_size, input_size, 0, 0, input_size, input_size]],
            device=device, dtype=dtype,
        )
        self.register_buffer("add_time_ids", add_time_ids, persistent=False)

        del te, te2, tok, tok2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.timestep = timestep
        self.up_ft_index = up_ft_index
        self.model_device = device
        self.model_dtype = dtype
        self.seed = seed

        self._hooked_feat: torch.Tensor | None = None
        self.unet.up_blocks[up_ft_index].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._hooked_feat = output[0] if isinstance(output, tuple) else output

    @staticmethod
    @torch.inference_mode()
    def _encode_prompt(prompt, tok, tok2, te, te2, device, dtype):
        def tokenize(t, string):
            return t(
                string,
                padding="max_length",
                max_length=t.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

        ids1 = tokenize(tok, prompt)
        ids2 = tokenize(tok2, prompt)
        o1 = te(ids1, output_hidden_states=True)
        o2 = te2(ids2, output_hidden_states=True)
        # SDXL concatenates penultimate hidden states from both text encoders.
        embeds = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1).to(dtype)
        pooled = o2[0].to(dtype)  # projected pooled output from TE2
        return embeds, pooled

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        x = pixel_values.to(self.model_dtype)
        latents = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        gen = torch.Generator(device=self.model_device).manual_seed(self.seed)
        noise = torch.randn(
            latents.shape, generator=gen, device=self.model_device, dtype=self.model_dtype
        )
        t = torch.tensor([self.timestep], device=self.model_device, dtype=torch.long)
        noisy = self.scheduler.add_noise(latents, noise, t)

        _ = self.unet(
            noisy,
            t,
            encoder_hidden_states=self.prompt_embeds,
            added_cond_kwargs={
                "text_embeds": self.pooled_embeds,
                "time_ids": self.add_time_ids,
            },
            return_dict=False,
        )

        feat = self._hooked_feat
        assert feat is not None, "forward hook did not fire"
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous().float()
        return SimpleNamespace(last_hidden_state=tokens)

    def probe_patch_size(self, input_size: int) -> int:
        dummy = torch.zeros(1, 3, input_size, input_size,
                            device=self.model_device, dtype=self.model_dtype)
        self.forward(dummy)
        assert self._hooked_feat is not None
        H = self._hooked_feat.shape[-1]
        if input_size % H != 0:
            raise ValueError(
                f"input_size={input_size} not divisible by DIFT feature grid H={H}"
            )
        return input_size // H


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    timestep: int = DEFAULT_TIMESTEP,
    up_ft_index: int = DEFAULT_UP_FT_INDEX,
    prompt: str = DEFAULT_PROMPT,
    seed: int = 0,
    fg_threshold: float = 0.5,
    score_type: str = "recall_fg",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> PatchFgMatcher:
    model = SDXLDIFTModel(
        model_id=model_id,
        timestep=timestep,
        up_ft_index=up_ft_index,
        input_size=input_size,
        prompt=prompt,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    patch_size = model.probe_patch_size(input_size)

    return PatchFgMatcher(
        model=model,
        input_size=input_size,
        patch_size=patch_size,
        n_prefix_tokens=0,
        image_mean=[0.5, 0.5, 0.5],  # [0,1] → [-1,1] for SDXL VAE
        image_std=[0.5, 0.5, 0.5],
        device=device,
        fg_threshold=fg_threshold,
        score_type=score_type,
    )
