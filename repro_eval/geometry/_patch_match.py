"""Shared patch-cosine matcher, usable as either a mutual-NN recall-style
foreground-recall metric or a vanilla-DIFT-style global similarity
baseline. All three scores are always computed; `score_type` picks
which becomes the top-level `score`.

`score_type="recall_fg"` (default) — mutual-NN foreground-recall semantic:

1. Compute mutual nearest-neighbor matches over ALL patches (global).
2. For each mutual match, check fg membership on both sides via the
   patch-grid-downsampled mask.
3. Score = n_both_in_fg / n_ref_in_fg where
     n_ref_in_fg  = # mutual matches whose ref patch is in fg,
     n_both_in_fg = # mutual matches whose ref AND out patch are in fg.

   Reads as "of mutual matches sourced in the ref concept region, what
   fraction land in the out concept region."

`score_type="global_maxcos"` — vanilla baseline:

1. For each ref patch (regardless of fg), take its max cosine to any
   out patch.
2. Score = mean of those max-cosines across all ref patches.

   No mask, no mutuality — the standard feature-space similarity
   aggregate used by DINO-score / CLIP-I-style metrics. Used to
   position DIFT as a "beat me" baseline for the mutual-NN recall-style metric.

`score_type="masked_maxcos"` — mask-restricted maxcos ablation:

1. For each ref patch in the fg, take its max cosine to any out patch.
2. Score = mean of those max-cosines across fg-only ref patches.

   Isolates "what does the mask buy us" from "what does the recall
   formula buy us": same aggregator as global_maxcos, but restricted to
   the concept region. If masked_maxcos > global_maxcos, the mask helps;
   if masked_maxcos > recall_fg, the recall formula is too strict.

`score_type="weighted_maxcos"` — soft-mask-weighted maxcos ablation:

1. For every ref patch, compute its max cosine to any out patch.
2. Weight each patch by its fractional fg coverage w_p ∈ [0, 1]
   (fg pixels / total pixels in that patch, no 0.5 threshold).
3. Score = Σ(w_p · max_cos_p) / Σ(w_p).

   Same signal as masked_maxcos but without the hard threshold —
   boundary patches that straddle the mask contribute partially,
   proportional to how much of them is actually foreground.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass
class PatchFgMatcher:
    model: torch.nn.Module  # must accept pixel_values and return .last_hidden_state
    input_size: int
    patch_size: int
    n_prefix_tokens: int
    image_mean: list[float]
    image_std: list[float]
    device: str
    fg_threshold: float = 0.5
    score_type: str = "recall_fg"  # or "global_maxcos"
    # SigLIP2-NaFlex support. When `naflex_processor` is set, we skip the
    # fixed-size resize path and feed the processor-packed
    # (pixel_values, pixel_attention_mask, spatial_shapes) triple into
    # the vision model instead. `naflex_max_patches` controls the patch
    # budget (default 256; 1024 gives a 32×32 square grid matching our
    # patch16-512 baseline).
    naflex_processor: object = None
    naflex_max_patches: int = 0
    # Gaussian blur applied to the fg mask BEFORE downsample to the patch
    # grid, so boundary patches pick up a wider band of fractional weights.
    # Units: pixels at the (Hp * patch_size) intermediate resolution — for
    # 32×32 grid with patch_size=16, sigma=16 ≈ one patch wide.
    # 0.0 disables (default, matches baseline masked_maxcos behavior).
    mask_blur_sigma: float = 0.0

    @torch.inference_mode()
    def _embed_patches(
        self, image: Image.Image, fg_mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (patch_tokens (N, D), fg_weights (N,) in [0, 1])."""
        if self.naflex_processor is not None:
            return self._embed_patches_naflex(image, fg_mask)

        pixel_values = self._preprocess_image(image)
        fg = self._preprocess_mask_to_patch_grid(fg_mask)  # (Hp, Wp)

        out = self.model(pixel_values)
        patch_tokens = out.last_hidden_state[0, self.n_prefix_tokens:, :]  # (Hp*Wp, D)

        Hp = Wp = self.input_size // self.patch_size
        assert patch_tokens.shape[0] == Hp * Wp, (patch_tokens.shape, Hp, Wp)

        return patch_tokens, fg.flatten().to(patch_tokens.dtype)

    @torch.inference_mode()
    def _embed_patches_naflex(
        self, image: Image.Image, fg_mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NaFlex path: pre-patchification via the processor, which emits
        variable-size (pixel_values, pixel_attention_mask, spatial_shapes)
        triples. For our square DB++ inputs we resolve to a fixed Hp×Wp
        grid determined by max_num_patches — equivalent patch count as
        the patch16-<hw> baseline, only the training recipe differs."""
        inputs = self.naflex_processor(
            images=image.convert("RGB"),
            max_num_patches=self.naflex_max_patches,
            return_tensors="pt",
        )
        # `siglip2.py` passes us `raw.vision_model`, which on transformers
        # 5.6+ resolves to `Siglip2VisionModel.forward(pixel_values,
        # pixel_attention_mask, spatial_shapes)`. The kwarg name is
        # `pixel_attention_mask`, not `attention_mask` — the same gotcha
        # that bit `siglip2_global.py`. Earlier transformers (≤ 5.5)
        # accepted the looser `attention_mask=` kwarg silently; current
        # versions raise a TypeError without it.
        fwd = {
            "pixel_values": inputs["pixel_values"].to(self.device),
            "pixel_attention_mask": inputs["pixel_attention_mask"].to(self.device),
            "spatial_shapes": inputs["spatial_shapes"].to(self.device),
        }
        Hp, Wp = [int(x) for x in inputs["spatial_shapes"][0].tolist()]

        out = self.model(**fwd)
        valid = fwd["pixel_attention_mask"][0] > 0
        patch_tokens = out.last_hidden_state[0][valid]  # (Hp*Wp, D)
        assert patch_tokens.shape[0] == Hp * Wp, (patch_tokens.shape, Hp, Wp)

        fg = self._downsample_mask(fg_mask, Hp, Wp)
        return patch_tokens, fg.flatten().to(patch_tokens.dtype)

    def fg_match(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
        out_fg_mask: np.ndarray,
    ) -> dict:
        ref_tok, ref_w = self._embed_patches(ref_image, ref_fg_mask)
        out_tok, out_w = self._embed_patches(out_image, out_fg_mask)

        ref_unit = F.normalize(ref_tok, dim=-1)
        out_unit = F.normalize(out_tok, dim=-1)
        r2o, o2r, nn_cos_per_ref = _mutual_nn(ref_unit, out_unit)

        n_ref = ref_unit.shape[0]
        arange_ref = torch.arange(n_ref, device=r2o.device)
        mutual = (o2r[r2o] == arange_ref)

        ref_in_fg = ref_w > self.fg_threshold
        out_in_fg = out_w > self.fg_threshold
        partner_in_fg = out_in_fg[r2o]

        ref_match_fg = mutual & ref_in_fg
        both_match_fg = ref_match_fg & partner_in_fg

        n_ref_in_fg = int(ref_match_fg.sum())
        n_both_in_fg = int(both_match_fg.sum())
        recall_fg_score = n_both_in_fg / n_ref_in_fg if n_ref_in_fg > 0 else 0.0

        mean_nn_cos_fg = (
            float(nn_cos_per_ref[ref_match_fg].mean()) if n_ref_in_fg > 0 else 0.0
        )

        # Global max-cos baseline: ignore masks and mutuality.
        global_maxcos_score = float(nn_cos_per_ref.mean())

        # Masked max-cos ablation: mask restriction, no mutuality.
        n_ref_fg_only = int(ref_in_fg.sum())
        masked_maxcos_score = (
            float(nn_cos_per_ref[ref_in_fg].mean()) if n_ref_fg_only > 0 else 0.0
        )

        # Weighted max-cos ablation: soft fg-fraction weights in [0, 1],
        # no threshold. Score = Σ(w_p · max_cos_p) / Σ(w_p).
        total_ref_w = float(ref_w.sum())
        weighted_maxcos_score = (
            float((ref_w * nn_cos_per_ref).sum() / total_ref_w)
            if total_ref_w > 0 else 0.0
        )

        if self.score_type == "global_maxcos":
            score = global_maxcos_score
        elif self.score_type == "recall_fg":
            score = recall_fg_score
        elif self.score_type == "masked_maxcos":
            score = masked_maxcos_score
        elif self.score_type == "weighted_maxcos":
            score = weighted_maxcos_score
        else:
            raise ValueError(f"Unknown score_type: {self.score_type!r}")

        return {
            "score": score,
            "extras": {
                "score_type": self.score_type,
                "recall_fg_score": recall_fg_score,
                "global_maxcos_score": global_maxcos_score,
                "masked_maxcos_score": masked_maxcos_score,
                "weighted_maxcos_score": weighted_maxcos_score,
                "n_ref_in_fg": n_ref_in_fg,
                "n_ref_fg_only": n_ref_fg_only,
                "n_both_in_fg": n_both_in_fg,
                "n_mutual_total": int(mutual.sum()),
                "n_ref_patches_total": int(n_ref),
                "mean_nn_cos_fg": mean_nn_cos_fg,
                "fg_frac_ref": float(ref_w.mean()),
                "fg_frac_out": float(out_w.mean()),
                "total_ref_w": total_ref_w,
            },
        }

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize(
            (self.input_size, self.input_size), Image.BILINEAR
        )
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=self.image_mean, std=self.image_std)
        return t.unsqueeze(0).to(self.device)

    def _preprocess_mask_to_patch_grid(self, fg_mask: np.ndarray) -> torch.Tensor:
        Hp = Wp = self.input_size // self.patch_size
        return self._downsample_mask(fg_mask, Hp, Wp)

    def _downsample_mask(self, fg_mask: np.ndarray, Hp: int, Wp: int) -> torch.Tensor:
        """Shared bilinear (+ optional Gaussian blur) downsample to Hp×Wp.

        If `mask_blur_sigma > 0`, the mask is first resized to the patch-
        aligned intermediate resolution (Hp·patch_size × Wp·patch_size),
        blurred with sigma in pixels at that resolution, and only then
        downsampled to (Hp, Wp). This widens the boundary band at the
        patch grid, so weighted_maxcos has non-trivial soft weights to
        work with.
        """
        m = torch.from_numpy(fg_mask.astype(np.float32) / 255.0)
        m = m.unsqueeze(0).unsqueeze(0)
        if self.mask_blur_sigma > 0:
            H_int, W_int = Hp * self.patch_size, Wp * self.patch_size
            m = F.interpolate(m, size=(H_int, W_int),
                              mode="bilinear", align_corners=False)
            k = max(3, int(2 * round(3 * self.mask_blur_sigma) + 1))
            if k % 2 == 0:
                k += 1
            m = TF.gaussian_blur(m, kernel_size=[k, k],
                                 sigma=[self.mask_blur_sigma, self.mask_blur_sigma])
        m = F.interpolate(m, size=(Hp, Wp), mode="bilinear", align_corners=False)
        return m[0, 0].clamp(0.0, 1.0).to(self.device)


def _mutual_nn(
    ref_unit: torch.Tensor, out_unit: torch.Tensor, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked argmax on both axes of the cosine matrix.

    Returns (r2o, o2r, nn_cos_per_ref):
        r2o[i] = argmax_j cos(ref_i, out_j)
        o2r[j] = argmax_i cos(ref_i, out_j)
        nn_cos_per_ref[i] = cos(ref_i, out_{r2o[i]})

    Avoids materializing the full (n_ref, n_out) cosine matrix at once,
    which matters for 128x128 grids (~16k tokens) on GPUs shared with a
    large model (e.g. SDXL for DIFT).
    """
    n_ref = ref_unit.shape[0]
    n_out = out_unit.shape[0]
    device = ref_unit.device

    r2o = torch.empty(n_ref, dtype=torch.long, device=device)
    nn_cos_per_ref = torch.empty(n_ref, device=device, dtype=ref_unit.dtype)
    o2r_max = torch.full((n_out,), float("-inf"), device=device, dtype=ref_unit.dtype)
    o2r = torch.empty(n_out, dtype=torch.long, device=device)

    for i in range(0, n_ref, chunk):
        sl = slice(i, i + chunk)
        sims = ref_unit[sl] @ out_unit.T  # (chunk, n_out)
        chunk_max, chunk_idx = sims.max(dim=1)
        r2o[sl] = chunk_idx
        nn_cos_per_ref[sl] = chunk_max

        # Update o2r running argmax.
        col_max, col_idx = sims.max(dim=0)
        better = col_max > o2r_max
        o2r_max = torch.where(better, col_max, o2r_max)
        o2r = torch.where(better, col_idx + i, o2r)

    return r2o, o2r, nn_cos_per_ref
