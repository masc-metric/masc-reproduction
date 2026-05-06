"""Geometry via RoMa dense matching.

Installed per upstream README from `third_party/RoMa` (`pip install -e
third_party/RoMa`). The RoMa model outputs a dense warp + per-pixel
certainty at 864×864 upsample resolution; its `sample` method draws
`n_sample` high-certainty matches, which we convert back to each
image's native pixel frame via `to_pixel_coordinates`.

Two score paths, selected by `score_type`:

`score_type="recall_fg"` (default) — mutual-NN foreground-recall semantic, same formula as
the patch-cosine matchers. Sample `n_sample` high-certainty matches,
pixel-level fg lookup on both endpoints:

    score = n_both_in_fg / n_ref_in_fg

`score_type="masked_maxcos"` — mask-aware, dense, no sampling. Use the
per-pixel certainty map directly on the ref-side grid:

    score = mean(certainty[fg_ref_grid])

    For each fg-ref cell (on RoMa's 864×864 grid after upsample), the
    certainty is the sigmoid-normalized confidence that this ref
    location has a valid match anywhere in out. No mutuality, no
    out-side mask. Matches the `masked_maxcos` semantic in
    `_patch_match.py` but with RoMa certainty standing in for patch
    cosine — the model's internal confidence ∈ [0, 1] already encodes
    "best match anywhere in out" because RoMa's dense warp is predicted
    per-ref-pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from romatch import roma_outdoor


@dataclass
class RoMaMatcher:
    model: torch.nn.Module
    device: str
    n_sample: int
    fg_threshold: float = 0.5
    score_type: str = "recall_fg"  # or "masked_maxcos"

    def fg_match(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
        out_fg_mask: np.ndarray,
    ) -> dict:
        if self.score_type == "masked_maxcos":
            return self._masked_maxcos(ref_image, ref_fg_mask, out_image)
        if self.score_type == "recall_fg":
            return self._recall_fg(ref_image, ref_fg_mask, out_image, out_fg_mask)
        raise ValueError(f"Unknown score_type: {self.score_type!r}")

    @torch.inference_mode()
    def _recall_fg(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
        out_fg_mask: np.ndarray,
    ) -> dict:
        ref_rgb = ref_image.convert("RGB")
        out_rgb = out_image.convert("RGB")
        W_ref, H_ref = ref_rgb.size
        W_out, H_out = out_rgb.size

        warp, certainty = self.model.match(ref_rgb, out_rgb, device=self.device)
        matches, sampled_cert = self.model.sample(warp, certainty, num=self.n_sample)
        kp_ref, kp_out = self.model.to_pixel_coordinates(
            matches, H_ref, W_ref, H_out, W_out
        )
        kp_ref = kp_ref.cpu().numpy()
        kp_out = kp_out.cpu().numpy()
        sampled_cert = sampled_cert.cpu().numpy()

        ref_in_fg = _points_in_mask(kp_ref, ref_fg_mask, self.fg_threshold)
        out_in_fg = _points_in_mask(kp_out, out_fg_mask, self.fg_threshold)
        both_in_fg = ref_in_fg & out_in_fg

        n_ref_in_fg = int(ref_in_fg.sum())
        n_both_in_fg = int(both_in_fg.sum())
        score = n_both_in_fg / n_ref_in_fg if n_ref_in_fg > 0 else 0.0

        return {
            "score": score,
            "extras": {
                "n_matches": int(kp_ref.shape[0]),
                "n_ref_in_fg": n_ref_in_fg,
                "n_both_in_fg": n_both_in_fg,
                "mean_certainty_fg": (
                    float(sampled_cert[both_in_fg].mean()) if n_both_in_fg > 0 else 0.0
                ),
                "n_sample": self.n_sample,
            },
        }

    @torch.inference_mode()
    def _masked_maxcos(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
    ) -> dict:
        """Dense ref-side certainty averaged over fg-ref cells.

        Skips `sample()` entirely — `match()` already produces a dense
        (H×W) certainty on the ref grid; sampling would just add noise.
        Under `symmetric=True` (roma_outdoor default), certainty is
        concatenated along width as [ref→out | out→ref]; we take the
        left half.
        """
        ref_rgb = ref_image.convert("RGB")
        out_rgb = out_image.convert("RGB")

        _warp, certainty = self.model.match(ref_rgb, out_rgb, device=self.device)
        # certainty: (1, hs, 2*ws) in symmetric mode; left half = ref→out.
        hs = certainty.shape[1]
        total_w = certainty.shape[2]
        if total_w % 2 != 0:
            raise RuntimeError(
                f"Expected symmetric RoMa certainty (even width); got {total_w}"
            )
        ws = total_w // 2
        cert_ref = certainty[0, :, :ws]  # (hs, ws), already sigmoid-normalized ∈ [0, 1]

        fg = self._downsample_mask(ref_fg_mask, (hs, ws), cert_ref.device)
        fg_bool = fg > self.fg_threshold

        n_fg = int(fg_bool.sum())
        if n_fg == 0:
            return {
                "score": 0.0,
                "extras": {"n_fg_ref": 0, "resolution": [hs, ws]},
            }

        score = float(cert_ref[fg_bool].mean())
        return {
            "score": score,
            "extras": {
                "n_fg_ref": n_fg,
                "mean_certainty_global": float(cert_ref.mean()),
                "resolution": [hs, ws],
            },
        }

    @staticmethod
    def _downsample_mask(mask_np: np.ndarray, target_hw: tuple[int, int],
                         device: torch.device) -> torch.Tensor:
        mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0)[None, None]
        resized = F.interpolate(mask_t, size=target_hw, mode="bilinear",
                                align_corners=False)[0, 0]
        return resized.to(device)


def _points_in_mask(pts_xy: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Pixel-level fg lookup. `pts_xy` are (x, y) in `mask`'s own frame."""
    if pts_xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    H, W = mask.shape[:2]
    xs = np.clip(pts_xy[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(pts_xy[:, 1].astype(np.int64), 0, H - 1)
    return (mask[ys, xs] / 255.0) > threshold


def load_matcher(
    n_sample: int = 5000,
    fg_threshold: float = 0.5,
    score_type: str = "recall_fg",
    device: str = "cuda",
) -> RoMaMatcher:
    # `use_custom_corr=False` routes through the native-torch local
    # correlation fallback. The CUDA-fused kernel is the RoMa README's
    # optional `fused-local-corr` extra; we don't need it for eval.
    model = roma_outdoor(device=device, use_custom_corr=False)
    return RoMaMatcher(
        model=model,
        device=device,
        n_sample=n_sample,
        fg_threshold=fg_threshold,
        score_type=score_type,
    )
