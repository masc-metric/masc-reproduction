"""Geometry via LoFTR detector-free dense matching.

Uses kornia's integrated LoFTR (the upstream LoFTR README's recommended
install path). Outdoor-pretrained weights by default — closer to
DreamBooth's typical scenes than ScanNet-trained indoor weights.

Two score paths, selected by `score_type`:

`score_type="recall_fg"` (default) — mutual-NN foreground-recall semantic. LoFTR emits
confidence-filtered sparse matches; pixel-level fg lookup on both
endpoints:

    score = n_both_in_fg / n_ref_in_fg

`score_type="masked_maxcos"` — mask-aware, uses the coarse-level
assignment matrix directly (no thresholding, no sparse NN extraction).
kornia's LoFTR populates a (1, L, S) dual-softmax `conf_matrix` inside
`coarse_matching`; we intercept it via a forward hook. Per ref coarse
cell we take the max assignment probability over all out coarse cells
— "best-match confidence for this ref cell" — then mean under
fg-mask-at-coarse-grid:

    score = mean(max_j conf_matrix[i, j]  for fg-ref coarse cells i)

Grid resolution is input_size/8 (LoFTR's coarse stride). Semantically
mirrors `masked_maxcos` on patch cosine, with the per-cell "score"
being LoFTR's trained assignment prob rather than a raw cosine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from kornia.feature import LoFTR
from PIL import Image


@dataclass
class LoFTRMatcher:
    model: torch.nn.Module
    device: str
    input_size: int  # longest-edge resize target; both images forced to square
    fg_threshold: float = 0.5
    confidence_threshold: float = 0.2  # kornia LoFTR default-ish
    score_type: str = "recall_fg"  # or "masked_maxcos"
    _hook_cache: dict = field(default_factory=dict)
    _hook_handle: object = None

    def __post_init__(self) -> None:
        # Hook `coarse_matching` to capture conf_matrix. kornia's LoFTR
        # calls `self.coarse_matching(feat_c0, feat_c1, _data, ...)`
        # and the module mutates `_data` in place; we grab _data here.
        if self.score_type == "masked_maxcos" and self._hook_handle is None:
            cache = self._hook_cache
            def hook(_module, inputs, _outputs):
                # inputs == (feat_c0, feat_c1, data_dict); data_dict now has 'conf_matrix'
                data_dict = inputs[2]
                if "conf_matrix" in data_dict:
                    cache["conf_matrix"] = data_dict["conf_matrix"].detach()
                    cache["hw0_c"] = tuple(data_dict["hw0_c"])
                    cache["hw1_c"] = tuple(data_dict["hw1_c"])
            self._hook_handle = self.model.coarse_matching.register_forward_hook(hook)

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
        ref_t, ref_mask_np = _to_gray_and_mask(
            ref_image, ref_fg_mask, self.input_size, self.device
        )
        out_t, out_mask_np = _to_gray_and_mask(
            out_image, out_fg_mask, self.input_size, self.device
        )

        res = self.model({"image0": ref_t, "image1": out_t})
        mkpts0 = res["keypoints0"].cpu().numpy()
        mkpts1 = res["keypoints1"].cpu().numpy()
        conf = res["confidence"].cpu().numpy()

        keep = conf >= self.confidence_threshold
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        conf = conf[keep]

        ref_in_fg = _points_in_mask(mkpts0, ref_mask_np, self.fg_threshold)
        out_in_fg = _points_in_mask(mkpts1, out_mask_np, self.fg_threshold)
        both_in_fg = ref_in_fg & out_in_fg

        n_ref_in_fg = int(ref_in_fg.sum())
        n_both_in_fg = int(both_in_fg.sum())
        score = n_both_in_fg / n_ref_in_fg if n_ref_in_fg > 0 else 0.0

        return {
            "score": score,
            "extras": {
                "n_matches_raw": int(res["keypoints0"].shape[0]),
                "n_matches": int(mkpts0.shape[0]),
                "n_ref_in_fg": n_ref_in_fg,
                "n_both_in_fg": n_both_in_fg,
                "mean_confidence_fg": (
                    float(conf[both_in_fg].mean()) if n_both_in_fg > 0 else 0.0
                ),
                "input_size": self.input_size,
                "confidence_threshold": self.confidence_threshold,
            },
        }

    @torch.inference_mode()
    def _masked_maxcos(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
    ) -> dict:
        ref_t, ref_mask_np = _to_gray_and_mask(
            ref_image, ref_fg_mask, self.input_size, self.device
        )
        out_t, _ = _to_gray_and_mask(
            out_image, ref_fg_mask, self.input_size, self.device  # out mask unused
        )

        self._hook_cache.clear()
        _ = self.model({"image0": ref_t, "image1": out_t})

        if "conf_matrix" not in self._hook_cache:
            raise RuntimeError(
                "coarse_matching forward hook didn't fire; kornia LoFTR internals may have changed"
            )

        conf = self._hook_cache["conf_matrix"][0]     # (L, S) = (H0_c*W0_c, H1_c*W1_c)
        H_c, W_c = self._hook_cache["hw0_c"]
        max_per_ref = conf.max(dim=1).values          # (L,), row-wise best-match probability
        max_grid = max_per_ref.reshape(H_c, W_c)       # (H_c, W_c) at input_size/8

        fg = self._downsample_mask(ref_mask_np, (H_c, W_c), max_grid.device)
        fg_bool = fg > self.fg_threshold

        n_fg = int(fg_bool.sum())
        if n_fg == 0:
            return {
                "score": 0.0,
                "extras": {"n_fg_ref": 0, "coarse_grid": [int(H_c), int(W_c)]},
            }

        score = float(max_grid[fg_bool].mean())
        return {
            "score": score,
            "extras": {
                "n_fg_ref": n_fg,
                "mean_maxprob_global": float(max_grid.mean()),
                "coarse_grid": [int(H_c), int(W_c)],
                "input_size": self.input_size,
            },
        }

    @staticmethod
    def _downsample_mask(mask_np: np.ndarray, target_hw: tuple[int, int],
                         device: torch.device) -> torch.Tensor:
        mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0)[None, None]
        resized = F.interpolate(mask_t, size=target_hw, mode="bilinear",
                                align_corners=False)[0, 0]
        return resized.to(device)


def _to_gray_and_mask(
    image: Image.Image, fg_mask: np.ndarray, input_size: int, device: str
) -> tuple[torch.Tensor, np.ndarray]:
    """Resize RGB image + fg mask to (input_size, input_size); return
    (gray tensor in [0,1], binary mask ndarray in {0, 255})."""
    gray = image.convert("L").resize((input_size, input_size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(device)

    mask_pil = Image.fromarray(fg_mask).resize(
        (input_size, input_size), Image.NEAREST
    )
    mask_np = np.asarray(mask_pil)
    return t, mask_np


def _points_in_mask(pts_xy: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    if pts_xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    H, W = mask.shape[:2]
    xs = np.clip(pts_xy[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(pts_xy[:, 1].astype(np.int64), 0, H - 1)
    return (mask[ys, xs] / 255.0) > threshold


def load_matcher(
    pretrained: str = "outdoor",
    input_size: int = 512,
    fg_threshold: float = 0.5,
    confidence_threshold: float = 0.2,
    score_type: str = "recall_fg",
    device: str = "cuda",
) -> LoFTRMatcher:
    if input_size % 8 != 0:
        raise ValueError(f"input_size={input_size} must be divisible by 8 (LoFTR coarse grid)")
    model = LoFTR(pretrained=pretrained).eval().to(device)
    return LoFTRMatcher(
        model=model,
        device=device,
        input_size=input_size,
        fg_threshold=fg_threshold,
        confidence_threshold=confidence_threshold,
        score_type=score_type,
    )
