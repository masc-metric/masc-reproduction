"""Geometry via MASt3R dense matching (3D-aware).

Installed per upstream README from `third_party/mast3r` (git submodule
with nested `dust3r`/`croco` submodules, plus its pip requirements).
MASt3R isn't packaged for pip install, so we prepend its root to
sys.path and rely on its internal `path_to_dust3r` / `path_to_croco`
injectors for the nested submodules.

Two score paths, selected by `score_type`:

`score_type="recall_fg"` (default) — mutual-NN foreground-recall semantic. `fast_reciprocal_NNs`
mutual-NN over dense descriptors, pixel-level fg lookup on both endpoints:

    score = n_both_in_fg / n_ref_in_fg

`score_type="masked_maxcos"` — mask-aware, no mutuality. L2-normalize
the dense descriptor grid, compute ref × out cosine, take row-wise max
per ref descriptor, average over fg-ref descriptors:

    score = mean(max_j cos(desc_ref[i], desc_out[j])  for fg-ref i)

Mirrors `_patch_match.py`'s `masked_maxcos` but on MASt3R's learned
descriptors instead of backbone patch tokens. Mask lookup happens at
the descriptor grid (input crop H/patch_size × W/patch_size).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _import_mast3r():
    """Lazy import: MASt3R isn't pip-installable, so we only resolve it
    when `load_matcher` is actually called. Reviewers who don't run the
    MASt3R ablation can import this module without cloning the upstream
    repo.
    """
    mast3r_root = Path(__file__).resolve().parent.parent.parent / "third_party" / "mast3r"
    if not mast3r_root.exists():
        raise RuntimeError(
            f"MASt3R not found at {mast3r_root}. Clone the upstream repo "
            f"(with its dust3r/croco submodules) into third_party/mast3r/. "
            f"See https://github.com/naver/mast3r for setup."
        )
    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r  # noqa: F401  side-effect: adds dust3r to sys.path
    from dust3r.inference import inference
    from dust3r.utils.image import ImgNorm, _resize_pil_image
    return {
        "AsymmetricMASt3R": AsymmetricMASt3R,
        "fast_reciprocal_NNs": fast_reciprocal_NNs,
        "inference": inference,
        "ImgNorm": ImgNorm,
        "_resize_pil_image": _resize_pil_image,
    }


# Populated by `load_matcher` on first call.
_MAST3R_API: dict = {}


@dataclass
class MASt3RMatcher:
    model: torch.nn.Module
    device: str
    input_size: int  # longest side target (before patch-size cropping)
    patch_size: int
    subsample_or_initxy1: int
    border: int  # ignore matches this close to the edges
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
        view_ref, mask_ref = self._prepare_view(ref_image, ref_fg_mask, idx=0)
        view_out, mask_out = self._prepare_view(out_image, out_fg_mask, idx=1)

        desc_ref, desc_out = self._forward_descs(view_ref, view_out)

        matches_ref, matches_out = _MAST3R_API["fast_reciprocal_NNs"](
            desc_ref, desc_out,
            subsample_or_initxy1=self.subsample_or_initxy1,
            device=self.device, dist="dot", block_size=2**13,
        )

        H_r, W_r = view_ref["true_shape"][0]
        H_o, W_o = view_out["true_shape"][0]

        valid = (
            (matches_ref[:, 0] >= self.border) & (matches_ref[:, 0] < int(W_r) - self.border) &
            (matches_ref[:, 1] >= self.border) & (matches_ref[:, 1] < int(H_r) - self.border) &
            (matches_out[:, 0] >= self.border) & (matches_out[:, 0] < int(W_o) - self.border) &
            (matches_out[:, 1] >= self.border) & (matches_out[:, 1] < int(H_o) - self.border)
        )
        matches_ref = matches_ref[valid]
        matches_out = matches_out[valid]

        ref_in_fg = _points_in_mask(matches_ref, mask_ref, self.fg_threshold)
        out_in_fg = _points_in_mask(matches_out, mask_out, self.fg_threshold)
        both_in_fg = ref_in_fg & out_in_fg

        n_ref_in_fg = int(ref_in_fg.sum())
        n_both_in_fg = int(both_in_fg.sum())
        score = n_both_in_fg / n_ref_in_fg if n_ref_in_fg > 0 else 0.0

        return {
            "score": score,
            "extras": {
                "n_matches": int(matches_ref.shape[0]),
                "n_ref_in_fg": n_ref_in_fg,
                "n_both_in_fg": n_both_in_fg,
                "subsample": self.subsample_or_initxy1,
                "border": self.border,
            },
        }

    @torch.inference_mode()
    def _masked_maxcos(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
    ) -> dict:
        """Dense descriptor cosine, masked on ref side, no mutuality.

        MASt3R's descriptor head is per-pixel (DPT-style, full
        cropped-image resolution). A full N_pixel × N_pixel cosine
        matrix would be ~150 GB at 512 input; we subsample both sides
        by `subsample_or_initxy1` (the same stride the recall_fg path
        uses internally via `fast_reciprocal_NNs`), so the matmul fits
        trivially and ref-side and recall-side see the same grid.
        """
        view_ref, mask_ref = self._prepare_view(ref_image, ref_fg_mask, idx=0)
        view_out, _ = self._prepare_view(out_image, ref_fg_mask, idx=1)  # out mask unused

        desc_ref, desc_out = self._forward_descs(view_ref, view_out)
        s = self.subsample_or_initxy1
        # desc shape: (H, W, D). Take every s-th pixel on both axes.
        dref_sub = desc_ref[::s, ::s, :]     # (H_r/s, W_r/s, D)
        dout_sub = desc_out[::s, ::s, :]     # (H_o/s, W_o/s, D)
        H_r, W_r, D = dref_sub.shape
        H_o, W_o, _ = dout_sub.shape

        dref = F.normalize(dref_sub.reshape(-1, D).float(), dim=-1)
        dout = F.normalize(dout_sub.reshape(-1, D).float(), dim=-1)
        cos = dref @ dout.T                  # (H_r*W_r, H_o*W_o)
        maxcos = cos.max(dim=1).values       # (H_r*W_r,)
        maxcos_grid = maxcos.reshape(H_r, W_r)

        fg = self._downsample_mask(mask_ref, (H_r, W_r), maxcos_grid.device)
        fg_bool = fg > self.fg_threshold

        n_fg = int(fg_bool.sum())
        if n_fg == 0:
            return {
                "score": 0.0,
                "extras": {"n_fg_ref": 0, "desc_grid": [int(H_r), int(W_r)],
                           "subsample": s},
            }

        score = float(maxcos_grid[fg_bool].mean())
        return {
            "score": score,
            "extras": {
                "n_fg_ref": n_fg,
                "mean_maxcos_global": float(maxcos_grid.mean()),
                "desc_grid": [int(H_r), int(W_r)],
                "subsample": s,
            },
        }

    def _forward_descs(self, view_ref: dict, view_out: dict) -> tuple[torch.Tensor, torch.Tensor]:
        output = _MAST3R_API["inference"](
            [(view_ref, view_out)], self.model, self.device, batch_size=1, verbose=False
        )
        desc_ref = output["pred1"]["desc"].squeeze(0).detach()
        desc_out = output["pred2"]["desc"].squeeze(0).detach()
        return desc_ref, desc_out

    def _prepare_view(
        self, pil_image: Image.Image, fg_mask: np.ndarray, idx: int
    ) -> tuple[dict, np.ndarray]:
        """Replicate `dust3r.utils.image.load_images` transform on a PIL image and
        apply the matching resize+crop to the fg mask so it stays pixel-aligned."""
        img = pil_image.convert("RGB")
        mask_pil = Image.fromarray(fg_mask)

        img_r = _MAST3R_API["_resize_pil_image"](img, self.input_size)
        mask_r = mask_pil.resize(img_r.size, Image.NEAREST)  # binary-preserving

        W, H = img_r.size
        cx, cy = W // 2, H // 2
        halfw = ((2 * cx) // self.patch_size) * self.patch_size / 2
        halfh = ((2 * cy) // self.patch_size) * self.patch_size / 2
        if W == H:  # mirrors dust3r.load_images branch
            halfh = 3 * halfw / 4
        box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
        img_c = img_r.crop(box)
        mask_c = mask_r.crop(box)

        view = {
            "img": _MAST3R_API["ImgNorm"](img_c)[None].to(self.device),
            "true_shape": np.int32([img_c.size[::-1]]),  # (H, W)
            "idx": idx,
            "instance": str(idx),
        }
        return view, np.asarray(mask_c)

    @staticmethod
    def _downsample_mask(mask_np: np.ndarray, target_hw: tuple[int, int],
                         device: torch.device) -> torch.Tensor:
        mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0)[None, None]
        resized = F.interpolate(mask_t, size=target_hw, mode="bilinear",
                                align_corners=False)[0, 0]
        return resized.to(device)


def _points_in_mask(pts_xy, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Pixel-level fg lookup. `pts_xy` are (x, y) in `mask`'s own frame."""
    pts = np.asarray(pts_xy)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    H, W = mask.shape[:2]
    xs = np.clip(pts[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(pts[:, 1].astype(np.int64), 0, H - 1)
    return (mask[ys, xs] / 255.0) > threshold


def load_matcher(
    checkpoint: str = "third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    input_size: int = 512,
    patch_size: int = 16,
    subsample_or_initxy1: int = 8,
    border: int = 3,
    fg_threshold: float = 0.5,
    score_type: str = "recall_fg",
    device: str = "cuda",
) -> MASt3RMatcher:
    if not _MAST3R_API:
        _MAST3R_API.update(_import_mast3r())
    model = _MAST3R_API["AsymmetricMASt3R"].from_pretrained(checkpoint).to(device)
    return MASt3RMatcher(
        model=model,
        device=device,
        input_size=input_size,
        patch_size=patch_size,
        subsample_or_initxy1=subsample_or_initxy1,
        border=border,
        fg_threshold=fg_threshold,
        score_type=score_type,
    )
