"""Geometry via SuperPoint + LightGlue sparse keypoint matching.

Sparse matcher: SuperPoint detects keypoints, LightGlue matches
descriptors. Mask lookup at native image resolution (fg masks are
passed in at each image's own pixel frame).

Score (unified with the rest of the geometry matchers):

    score = n_both_in_fg / n_ref_in_fg

where
    n_ref_in_fg  = # matched pairs whose ref-endpoint is in ref-fg,
    n_both_in_fg = # matched pairs with BOTH endpoints in fg.

Reads as "of matched pairs sourced in the ref concept region, what
fraction terminate in the out concept region."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from PIL import Image


@dataclass
class LightGlueMatcher:
    extractor: torch.nn.Module
    matcher: torch.nn.Module
    device: str
    fg_threshold: float = 0.5  # mask pixel value / 255 must exceed this

    @torch.inference_mode()
    def fg_match(
        self,
        ref_image: Image.Image,
        ref_fg_mask: np.ndarray,
        out_image: Image.Image,
        out_fg_mask: np.ndarray,
    ) -> dict:
        ref_t = _pil_to_tensor(ref_image).to(self.device)
        out_t = _pil_to_tensor(out_image).to(self.device)

        ref_feats = self.extractor.extract(ref_t)
        out_feats = self.extractor.extract(out_t)

        matches01 = self.matcher({"image0": ref_feats, "image1": out_feats})
        ref_feats_s, out_feats_s, matches01_s = (
            rbd(ref_feats), rbd(out_feats), rbd(matches01)
        )

        ref_kp = ref_feats_s["keypoints"].cpu().numpy()  # (Nr, 2) xy
        out_kp = out_feats_s["keypoints"].cpu().numpy()  # (No, 2) xy
        matches = matches01_s["matches"].cpu().numpy()  # (K, 2) indices
        match_scores = matches01_s.get("scores")

        ref_kp_in_fg_all = _kps_in_mask(ref_kp, ref_fg_mask, self.fg_threshold)
        out_kp_in_fg_all = _kps_in_mask(out_kp, out_fg_mask, self.fg_threshold)

        if matches.shape[0] == 0:
            return {
                "score": 0.0,
                "extras": {
                    "n_kp_ref": int(ref_kp.shape[0]),
                    "n_kp_out": int(out_kp.shape[0]),
                    "n_matches": 0,
                    "n_ref_in_fg": 0,
                    "n_both_in_fg": 0,
                    "mean_match_score_fg": 0.0,
                },
            }

        ref_match_in_fg = ref_kp_in_fg_all[matches[:, 0]]
        out_match_in_fg = out_kp_in_fg_all[matches[:, 1]]
        both_in_fg = ref_match_in_fg & out_match_in_fg

        n_ref_in_fg = int(ref_match_in_fg.sum())
        n_both_in_fg = int(both_in_fg.sum())
        score = n_both_in_fg / n_ref_in_fg if n_ref_in_fg > 0 else 0.0

        if match_scores is not None and n_both_in_fg > 0:
            scores_np = match_scores.cpu().numpy()
            mean_match_score_fg = float(scores_np[both_in_fg].mean())
        else:
            mean_match_score_fg = 0.0

        return {
            "score": score,
            "extras": {
                "n_kp_ref": int(ref_kp.shape[0]),
                "n_kp_out": int(out_kp.shape[0]),
                "n_matches": int(matches.shape[0]),
                "n_ref_in_fg": n_ref_in_fg,
                "n_both_in_fg": n_both_in_fg,
                "mean_match_score_fg": mean_match_score_fg,
            },
        }


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL → CxHxW float tensor in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _kps_in_mask(kps_xy: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Bool vector: for each keypoint, is the corresponding mask pixel fg?

    `kps_xy` is in the coordinate frame of the image passed to the
    extractor; the mask must match that same (H, W) shape. This holds
    by construction — SuperPoint's extract() reports keypoints in the
    input image's own frame, and we pass the native-resolution image
    that the mask was generated for.
    """
    if kps_xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    H, W = mask.shape[:2]
    xs = np.clip(kps_xy[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(kps_xy[:, 1].astype(np.int64), 0, H - 1)
    return (mask[ys, xs] / 255.0) > threshold


def load_matcher(
    max_num_keypoints: int = 2048,
    fg_threshold: float = 0.5,
    device: str = "cuda",
) -> LightGlueMatcher:
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return LightGlueMatcher(
        extractor=extractor,
        matcher=matcher,
        device=device,
        fg_threshold=fg_threshold,
    )
