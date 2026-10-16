"""Reference-mask perturbation and CP recomputation from stored patch arrays.

MaSC CP masks only the reference. For every (reference, output) pair a
scoring run with `--dump-patch-arrays` stores `nn_cos_per_ref`, each reference
patch's best cosine to any output patch, which does not depend on any mask.
CP under another reference mask M is then exactly

    mean(nn_cos_per_ref[downsample(M) > 0.5])     (0.0 if no patch qualifies)

with `downsample` the bilinear projection of `PatchFgMatcher._downsample_mask`.

Perturbation (`perturb`): p is a signed fraction, p > 0 dilates and p < 0
erodes, with a Euclidean disk of radius round(|p| * min(H, W)) pixels. The
mask is binarised at > 127 first; p = 0 returns the mask untouched. Erosion
treats pixels outside the image as background.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


def load_mask(path: Path) -> np.ndarray:
    """uint8 array exactly as compute_geometry.py passes it to the matcher."""
    return np.array(Image.open(path))


def binarise(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def perturb(mask: np.ndarray, p: float) -> np.ndarray:
    """Return a uint8 {0, 255} mask (or the input itself when p == 0)."""
    if p == 0:
        return mask
    b = binarise(mask)
    h, w = b.shape
    if not b.any():
        return np.zeros_like(b, dtype=np.uint8)
    r = round(abs(p) * min(h, w))
    if p > 0:
        out = b | (ndimage.distance_transform_edt(~b) <= r)
    else:
        # Pad with background so erosion also removes pixels near the border.
        inside = ndimage.distance_transform_edt(np.pad(b, 1, constant_values=False))[1:-1, 1:-1]
        out = b & (inside > r)
    return out.astype(np.uint8) * 255


def grid_weights(mask: np.ndarray, hp: int, wp: int) -> np.ndarray:
    """Identical to PatchFgMatcher._downsample_mask (no blur): bilinear,
    align_corners=False, clamped to [0, 1], on CPU in float32."""
    m = torch.from_numpy(np.asarray(mask).astype(np.float32) / 255.0)[None, None]
    m = F.interpolate(m, size=(hp, wp), mode="bilinear", align_corners=False)
    return m[0, 0].clamp(0.0, 1.0).numpy().reshape(-1)


def masked_maxcos(nn_cos: np.ndarray, weights: np.ndarray, threshold: float = 0.5) -> float:
    fg = weights > threshold
    return float(nn_cos[fg].mean()) if fg.any() else 0.0


def fg_fraction(mask: np.ndarray) -> float:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return float((m > 0).sum()) / float(m.size)


@lru_cache(maxsize=4096)
def _cached_mask(path: str) -> np.ndarray:
    return load_mask(Path(path))


@lru_cache(maxsize=65536)
def perturbed_grid(path: str, p: float, hp: int, wp: int) -> np.ndarray:
    """Patch-grid weights of a perturbed mask; cached because a DreamBench++
    reference mask is shared by all 63 pairs of its subject."""
    g = grid_weights(perturb(_cached_mask(path), p), hp, wp)
    g.setflags(write=False)
    return g
