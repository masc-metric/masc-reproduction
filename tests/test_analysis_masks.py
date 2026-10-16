"""Reference-mask perturbation, CP recomputation and the paper comparison,
on synthetic data (CPU only)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repro_eval.analysis import masks  # noqa: E402
from repro_eval.analysis.tables import PaperCheck  # noqa: E402
from repro_eval.geometry._patch_match import PatchFgMatcher  # noqa: E402


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.uint8) * 255


def test_perturb_zero_is_identity_and_signs():
    m = disk(200, 300, 100, 150, 40)
    assert masks.perturb(m, 0.0) is m
    a = masks.fg_fraction(m)
    assert masks.fg_fraction(masks.perturb(m, 0.05)) > a > masks.fg_fraction(masks.perturb(m, -0.05))


def test_radius_is_fraction_of_short_side():
    m = disk(200, 300, 100, 150, 40)
    d = masks.perturb(m, 0.05)  # r = round(0.05 * 200) = 10
    ys = np.flatnonzero(d[:, 150] > 0)
    assert ys.min() == 100 - 50 and ys.max() == 100 + 50
    e = masks.perturb(m, -0.05)
    ys = np.flatnonzero(e[:, 150] > 0)
    assert ys.min() == 100 - 30 and ys.max() == 100 + 30


def test_erosion_treats_image_border_as_background():
    m = np.zeros((50, 50), np.uint8)
    m[:, :25] = 255  # touches the left border
    e = masks.perturb(m, -0.1)  # r = 5
    assert e[:, :5].max() == 0


def test_empty_mask_stays_empty():
    assert masks.perturb(np.zeros((40, 40), np.uint8), 0.16).max() == 0


def test_grid_weights_match_the_matcher():
    # Only the attributes `_downsample_mask` reads; no backbone is loaded.
    matcher = SimpleNamespace(mask_blur_sigma=0.0, patch_size=16, device="cpu")
    m = disk(512, 384, 200, 190, 90)
    ref = PatchFgMatcher._downsample_mask(matcher, m, 32, 24).numpy().reshape(-1)
    np.testing.assert_array_equal(masks.grid_weights(m, 32, 24), ref)


def test_masked_maxcos_matches_direct_mean():
    rng = np.random.default_rng(0)
    nn = rng.random(32 * 32).astype(np.float32)
    m = disk(512, 512, 256, 256, 120)
    w = masks.grid_weights(m, 32, 32)
    assert masks.masked_maxcos(nn, w) == pytest.approx(float(nn[w > 0.5].mean()))
    assert masks.masked_maxcos(nn, np.zeros_like(w)) == 0.0


def test_paper_check_compares_at_printed_precision():
    c = PaperCheck()
    c.add("same", 0.47149, 0.471)
    c.add("rounds up", 0.4716, 0.472)
    c.add("count", 7304, 7304, 0)
    assert c.mismatches() == []
    c.add("off", 0.4726, 0.471)
    assert len(c.mismatches()) == 1 and c.report() == 1
