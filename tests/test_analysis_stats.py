"""Appendix statistics: α, ρ, AUC and the paired bootstrap (CPU only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repro_eval.analysis.stats import (  # noqa: E402
    HUMAN, RatedTable, alpha_interval, auc, bootstrap_auc, bootstrap_rated, diff_summary,
    kd_alpha, spearman, table_rng)


def _alpha_coincidence(x, y):
    """Textbook interval α from the coincidence matrix (O(V^2)), two raters."""
    vals = np.concatenate([x, y])
    uniq = np.unique(vals)
    pos = {v: i for i, v in enumerate(uniq)}
    o = np.zeros((len(uniq), len(uniq)))
    for a, b in zip(x, y):
        o[pos[a], pos[b]] += 1
        o[pos[b], pos[a]] += 1
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = (uniq[:, None] - uniq[None, :]) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return 1 - d_o / d_e


def test_alpha_matches_coincidence_matrix():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 200).astype(float)
    y = np.clip(x + rng.integers(-1, 2, 200), 0, 4).astype(float)
    assert alpha_interval(x, y) == pytest.approx(_alpha_coincidence(x, y), abs=1e-12)


def test_alpha_matches_release_formula():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plot_cp", Path(__file__).resolve().parent.parent / "scripts" / "plot_dreambenchplus_cp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(1)
    row = rng.normal(size=300)
    g1 = rng.integers(0, 5, 300).astype(float)
    g2 = rng.integers(0, 5, 300).astype(float)
    ours = kd_alpha(row, g1, g2, ordinal=False, paper_round=True)
    paper = mod.alpha_cell(row, g1, g2, ordinal=False, row_range=(row.min(), row.max()))
    assert float(ours) == paper
    ours_o = kd_alpha(g1, g1, g2, ordinal=True, paper_round=True)
    paper_o = mod.alpha_cell(g1, g1, g2, ordinal=True, row_range=(0.0, 4.0))
    assert float(ours_o) == paper_o


def test_alpha_rowwise_equals_loop():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 50))
    g1 = rng.integers(0, 5, (5, 50)).astype(float)
    g2 = rng.integers(0, 5, (5, 50)).astype(float)
    batched = kd_alpha(X, g1, g2, ordinal=False)
    looped = [kd_alpha(X[i], g1[i], g2[i], ordinal=False) for i in range(5)]
    np.testing.assert_allclose(batched, looped)


def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    x = rng.integers(0, 6, 400).astype(float)
    y = (x + rng.integers(0, 3, 400)) / 2
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)


def test_auc_matches_pairwise_count_with_ties():
    rng = np.random.default_rng(4)
    w = rng.integers(0, 5, 100).astype(float)
    c = rng.integers(0, 4, 120).astype(float)
    diff = w[:, None] - c[None, :]
    ref = ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size
    assert auc(w, c) == pytest.approx(ref, abs=1e-12)


def _toy_table(n=400, seed=5):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    g1 = np.clip(np.round(2 + truth + rng.normal(scale=0.7, size=n)), 0, 4)
    g2 = np.clip(np.round(2 + truth + rng.normal(scale=0.7, size=n)), 0, 4)
    rows = {"good": truth + rng.normal(scale=0.3, size=n),
            "weak": truth + rng.normal(scale=2.0, size=n),
            "rubric": np.clip(np.round(2 + truth), 0, 4)}
    return RatedTable(rows=rows, ordinal={"good": False, "weak": False, "rubric": True},
                      g1=g1, g2=g2, clusters=np.arange(n) // 8)


def test_bootstrap_is_deterministic_and_paired():
    t = _toy_table()
    a = bootstrap_rated(t, "toy", B=300, chunk=64)
    b = bootstrap_rated(t, "toy", B=300, chunk=64)
    np.testing.assert_array_equal(a["good"]["alpha"].reps, b["good"]["alpha"].reps)
    d = diff_summary(a["good"]["alpha"], a["weak"]["alpha"])
    assert d["excludes_zero"] and d["diff_point"] > 0
    lo, hi = a["good"]["alpha"].ci()
    assert lo < a["good"]["alpha"].point < hi
    assert HUMAN in a
    # A different table id draws different resamples.
    c = bootstrap_rated(t, "toy-2", B=300, chunk=64)
    assert not np.array_equal(a["good"]["alpha"].reps, c["good"]["alpha"].reps)


def test_cluster_bootstrap_is_wider_for_clustered_noise():
    rng = np.random.default_rng(6)
    n, k = 600, 30
    cl = np.repeat(np.arange(k), n // k)
    shared = rng.normal(size=k)[cl]  # cluster-level noise in the metric
    truth = rng.normal(size=n)
    g1 = np.clip(np.round(2 + truth + rng.normal(scale=0.5, size=n)), 0, 4)
    g2 = np.clip(np.round(2 + truth + rng.normal(scale=0.5, size=n)), 0, 4)
    t = RatedTable(rows={"m": truth + 1.5 * shared}, ordinal={"m": False}, g1=g1, g2=g2,
                   clusters=cl, human_row=False)
    keys = bootstrap_rated(t, "clu", B=400)
    clus = bootstrap_rated(t, "clu", B=400, cluster=True)
    assert np.std(clus["m"]["rho"].reps) > np.std(keys["m"]["rho"].reps)


def test_bootstrap_auc_strata_and_point():
    rng = np.random.default_rng(7)
    w = rng.normal(1.0, 1.0, 200)
    c = rng.normal(0.0, 1.0, 200)
    res = bootstrap_auc({"a": (w, c), "b": (w + rng.normal(0, 3, 200), c)}, "auc-toy", B=300)
    assert res["a"].point == pytest.approx(auc(w, c))
    assert diff_summary(res["a"], res["b"])["diff_point"] > 0


def test_table_rng_stable():
    assert table_rng("x").integers(0, 1_000_000) == table_rng("x").integers(0, 1_000_000)
