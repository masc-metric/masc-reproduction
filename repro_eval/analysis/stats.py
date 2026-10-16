"""Agreement statistics and the paired bootstrap of the appendix analyses.

All statistics work on the last axis, so a (B, n) matrix of resampled rows
gives B values at once.

- `alpha_interval`: two-rater interval Krippendorff's α, closed form,
  identical to `plot_dreambenchplus_cp.alpha_interval_arr`.
- `kd_alpha`: the paper's Kd_o, the mean of α(row, g1) and α(row, g2).
  Continuous rows are min-max normalised to [0, 1] on the rows being
  evaluated (so inside a bootstrap replicate, on that replicate's rows) and
  humans are divided by 4; rows on the 0-4 rubric (GPT judges, human groups)
  are used raw. `paper_round=True` reproduces the paper's 3-decimal rounding
  of each α and of their mean; the bootstrap never rounds.
- `spearman`: Spearman ρ with average ranks for ties.
- `auc`: P(within > cross), ties counted 1/2 (Mann-Whitney).

Bootstrap: resample matched keys with replacement, one index matrix per
table shared by every metric of the table, so differences are paired.
Percentile 95% intervals. The subject-cluster variant resamples subjects and
keeps all their keys. Every table seeds its own generator from a fixed root
and the table's label (`table_rng`), so each interval is reproducible;
changing a label or the root changes that table's intervals.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

SEED_ROOT = 20261106  # fixed; changing it changes every interval
DEFAULT_B = 10_000


def table_rng(table_id: str) -> np.random.Generator:
    """Per-table generator: SeedSequence([20261106, crc32(table_id)])."""
    return np.random.default_rng(np.random.SeedSequence([SEED_ROOT, zlib.crc32(table_id.encode())]))


def alpha_interval(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interval Krippendorff's α for two raters, along the last axis.

    D_o = mean((x - y)^2);  D_e = 2 M Var(x ∪ y) / (M - 1), M = 2N;  α = 1 - D_o / D_e.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[-1]
    d_o = np.mean((x - y) ** 2, axis=-1)
    v = np.concatenate([x, y], axis=-1)
    m = 2 * n
    d_e = 2.0 * m * np.var(v, axis=-1) / (m - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d_e > 0, 1.0 - d_o / d_e, np.nan)


def kd_alpha(
    row: np.ndarray, g1: np.ndarray, g2: np.ndarray, *, ordinal: bool, paper_round: bool = False
) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    if ordinal:
        x, a, b = row, g1, g2
    else:
        lo = row.min(axis=-1, keepdims=True)
        hi = row.max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (row - lo) / (hi - lo)
        a, b = g1 / 4.0, g2 / 4.0
    a1 = alpha_interval(x, a)
    a2 = alpha_interval(x, b)
    if paper_round:
        # Python's round(), as in the paper scripts (np.round can differ at
        # half-way cases because it scales by 10^3 first).
        r = np.vectorize(lambda u, v: round((round(float(u), 3) + round(float(v), 3)) / 2.0, 3))
        return r(a1, a2)
    return (a1 + a2) / 2.0


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _pearson_rows(rankdata(x, axis=-1), rankdata(y, axis=-1))


def auc(within: np.ndarray, cross: np.ndarray) -> np.ndarray:
    within = np.asarray(within, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    n_w, n_c = within.shape[-1], cross.shape[-1]
    ranks = rankdata(np.concatenate([within, cross], axis=-1), axis=-1)
    r_w = ranks[..., :n_w].sum(axis=-1)
    return (r_w - n_w * (n_w + 1) / 2.0) / (n_w * n_c)


# --------------------------------------------------------------------------
# Bootstrap containers


@dataclass
class Dist:
    """A statistic: full-sample point estimate plus its bootstrap replicates."""

    point: float
    reps: np.ndarray
    point_paper: float | None = None

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        q = (1 - level) / 2 * 100
        lo, hi = np.nanpercentile(self.reps, [q, 100 - q])
        return float(lo), float(hi)

    def summary(self) -> dict:
        lo, hi = self.ci()
        out = {
            "point": self.point,
            "ci95": [lo, hi],
            "se": float(np.nanstd(self.reps, ddof=1)),
            "n_nan_reps": int(np.isnan(self.reps).sum()),
        }
        if self.point_paper is not None:
            out["point_paper_rounded"] = self.point_paper
        return out


def diff_summary(a: Dist, b: Dist) -> dict:
    """Paired difference a - b over the shared replicates."""
    d = a.reps - b.reps
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    p_le = float(np.nanmean(d <= 0))
    p_ge = float(np.nanmean(d >= 0))
    return {
        "diff_point": a.point - b.point,
        "ci95": [float(lo), float(hi)],
        "se": float(np.nanstd(d, ddof=1)),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_boot_two_sided": float(min(1.0, 2 * min(p_le, p_ge))),
    }


@dataclass
class RatedTable:
    """Matched keys of one table: metric rows and both human rating groups."""

    rows: dict[str, np.ndarray]
    ordinal: dict[str, bool]
    g1: np.ndarray
    g2: np.ndarray
    clusters: np.ndarray | None = None  # e.g. concept_id per key (cluster bootstrap)
    human_row: bool = True  # add the human inter-rater ceiling row
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.g1)


HUMAN = "Human inter-rater"


def _stats_on(
    X: np.ndarray, G1: np.ndarray, G2: np.ndarray, ordinal: bool, Hrank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = kd_alpha(X, G1, G2, ordinal=ordinal)
    r = _pearson_rows(rankdata(X, axis=-1), Hrank)
    return a, r


def bootstrap_rated(
    t: RatedTable, table_id: str, B: int = DEFAULT_B, chunk: int = 250, cluster: bool = False
) -> dict[str, dict[str, Dist]]:
    """α and ρ for every row of `t`, with B paired replicates.

    `cluster=False`: resample keys. `cluster=True`: resample the clusters in
    `t.clusters` and keep all their keys.
    Returns {row: {"alpha": Dist, "rho": Dist}}, plus the human ceiling row
    (α(g1, g2) on raw ratings and ρ(g1, g2)) when `t.human_row`.
    """
    g1, g2 = t.g1.astype(np.float64), t.g2.astype(np.float64)
    h = (g1 + g2) / 2.0
    names = list(t.rows)
    res: dict[str, dict[str, Dist]] = {}
    for name in names:
        x = t.rows[name].astype(np.float64)
        a_paper = float(kd_alpha(x, g1, g2, ordinal=t.ordinal[name], paper_round=True))
        res[name] = {
            "alpha": Dist(
                float(kd_alpha(x, g1, g2, ordinal=t.ordinal[name])), np.empty(B), a_paper
            ),
            "rho": Dist(float(spearman(x, h)), np.empty(B)),
        }
    if t.human_row:
        res[HUMAN] = {
            "alpha": Dist(
                float(alpha_interval(g1, g2)), np.empty(B), round(float(alpha_interval(g1, g2)), 3)
            ),
            "rho": Dist(float(spearman(g1, g2)), np.empty(B)),
        }
    rng = table_rng(table_id + ("|cluster" if cluster else ""))
    if not cluster:
        for s in range(0, B, chunk):
            b = min(chunk, B - s)
            idx = rng.integers(0, t.n, size=(b, t.n))
            _fill(res, t, names, idx, slice(s, s + b), g1, g2)
    else:
        if t.clusters is None:
            raise ValueError("cluster bootstrap needs t.clusters")
        uniq, inv = np.unique(t.clusters, return_inverse=True)
        members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
        for i in range(B):
            draw = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[c] for c in draw])[None, :]
            _fill(res, t, names, idx, slice(i, i + 1), g1, g2)
    return res


def _fill(res, t, names, idx, sl, g1, g2):
    G1, G2 = g1[idx], g2[idx]
    Hrank = rankdata((G1 + G2) / 2.0, axis=-1)
    for name in names:
        a, r = _stats_on(t.rows[name][idx], G1, G2, t.ordinal[name], Hrank)
        res[name]["alpha"].reps[sl] = a
        res[name]["rho"].reps[sl] = r
    if HUMAN in res:
        res[HUMAN]["alpha"].reps[sl] = alpha_interval(G1, G2)
        res[HUMAN]["rho"].reps[sl] = _pearson_rows(rankdata(G1, axis=-1), rankdata(G2, axis=-1))


def bootstrap_auc(
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    table_id: str,
    B: int = DEFAULT_B,
    chunk: int = 250,
    clusters: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Dist]:
    """AUC(within > cross) per metric. Within and cross pairs are resampled
    separately with index matrices shared by all metrics (paired).
    `clusters=(within_subject, cross_anchor_subject)` switches to the
    subject-cluster sensitivity bootstrap."""
    names = list(scores)
    n_w = len(next(iter(scores.values()))[0])
    n_c = len(next(iter(scores.values()))[1])
    res = {k: Dist(float(auc(w, c)), np.empty(B)) for k, (w, c) in scores.items()}
    rng = table_rng(table_id + ("|cluster" if clusters is not None else ""))
    if clusters is None:
        for s in range(0, B, chunk):
            b = min(chunk, B - s)
            iw = rng.integers(0, n_w, size=(b, n_w))
            ic = rng.integers(0, n_c, size=(b, n_c))
            for k in names:
                w, c = scores[k]
                res[k].reps[s : s + b] = auc(w[iw], c[ic])
    else:
        sw, sc = clusters
        subj = np.unique(np.concatenate([sw, sc]))
        mw = {u: np.flatnonzero(sw == u) for u in subj}
        mc = {u: np.flatnonzero(sc == u) for u in subj}
        for i in range(B):
            draw = rng.integers(0, len(subj), size=len(subj))
            iw = np.concatenate([mw[subj[d]] for d in draw])
            ic = np.concatenate([mc[subj[d]] for d in draw])
            for k in names:
                w, c = scores[k]
                res[k].reps[i] = auc(w[iw], c[ic]) if len(iw) and len(ic) else np.nan
    return res
