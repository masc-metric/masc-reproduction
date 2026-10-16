"""Run one bootstrapped table, write it, and compare it with the paper.

Every table writes `report/appendix/<name>.json` (point estimates, 95%
intervals, paired differences under the key bootstrap and the
subject-cluster bootstrap). Scripts print the table in the paper's layout and,
with `check_paper`, compare every printed number with the camera-ready paper
at its three-decimal rounding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from repro_eval.data import REPO_ROOT

from .stats import (
    DEFAULT_B,
    HUMAN,
    SEED_ROOT,
    RatedTable,
    bootstrap_auc,
    bootstrap_rated,
    diff_summary,
)

OUT_DIR = REPO_ROOT / "report" / "appendix"


@dataclass(frozen=True)
class Row:
    label: str
    column: str
    ordinal: bool = False  # on the 0-4 rubric (GPT judges): used raw, not min-max normalised
    kind: str = "non-LLM"  # "ours", "LLM judge", "non-LLM", "control"


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_json(name: str, obj: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n")
    return path


def rated_table(
    sub: pd.DataFrame,
    rows: list[Row],
    seed_label: str,
    *,
    g1: str,
    g2: str,
    comparisons: list[tuple[str, str]] = (),
    cluster_col: str | None = "subject",
    B: int = DEFAULT_B,
    human_row: bool = True,
) -> dict:
    """α and ρ of every row on the keys of `sub`, with paired 95% intervals.

    `seed_label` seeds the resampling (`stats.table_rng`); the appendix tables
    keep the labels the paper's intervals were computed with.
    `comparisons`: (a, b) row labels; a - b is reported for α and ρ.
    """
    t = RatedTable(
        rows={r.label: sub[r.column].to_numpy(float) for r in rows},
        ordinal={r.label: r.ordinal for r in rows},
        g1=sub[g1].to_numpy(float),
        g2=sub[g2].to_numpy(float),
        clusters=sub[cluster_col].to_numpy() if cluster_col else None,
        human_row=human_row,
    )
    for r in rows:
        if np.isnan(t.rows[r.label]).any():
            raise ValueError(f"{seed_label}: NaN in {r.label}; the key set must exclude them")
    kinds = {r.label: r.kind for r in rows} | ({HUMAN: "human ceiling"} if human_row else {})

    def pack(res):
        return {
            lab: {"alpha": d["alpha"].summary(), "rho": d["rho"].summary(), "kind": kinds[lab]}
            for lab, d in res.items()
        }

    def comps(res):
        return [
            {"a": a, "b": b, "stat": stat, **diff_summary(res[a][stat], res[b][stat])}
            for a, b in comparisons
            for stat in ("alpha", "rho")
        ]

    res = bootstrap_rated(t, seed_label, B=B)
    out = {
        "seed_label": seed_label,
        "n_keys": int(t.n),
        "B": B,
        "rng": f"numpy default_rng(SeedSequence([{SEED_ROOT}, crc32(seed_label)]))",
        "interval": "percentile 95%",
        "alpha": "Kd_o: mean of interval α against each rating group; continuous rows "
        "min-max normalised on the evaluated keys (in every replicate)",
        "rho": "Spearman vs the mean of both rating groups",
        "rows": pack(res),
        "comparisons": comps(res),
    }
    if cluster_col:
        res_c = bootstrap_rated(t, seed_label, B=B, cluster=True)
        out["cluster_bootstrap"] = {
            "unit": cluster_col,
            "n_clusters": int(len(np.unique(t.clusters))),
            "rows": pack(res_c),
            "comparisons": comps(res_c),
        }
    return out


def auc_table(
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    seed_label: str,
    *,
    comparisons: list[tuple[str, str]] = (),
    B: int = DEFAULT_B,
) -> dict:
    """AUC(within > cross) per metric; within and cross pairs are resampled
    separately with index matrices shared by all metrics (paired)."""
    res = bootstrap_auc(scores, seed_label, B=B)
    n_w, n_c = (len(v) for v in next(iter(scores.values())))
    return {
        "seed_label": seed_label,
        "n_within": n_w,
        "n_cross": n_c,
        "B": B,
        "rng": f"numpy default_rng(SeedSequence([{SEED_ROOT}, crc32(seed_label)]))",
        "interval": "percentile 95%; within and cross pairs resampled separately",
        "rows": {k: v.summary() for k, v in res.items()},
        "comparisons": [{"a": a, "b": b, **diff_summary(res[a], res[b])} for a, b in comparisons],
    }


def comparison(
    result: dict, a: str, b: str, stat: str | None = None, cluster: bool = False
) -> dict:
    src = result["cluster_bootstrap"] if cluster else result
    for c in src["comparisons"]:
        if c["a"] == a and c["b"] == b and (stat is None or c["stat"] == stat):
            return c
    raise KeyError((a, b, stat))


# --------------------------------------------------------------------------
# Printing and the comparison with the paper


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci3(lo_hi) -> str:
    return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def print_table(title: str, header: list[str], lines: list[list[str]]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *lines]) for i in range(len(header))]
    print(f"\n{title}")
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in lines:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


class PaperCheck:
    """Collects (name, ours, paper) cells and compares them at the paper's
    printed precision, or within an absolute tolerance where one is given.
    With `intervals=False` (a run with fewer bootstrap replicates than the
    paper's) confidence limits are not compared."""

    def __init__(self, intervals: bool = True) -> None:
        self.cells: list[tuple[str, float, float, int, float | None]] = []
        self.intervals = intervals

    def add(self, name: str, ours: float, paper: float, digits: int = 3,
            tol: float | None = None) -> None:
        self.cells.append((name, float(ours), float(paper), digits, tol))

    def add_ci(self, name: str, ours: list[float], paper: tuple[float, float]) -> None:
        if not self.intervals:
            return
        self.add(f"{name} lo", ours[0], paper[0])
        self.add(f"{name} hi", ours[1], paper[1])

    @staticmethod
    def _ok(o: float, p: float, d: int, tol: float | None) -> bool:
        if tol is not None:
            return abs(o - p) <= tol + 10 ** -(d + 3)
        return f"{o:.{d}f}" == f"{p:.{d}f}"

    def mismatches(self) -> list[str]:
        return [
            f"{n}: ours {o:.{d}f} ({o:.6f}), paper {p:.{d}f}"
            + (f", tolerance {tol}" if tol is not None else "")
            for n, o, p, d, tol in self.cells
            if not self._ok(o, p, d, tol)
        ]

    def within_tolerance(self) -> list[str]:
        """Cells checked with a tolerance whose rounded values differ."""
        return [
            f"{n}: ours {o:.{d}f}, paper {p:.{d}f}"
            for n, o, p, d, tol in self.cells
            if tol is not None and self._ok(o, p, d, tol) and f"{o:.{d}f}" != f"{p:.{d}f}"
        ]

    def report(self) -> int:
        bad = self.mismatches()
        print(
            f"\nPaper check: {len(self.cells) - len(bad)}/{len(self.cells)} printed numbers reproduced"
        )
        if not self.intervals:
            print("  (confidence intervals not compared: fewer bootstrap replicates than the paper)")
        for line in bad:
            print(f"  MISMATCH {line}")
        for line in self.within_tolerance():
            print(f"  within tolerance: {line}")
        return 1 if bad else 0
