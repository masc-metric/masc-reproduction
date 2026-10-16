"""Appendix "Sensitivity to Mask-Area Filtering" (DreamBench++ CP).

    python scripts/appendix/mask_area_filtering.py [--B 10000]

Threshold sweep (first table): at threshold t a pair is kept when both SAM3
masks cover more than t of their image, so t = 0 removes exactly the pairs
with an empty mask. Empty-mask treatments (second table):
  valid reference   every pair whose reference mask has at least one
                    foreground patch on the SigLIP2 grid (CP is undefined
                    otherwise); no output-mask filter
  scale minimum     every rated pair; pairs with an empty reference or output
                    mask get the bottom of each metric's scale: 0 on the 0-4
                    judges, the metric's minimum over the other pairs otherwise
  raw zero          every rated pair; empty-mask pairs get a raw score of 0
                    before min-max normalisation
Within a condition every metric is evaluated on the same keys. MaSC minus
DreamSim is reported with paired key-bootstrap and subject-cluster intervals.
The filtering audit counts the pairs the default 5% filter excludes and their
human ratings. Reads results/analysis/dbpp_keys.csv.gz; writes
report/appendix/mask_area_filtering.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repro_eval.analysis.keytable import read_keys, retained  # noqa: E402
from repro_eval.analysis.stats import DEFAULT_B  # noqa: E402
from repro_eval.analysis.tables import (  # noqa: E402
    PaperCheck,
    Row,
    ci3,
    comparison,
    f3,
    print_table,
    rated_table,
    write_json,
)

ROWS = [
    Row("GPT-4o", "gpt4o", True, "LLM judge"),
    Row("MaSC", "masc", False, "ours"),
    Row("GPT-4V", "gpt4v", True, "LLM judge"),
    Row("DreamSim", "dreamsim"),
    Row("SigLIP2 global pool", "siglip2_global"),
    Row("DINOv3", "dinov3"),
]
THRESHOLDS = {"0%": 0.0, "2.5%": 0.025, "5%": 0.05, "10%": 0.10}
# condition -> resampling seed label (the labels the paper's intervals used)
SEED = {
    "0%": "filter_t0",
    "2.5%": "filter_t2.5",
    "5%": "filter_t5",
    "10%": "filter_t10",
    "valid reference": "filter_nofilter_rebuttal",
    "scale minimum": "filter_dropnothing_rebuttal",
    "raw zero": "filter_dropnothing_cp0_emptyonly",
}

# Camera-ready values: keys, alpha per row (ROWS order), MaSC - DreamSim alpha [95% CI]
PAPER = {
    "0%": (7717, (0.516, 0.474, 0.447, 0.440, 0.381, 0.362), 0.034, (0.013, 0.055)),
    "2.5%": (7561, (0.507, 0.478, 0.437, 0.436, 0.374, 0.347), 0.041, (0.018, 0.059)),
    "5%": (7304, (0.499, 0.465, 0.426, 0.421, 0.368, 0.343), 0.044, (0.012, 0.063)),
    "10%": (6351, (0.487, 0.448, 0.407, 0.396, 0.359, 0.330), 0.052, (0.011, 0.074)),
    "valid reference": (8694, (0.613, 0.548, 0.547, 0.531, 0.451, 0.492), 0.017, (0.003, 0.034)),
    "scale minimum": (9441, (0.530, 0.498, 0.468, 0.493, 0.471, 0.449), 0.005, (0.001, 0.011)),
    "raw zero": (9441, (0.530, 0.351, 0.468, 0.472, 0.390, 0.453), -0.122, (-0.133, -0.114)),
}
PAPER_AUDIT = {  # name -> (value, printed decimals)
    "reference-side exclusions": (882, 0),
    "output-side exclusions": (1264, 0),
    "mean human CP, retained": (2.24, 2),
    "mean human CP, all excluded": (1.41, 2),
    "mean human CP, output-side exclusions": (1.06, 2),
    "output-side exclusions with mean human CP >= 3, % of rated pairs": (1.35, 2),
    "Spearman(output-mask area, human CP), reference retained": (0.305, 3),
}


def condition(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """The pairs and scores of one condition, before matching on ratings."""
    cols = [r.column for r in ROWS]
    if name in THRESHOLDS:
        return df[retained(df, THRESHOLDS[name])].copy()
    if name == "valid reference":
        return df[df.masc_n_ref_fg_patches > 0].copy()
    d = df.copy()
    empty = ~retained(d, 0.0)
    if name == "scale minimum":
        for r in ROWS:
            d.loc[empty, r.column] = 0.0 if r.ordinal else float(d.loc[~empty, r.column].min())
    elif name == "raw zero":
        d.loc[empty, cols] = 0.0
    else:
        raise ValueError(name)
    return d


def matched(d: pd.DataFrame) -> pd.DataFrame:
    ok = d.cp_g1.notna() & d.cp_g2.notna()
    for r in ROWS:
        ok &= d[r.column].notna()
    return d[ok].reset_index(drop=True)


def audit(df: pd.DataFrame) -> dict:
    h = (df.cp_g1 + df.cp_g2) / 2
    rated = df.cp_g1.notna() & df.cp_g2.notna()
    keep = retained(df, 0.05)
    ref_drop = df.fg_ref <= 0.05
    out_drop = ~ref_drop & (df.fg_out <= 0.05)
    ref_ok = ~ref_drop
    return {
        "reference-side exclusions": int(ref_drop.sum()),
        "output-side exclusions": int(out_drop.sum()),
        "mean human CP, retained": float(h[keep & rated].mean()),
        "mean human CP, all excluded": float(h[~keep & rated].mean()),
        "mean human CP, output-side exclusions": float(h[out_drop & rated].mean()),
        "output-side exclusions with mean human CP >= 3, % of rated pairs": 100.0
        * float((out_drop & rated & (h >= 3)).sum() / rated.sum()),
        "Spearman(output-mask area, human CP), reference retained": float(
            spearmanr(df.fg_out[ref_ok & rated], h[ref_ok & rated]).statistic
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args()
    df = read_keys()

    results = {}
    for name in PAPER:
        sub = matched(condition(df, name))
        results[name] = rated_table(
            sub,
            ROWS,
            SEED[name],
            g1="cp_g1",
            g2="cp_g2",
            comparisons=[("MaSC", "DreamSim")],
            B=args.B,
        )
        print(f"  {name}: {len(sub):,} keys")
    aud = audit(df)
    path = write_json("mask_area_filtering", {"conditions": results, "audit": aud})

    check = PaperCheck(intervals=args.B == DEFAULT_B)
    for title, names in (
        ("Threshold sweep (CP alpha)", list(THRESHOLDS)),
        ("Empty-mask treatments (CP alpha)", ["valid reference", "scale minimum", "raw zero"]),
    ):
        lines = [["Keys", *[f"{results[n]['n_keys']:,}" for n in names]]]
        for i, r in enumerate(ROWS):
            lines.append(
                [r.label, *[f3(results[n]["rows"][r.label]["alpha"]["point"]) for n in names]]
            )
            for n in names:
                check.add(
                    f"{n} {r.label} alpha",
                    results[n]["rows"][r.label]["alpha"]["point"],
                    PAPER[n][1][i],
                )
        diff, diff_c = [], []
        for n in names:
            c = comparison(results[n], "MaSC", "DreamSim", "alpha")
            cc = comparison(results[n], "MaSC", "DreamSim", "alpha", cluster=True)
            diff.append(f"{c['diff_point']:+.3f} {ci3(c['ci95'])}")
            diff_c.append(ci3(cc["ci95"]))
            check.add(f"{n} keys", results[n]["n_keys"], PAPER[n][0], 0)
            check.add(f"{n} MaSC - DreamSim", c["diff_point"], PAPER[n][2])
            check.add_ci(f"{n} MaSC - DreamSim CI", c["ci95"], PAPER[n][3])
        lines += [["MaSC - DreamSim", *diff], ["  subject-cluster CI", *diff_c]]
        print_table(f"{title}, B = {args.B:,}", ["", *names], lines)
    print_table(
        "Filtering audit (5% threshold)",
        ["", "value"],
        [[k, f"{v:.3f}" if isinstance(v, float) else f"{v:,}"] for k, v in aud.items()],
    )
    for k, (v, digits) in PAPER_AUDIT.items():
        check.add(k, aud[k], v, digits)
    print(f"\nWrote {path}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
