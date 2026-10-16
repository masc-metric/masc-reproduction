"""Appendix "Sensitivity to Subject Stripping" (DreamBench++ PF).

    python scripts/appendix/subject_stripping.py [--B 10000]

The background-pooled image embedding is fixed; only the text changes: the
original prompt, regex stripping (the paper's default), or the syntax-aware
spaCy stripper. Evaluated on the 6,605 style-excluded keys retained by the 5%
filter on which all three variants are scored, with paired key-bootstrap
intervals. The Gemini 3.5 Flash rephrasing row of the paper needs a paid API
and is not regenerated here. Writes report/appendix/subject_stripping.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    Row("Regex", "pf_masc", False, "ours"),
    Row("No stripping", "pf_bg_full", False, "control"),
    Row("spaCy", "pf_bg_spacy", False, "control"),
]
# Camera-ready table: alpha, [CI], rho, [CI]
PAPER = {
    "No stripping": (0.343, (0.327, 0.363), 0.471, (0.453, 0.490)),
    "Regex": (0.355, (0.310, 0.396), 0.538, (0.521, 0.556)),
    "spaCy": (0.353, (0.306, 0.396), 0.540, (0.523, 0.557)),
}
PAPER_DIFFS = {
    ("Regex", "No stripping", "rho"): (0.067, (0.052, 0.082)),
    ("Regex", "No stripping", "alpha"): (0.011, (-0.034, 0.051)),
    ("spaCy", "Regex", "alpha"): (-0.001, (-0.004, 0.001)),
    ("spaCy", "Regex", "rho"): (0.002, (-0.001, 0.005)),
}
PAPER_KEYS = 6605


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args()
    df = read_keys()
    ok = retained(df, 0.05) & ~df.is_style & df.pf_g1.notna() & df.pf_g2.notna()
    for r in ROWS:
        ok &= df[r.column].notna()
    sub = df[ok].reset_index(drop=True)
    res = rated_table(
        sub,
        ROWS,
        "stripper_ablation",
        g1="pf_g1",
        g2="pf_g2",
        comparisons=[("Regex", "No stripping"), ("spaCy", "Regex")],
        cluster_col=None,
        B=args.B,
    )
    path = write_json("subject_stripping", res)

    check = PaperCheck(intervals=args.B == DEFAULT_B)
    check.add("keys", res["n_keys"], PAPER_KEYS, 0)
    lines = []
    for lab, (a, a_ci, r, r_ci) in PAPER.items():
        ra, rr = res["rows"][lab]["alpha"], res["rows"][lab]["rho"]
        lines.append(
            [lab, f"{f3(ra['point'])} {ci3(ra['ci95'])}", f"{f3(rr['point'])} {ci3(rr['ci95'])}"]
        )
        check.add(f"{lab} alpha", ra["point"], a)
        check.add_ci(f"{lab} alpha CI", ra["ci95"], a_ci)
        check.add(f"{lab} rho", rr["point"], r)
        check.add_ci(f"{lab} rho CI", rr["ci95"], r_ci)
    print_table(
        f"Subject stripping ({res['n_keys']:,} keys, B = {args.B:,})",
        ["Text processing", "alpha [95% CI]", "rho [95% CI]"],
        lines,
    )
    lines = []
    for (a, b, stat), (d, d_ci) in PAPER_DIFFS.items():
        c = comparison(res, a, b, stat)
        lines.append([f"{a} - {b}, {stat}", f"{c['diff_point']:+.3f} {ci3(c['ci95'])}"])
        check.add(f"{a} - {b} {stat}", c["diff_point"], d)
        check.add_ci(f"{a} - {b} {stat} CI", c["ci95"], d_ci)
    print_table("Paired differences", ["", "difference [95% CI]"], lines)
    print(f"\nWrote {path}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
