"""Table 2 (ORIDa): AUC with 95% paired bootstrap confidence intervals.

    python scripts/appendix/orida_auc_ci.py [--B 10000]

Reads the eight non-LLM ORIDa runs of README section 2 (2,250 within-subject
and 2,250 cross-subject pairs each). Within and cross pairs are resampled
separately, with the sampled indices shared across metrics. Prints the table's
columns (mean and standard deviation within and across subjects, the
normalised separation, AUC and its interval) and compares them with the
paper: AUC and interval exactly; the descriptive columns, which the paper took
from the submission-time run, within 0.002 (two cells differ in the third
decimal: MaSC's normalised separation and the global pool's cross-subject
mean). The GPT-4o row came from the OpenAI API and is not regenerated here.
Writes report/appendix/table2_orida_auc.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repro_eval.analysis.keytable import read_jsonl  # noqa: E402
from repro_eval.analysis.stats import DEFAULT_B  # noqa: E402
from repro_eval.analysis.tables import PaperCheck, auc_table, ci3, f3, print_table, write_json  # noqa: E402
from repro_eval.data import RESULTS_DIR  # noqa: E402

METRICS = {  # label -> result file (README section 2), in the paper's row order
    "MaSC": "geometry__siglip2-so400m__orida_train_50x10.jsonl",
    "AM-RADIO C-RADIOv4-SO400M": "geometry__C-RADIOv4__orida_train_50x10.jsonl",
    "SigLIP2 SO400M-NaFlex global pool": "geometry__siglip2-so400m__orida_train_50x10_global_naflex.jsonl",
    "DINOv3": "geometry__dinov3-vitl16__orida_train_50x10_dbplus.jsonl",
    "DreamSim": "geometry__dreamsim-ensemble__orida_train_50x10.jsonl",
    "DINO-I": "geometry__dino-vits8__orida_train_50x10.jsonl",
    "DIFT-SDXL": "geometry__dift-sdxl__orida_train_50x10_canonical.jsonl",
    "CLIP-I": "geometry__clip-vit__orida_train_50x10.jsonl",
}
# Camera-ready Table 2: mean within (sd), mean cross (sd), delta_norm, AUC, 95% CI
PAPER = {
    "MaSC": (0.740, 0.071, 0.402, 0.064, 0.513, 0.992, (0.990, 0.995)),
    "AM-RADIO C-RADIOv4-SO400M": (0.775, 0.124, 0.496, 0.081, 0.416, 0.961, (0.956, 0.967)),
    "SigLIP2 SO400M-NaFlex global pool": (0.795, 0.092, 0.597, 0.071, 0.318, 0.954, (0.948, 0.960)),
    "DINOv3": (0.553, 0.230, 0.134, 0.113, 0.394, 0.942, (0.935, 0.949)),
    "DreamSim": (0.521, 0.169, 0.308, 0.093, 0.241, 0.873, (0.862, 0.883)),
    "DINO-I": (0.495, 0.201, 0.245, 0.141, 0.238, 0.848, (0.836, 0.858)),
    "DIFT-SDXL": (0.737, 0.063, 0.652, 0.062, 0.174, 0.834, (0.822, 0.845)),
    "CLIP-I": (0.773, 0.085, 0.678, 0.074, 0.167, 0.800, (0.787, 0.812)),
}
KEY = ["concept_id", "ref", "out", "kind"]


def load_pairs() -> pd.DataFrame:
    """One row per ORIDa pair with every metric's score; the pair order is the
    MaSC run's (the order the bootstrap indices refer to)."""
    df = None
    for label, fname in METRICS.items():
        rows = read_jsonl(RESULTS_DIR / fname)
        if not rows:
            raise FileNotFoundError(f"{RESULTS_DIR / fname} (README section 2)")
        f = pd.DataFrame(
            {
                "concept_id": [r["concept_id"] for r in rows],
                "ref": [r["extras"]["ref"] for r in rows],
                "out": [r["extras"]["out"] for r in rows],
                "kind": [r["gen_method"] for r in rows],
                label: [r["score"] for r in rows],
            }
        )
        n = len(f)
        df = f if df is None else df.merge(f, on=KEY, how="inner")
        if len(df) != n:
            raise RuntimeError(f"{label}: pair set differs from the other metrics")
    return df


def describe(within: np.ndarray, cross: np.ndarray) -> dict:
    """Table 2's descriptive columns; delta_norm = mean difference over the
    metric's observed range on both pair sets."""
    lo, hi = min(within.min(), cross.min()), max(within.max(), cross.max())
    return {
        "mean_within": float(within.mean()),
        "sd_within": float(within.std()),
        "mean_cross": float(cross.mean()),
        "sd_cross": float(cross.std()),
        "delta_norm": float((within.mean() - cross.mean()) / (hi - lo)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args()

    df = load_pairs()
    w = (df.kind == "orida_within").to_numpy()
    scores = {k: (df[k].to_numpy(float)[w], df[k].to_numpy(float)[~w]) for k in METRICS}
    res = auc_table(
        scores,
        "table2_orida_auc",
        B=args.B,
        comparisons=[("MaSC", k) for k in METRICS if k != "MaSC"],
    )
    res["descriptive"] = {k: describe(*v) for k, v in scores.items()}
    path = write_json("table2_orida_auc", res)

    check, lines = PaperCheck(intervals=args.B == DEFAULT_B), []
    for k in METRICS:
        d, a = res["descriptive"][k], res["rows"][k]
        lines.append(
            [
                k,
                f"{f3(d['mean_within'])} ± {f3(d['sd_within'])}",
                f"{f3(d['mean_cross'])} ± {f3(d['sd_cross'])}",
                f"{d['delta_norm']:+.3f}",
                f3(a["point"]),
                ci3(a["ci95"]),
            ]
        )
        p = PAPER[k]
        for name, ours, paper in zip(
            ("mean within", "sd within", "mean cross", "sd cross", "delta_norm", "AUC"),
            (
                d["mean_within"],
                d["sd_within"],
                d["mean_cross"],
                d["sd_cross"],
                d["delta_norm"],
                a["point"],
            ),
            p[:6],
        ):
            exact = name == "AUC"
            check.add(f"{k} {name}", ours, paper, tol=None if exact else 0.002)
        check.add_ci(f"{k} AUC CI", a["ci95"], p[6])
    print_table(
        f"Table 2, ORIDa ({res['n_within']:,} within / {res['n_cross']:,} cross pairs, "
        f"B = {args.B:,})",
        ["Metric", "mean within", "mean cross", "Δnorm", "AUC", "95% CI"],
        lines,
    )
    print(f"\nWrote {path}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
