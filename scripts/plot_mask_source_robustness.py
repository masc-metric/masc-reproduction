"""Mask-source robustness sweep for MaSC on DreamBench++ — CP only.

Compares the SAM3 baseline against alternative segmenters (CLIPSeg,
Grounded-SAM2, OWLv2-SAM2). Each segmenter's MaSC run is filtered by
`build_dreambenchplus_samples`'s 5% min-fg-fraction rule on its own
masks, so each row sees a different surviving subset.

To make the rows directly comparable, we report α and ρ on the
N-way *intersection* of keys present under every segmenter (same
denominator across all rows).

Reads:
    results/geometry__siglip2-so400m__dreambenchplus_maskedmaxcos_naflex[_masks_<src>]_<method>.jsonl
    data/dreambench_plus/data_human_rating/merged_data/{group1,group2}/<method>-cp.json

Writes:
    report/tables/mask_source_robustness.md

Usage:
    python scripts/plot_mask_source_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import DREAMBENCHPLUS_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

RATINGS_ROOT = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"

METHODS: list[str] = [
    "dreambooth_sd",
    "dreambooth_lora_sdxl",
    "textual_inversion_sd",
    "blip_diffusion",
    "emu2",
    "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_g_sdxl",
]

MODEL_TAG = "siglip2-so400m"
SIGNAL = "geometry"
BASE_TAG = "dreambenchplus_maskedmaxcos_naflex"
RATING_SUFFIX = "cp"
TABLE_PATH = REPO_ROOT / "report" / "tables" / "mask_source_robustness.md"
TABLE_TITLE = "MaSC mask-source robustness on DreamBench++ (CP)"

# (display_label, source_label_or_empty). Empty source_label = SAM3 baseline.
SOURCES: list[tuple[str, str]] = [
    ("SAM3 (baseline)",  ""),
    ("CLIPSeg",          "masks_clipseg"),
    ("Grounded-SAM2",    "masks_grounded_sam2"),
    ("OWLv2 + SAM2",     "masks_owlv2_sam2"),
]


def alpha_interval_arr(x: np.ndarray, y: np.ndarray) -> float:
    N = len(x)
    if N < 2:
        return float("nan")
    d_o = float(np.mean((x - y) ** 2))
    v = np.concatenate([x, y])
    M = 2 * N
    d_e = 2.0 * M * float(np.var(v)) / (M - 1)
    if d_e == 0.0:
        return float("nan")
    return 1.0 - d_o / d_e


def alpha_pooled(row_vals: np.ndarray, g1_vals: np.ndarray, g2_vals: np.ndarray,
                 row_range: tuple[float, float]) -> float:
    if len(row_vals) < 2:
        return float("nan")
    lo, hi = row_range
    if hi == lo:
        return float("nan")
    x = (row_vals - lo) / (hi - lo)
    g1 = g1_vals / 4.0
    g2 = g2_vals / 4.0
    a1 = round(alpha_interval_arr(x, g1), 3)
    a2 = round(alpha_interval_arr(x, g2), 3)
    return round((a1 + a2) / 2.0, 3)


def run_tag_for(source_label: str) -> str:
    return BASE_TAG if not source_label else f"{BASE_TAG}_{source_label}"


def load_scores(source_label: str, method: str) -> dict[str, float]:
    tag = run_tag_for(source_label)
    path = RESULTS_DIR / f"{SIGNAL}__{MODEL_TAG}__{tag}_{method}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[f"{r['concept_id']}-{r['prompt_id']}"] = r["score"]
    return out


def load_group_ratings(method: str, group: str) -> dict[str, float]:
    path = RATINGS_ROOT / group / f"{method}-{RATING_SUFFIX}.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def per_source_finite_scored_keys() -> dict[str, dict[str, set[str]]]:
    """`{source_label: {method: set_of_keys_with_finite_score}}`.

    Sources with no result files on disk are omitted, so the N-way
    intersection reflects only segmenters that actually produced
    results."""
    out: dict[str, dict[str, set[str]]] = {}
    for _, sl in SOURCES:
        per_method: dict[str, set[str]] = {}
        any_present = False
        for m in METHODS:
            scores = load_scores(sl, m)
            if scores:
                any_present = True
            per_method[m] = {k for k, v in scores.items() if np.isfinite(v)}
        if any_present:
            out[sl] = per_method
    return out


def n_way_intersection_keys(
    per_source: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Per method, keys present and finite under every source."""
    out: dict[str, set[str]] = {}
    for m in METHODS:
        sets = [per_source[sl][m] for sl in per_source if per_source[sl][m]]
        out[m] = set.intersection(*sets) if sets else set()
    return out


def stats_on_keyset(
    source_label: str, restrict: dict[str, set[str]]
) -> dict:
    """Compute pooled α / ρ for `source_label` over the keys in
    `restrict[method]` ∩ rated ∩ scored, concatenated across methods."""
    all_xs, all_ys, all_g1, all_g2 = [], [], [], []
    for m in METHODS:
        scores = load_scores(source_label, m)
        g1 = load_group_ratings(m, "group1")
        g2 = load_group_ratings(m, "group2")
        keep = restrict.get(m, set())
        shared = sorted(
            {k for k, v in scores.items() if np.isfinite(v)}
            & set(g1) & set(g2) & keep
        )
        if not shared:
            continue
        xs = np.asarray([scores[k] for k in shared], dtype=float)
        avg = {k: (g1[k] + g2[k]) / 2 for k in shared}
        ys = np.asarray([avg[k] for k in shared], dtype=float)
        g1a = np.asarray([g1[k] for k in shared], dtype=float)
        g2a = np.asarray([g2[k] for k in shared], dtype=float)
        all_xs.append(xs); all_ys.append(ys); all_g1.append(g1a); all_g2.append(g2a)

    if not all_xs:
        return {"alpha": float("nan"), "rho": float("nan"), "n": 0}

    xs_cat = np.concatenate(all_xs)
    ys_cat = np.concatenate(all_ys)
    g1_cat = np.concatenate(all_g1)
    g2_cat = np.concatenate(all_g2)

    lo, hi = float(xs_cat.min()), float(xs_cat.max())
    rho, _ = spearmanr(xs_cat, ys_cat)
    return {
        "alpha": alpha_pooled(xs_cat, g1_cat, g2_cat, (lo, hi)),
        "rho": float(rho),
        "n": int(len(xs_cat)),
    }


def collect() -> dict:
    per_source = per_source_finite_scored_keys()
    intersect = n_way_intersection_keys(per_source)
    n_intersect = sum(len(v) for v in intersect.values())

    rows = []
    for label, sl in SOURCES:
        if sl not in per_source and sl != "":
            continue
        rows.append({
            "label": label,
            "source_label": sl,
            "stats": stats_on_keyset(sl, intersect),
        })
    return {"rows": rows, "n_intersection": n_intersect, "n_sources": len(rows)}


def _md_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    def cell(s: str, i: int) -> str:
        return s.rjust(widths[i]) if aligns[i] == "r" else s.ljust(widths[i])

    def sep(i: int) -> str:
        w = max(3, widths[i])
        return ("-" * (w - 1) + ":") if aligns[i] == "r" else ("-" * w)

    out = ["| " + " | ".join(cell(h, i) for i, h in enumerate(headers)) + " |",
           "| " + " | ".join(sep(i) for i in range(len(headers))) + " |"]
    for r in rows:
        out.append("| " + " | ".join(cell(r[i], i) for i in range(len(headers))) + " |")
    return "\n".join(out)


def _fmt_signed(v: float) -> str:
    return "n/a" if np.isnan(v) else f"{v:+.3f}"


def _delta(curr: float, base: float, is_baseline: bool) -> str:
    if is_baseline:
        return "—"
    if np.isnan(curr) or np.isnan(base):
        return "n/a"
    return f"{curr - base:+.3f}"


def render_table(payload: dict) -> str:
    rows = payload["rows"]
    n_int = payload["n_intersection"]
    n_src = payload["n_sources"]
    base = next(r for r in rows if r["source_label"] == "")
    base_a = base["stats"]["alpha"]
    base_r = base["stats"]["rho"]

    headers = ["Source", "α", "Δα", "ρ", "Δρ"]
    aligns = ["l", "r", "r", "r", "r"]
    body = []
    for r in rows:
        s = r["stats"]
        is_base = r["source_label"] == ""
        body.append([
            r["label"],
            _fmt_signed(s["alpha"]),
            _delta(s["alpha"], base_a, is_base),
            _fmt_signed(s["rho"]),
            _delta(s["rho"], base_r, is_base),
        ])

    return "\n".join([
        f"### {n_src}-way key intersection (apples-to-apples, "
        f"N = {n_int:,} keys present under every source)",
        "",
        _md_table(headers, body, aligns),
    ]) + "\n"


def main() -> int:
    payload = collect()
    table = render_table(payload)
    print()
    print(TABLE_TITLE)
    print()
    print(table)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(
        f"# {TABLE_TITLE}\n\n"
        "Each row uses MaSC = SigLIP2-so400m-naflex masked-maxcos with masks\n"
        "from a different segmenter. The 5% min-fg-fraction filter in\n"
        "`build_dreambenchplus_samples` is applied to each segmenter's masks\n"
        "independently, so each row's surviving subset differs. To keep rows\n"
        "directly comparable, α and ρ are computed on the intersection of\n"
        "keys present under every segmenter.\n\n"
        + table
    )
    print(f"Wrote {TABLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
