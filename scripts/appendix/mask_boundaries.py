"""Appendix "Sensitivity to Mask Boundaries": erode or dilate the reference mask.

    python scripts/appendix/mask_boundaries.py [--B 10000]

The reference foreground mask is eroded or dilated with a Euclidean disk of
radius |p| min(H, W), p in {-32, -16, -8, 0, +8, +16, +32}%, while the image
features and the evaluation pairs stay fixed. CP is recomputed on CPU from the
per-reference-patch maximum cosines stored by the MaSC runs with
--dump-patch-arrays; at p = 0 it matches the stored scores to float32
rounding (about 2e-7). If erosion leaves no foreground patch, CP is 0.

DreamBench++: the 7,304 keys retained by the default 5% filter (the filter is
not reapplied after perturbation). ORIDa: the 2,250 within- and 2,250
cross-subject pairs of Table 2. Also reports the share of DreamBench++ pairs
that erosion empties, the 32%-dilation vs default differences, and the
32%-dilated configuration vs GPT-4o. Writes report/appendix/mask_boundaries.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repro_eval.analysis import masks  # noqa: E402
from repro_eval.analysis.keytable import (  # noqa: E402
    decode_arr,
    load_patch_arrays,
    read_jsonl,
    read_keys,
    ref_mask_path,
    retained,
)
from repro_eval.analysis.stats import DEFAULT_B  # noqa: E402
from repro_eval.analysis.tables import (  # noqa: E402
    PaperCheck,
    Row,
    auc_table,
    ci3,
    comparison,
    f3,
    print_table,
    rated_table,
    write_json,
)
from repro_eval.data import REPO_ROOT, RESULTS_DIR, SAMPLE_BUILDERS  # noqa: E402

LEVELS = [-0.32, -0.16, -0.08, 0.0, 0.08, 0.16, 0.32]
ORIDA_CONFIG = REPO_ROOT / "configs/geometry/siglip2_so400m_naflex_orida_50x10.yaml"
ORIDA_RUN = "geometry__siglip2-so400m__orida_train_50x10"

# Camera-ready table: level -> (alpha, [CI], rho, [CI], ORIDa AUC, [CI])
PAPER = {
    -0.32: (-0.396, (-0.417, -0.372), 0.142, (0.118, 0.166), 0.500, (0.500, 0.500)),
    -0.16: (0.083, (0.064, 0.102), 0.226, (0.205, 0.248), 0.501, (0.497, 0.504)),
    -0.08: (0.069, (0.048, 0.087), 0.498, (0.478, 0.517), 0.575, (0.561, 0.590)),
    0.0: (0.465, (0.444, 0.484), 0.603, (0.586, 0.619), 0.992, (0.990, 0.995)),
    0.08: (0.483, (0.466, 0.500), 0.604, (0.587, 0.620), 0.987, (0.984, 0.990)),
    0.16: (0.502, (0.487, 0.518), 0.602, (0.586, 0.618), 0.974, (0.969, 0.978)),
    0.32: (0.506, (0.490, 0.522), 0.595, (0.578, 0.610), 0.949, (0.943, 0.955)),
}
PAPER_EMPTY = {-0.08: 6.2, -0.16: 39.4, -0.32: 88.0}  # % of DreamBench++ pairs, no fg patch left
PAPER_DIFFS = {  # name -> (difference, [CI])
    "DreamBench++ alpha, dilate 32% - default": (0.041, (0.029, 0.057)),
    "DreamBench++ rho, dilate 32% - default": (-0.008, (-0.017, 0.001)),
    "ORIDa AUC, default - dilate 32%": (0.043, (0.037, 0.049)),
    "DreamBench++ alpha, dilate 32% - GPT-4o": (0.006, (-0.009, 0.022)),
    "DreamBench++ rho, dilate 32% - GPT-4o": (-0.043, (-0.059, -0.028)),
}


def label(p: float) -> str:
    return f"p={p:+.2f}"


def name(p: float) -> str:
    return "Default" if p == 0 else f"{'Erode' if p < 0 else 'Dilate'} {abs(round(100 * p))}%"


def dbpp(B: int) -> tuple[dict, dict, dict]:
    df = read_keys()
    rated = df.cp_g1.notna() & df.cp_g2.notna() & df.masc.notna()
    sub = df[rated & retained(df, 0.05)].reset_index(drop=True)
    arrays = load_patch_arrays()
    grids = [(arrays[(r.method, r.key)], str(ref_mask_path(r))) for r in sub.itertuples()]
    for p in LEVELS:
        sub[label(p)] = [
            masks.masked_maxcos(nc, masks.perturbed_grid(mp, p, *g)) for (nc, g), mp in grids
        ]
    exact = float(np.max(np.abs(sub[label(0.0)].to_numpy() - sub.masc.to_numpy())))
    empty = {
        label(p): 100.0
        * float(
            np.mean([not (masks.perturbed_grid(mp, p, *g) > 0.5).any() for (_, g), mp in grids])
        )
        for p in LEVELS
    }
    rows = [Row(label(p), label(p)) for p in LEVELS]
    table = rated_table(
        sub,
        rows,
        "perturbation_dbpp_A",
        g1="cp_g1",
        g2="cp_g2",
        comparisons=[(label(0.32), label(0.0))],
        cluster_col=None,
        human_row=False,
        B=B,
    )
    table["max_abs_diff_p0_vs_stored_scores"] = exact
    table["no_foreground_patch_pct"] = empty
    vs = sub[sub.gpt4o.notna()].reset_index(drop=True)
    gpt = rated_table(
        vs,
        [
            Row(label(0.32), label(0.32)),
            Row(label(0.0), label(0.0)),
            Row("GPT-4o", "gpt4o", True, "LLM judge"),
        ],
        "perturbation_dilation_vs_gpt4o",
        g1="cp_g1",
        g2="cp_g2",
        comparisons=[(label(0.32), "GPT-4o")],
        cluster_col=None,
        B=B,
    )
    return table, gpt, {"n_keys": len(sub)}


def orida(B: int) -> dict:
    rows = read_jsonl(RESULTS_DIR / f"{ORIDA_RUN}.jsonl")
    aux = {
        (r["ref"], r["out"], r["concept_id"]): r
        for r in read_jsonl(RESULTS_DIR / "aux" / f"{ORIDA_RUN}.patch_arrays.jsonl")
    }
    if not rows or not aux:
        raise FileNotFoundError(f"{ORIDA_RUN}: score ORIDa MaSC with --dump-patch-arrays")
    cfg = yaml.safe_load(ORIDA_CONFIG.read_text())
    root = (REPO_ROOT / cfg["images_root"]).resolve()
    samples = SAMPLE_BUILDERS[cfg["sample_source"]](
        root, root, set(cfg["image_exts"]), **cfg["builder_kwargs"]
    )
    mask_of = {
        (s["ref_image"].name, s["out_image"].name, s["concept_id"]): str(s["ref_mask"])
        for s in samples
    }
    within = np.array([r["gen_method"] == "orida_within" for r in rows])
    stored = np.array([r["score"] for r in rows])
    keys = [(r["extras"]["ref"], r["extras"]["out"], r["concept_id"]) for r in rows]
    scores = {}
    for p in LEVELS:
        s = np.array(
            [
                masks.masked_maxcos(
                    decode_arr(aux[k]), masks.perturbed_grid(mask_of[k], p, *aux[k]["ref_grid"])
                )
                for k in keys
            ]
        )
        scores[label(p)] = (s[within], s[~within])
    out = auc_table(scores, "perturbation_orida_A", comparisons=[(label(0.0), label(0.32))], B=B)
    out["max_abs_diff_p0_vs_stored_scores"] = float(
        np.max(
            np.abs(
                np.concatenate(scores[label(0.0)])
                - np.concatenate([stored[within], stored[~within]])
            )
        )
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args()
    db, gpt, meta = dbpp(args.B)
    od = orida(args.B)
    path = write_json(
        "mask_boundaries", {"dreambenchplus": db, "dilation_vs_gpt4o": gpt, "orida": od, **meta}
    )

    check, lines = PaperCheck(intervals=args.B == DEFAULT_B), []
    for p in LEVELS:
        a, r, u = db["rows"][label(p)]["alpha"], db["rows"][label(p)]["rho"], od["rows"][label(p)]
        lines.append(
            [
                name(p),
                f"{f3(a['point'])} {ci3(a['ci95'])}",
                f"{f3(r['point'])} {ci3(r['ci95'])}",
                f"{f3(u['point'])} {ci3(u['ci95'])}",
                f"{db['no_foreground_patch_pct'][label(p)]:.1f}%",
            ]
        )
        pa = PAPER[p]
        check.add(f"{name(p)} alpha", a["point"], pa[0])
        check.add_ci(f"{name(p)} alpha CI", a["ci95"], pa[1])
        check.add(f"{name(p)} rho", r["point"], pa[2])
        check.add_ci(f"{name(p)} rho CI", r["ci95"], pa[3])
        check.add(f"{name(p)} ORIDa AUC", u["point"], pa[4])
        check.add_ci(f"{name(p)} ORIDa AUC CI", u["ci95"], pa[5])
        if p in PAPER_EMPTY:
            check.add(
                f"{name(p)} pairs without fg patch (%)",
                db["no_foreground_patch_pct"][label(p)],
                PAPER_EMPTY[p],
                1,
            )
    print_table(
        f"Mask boundaries ({meta['n_keys']:,} DreamBench++ keys, {od['n_within']:,} + "
        f"{od['n_cross']:,} ORIDa pairs, B = {args.B:,})",
        ["Perturbation", "DreamBench++ alpha", "DreamBench++ rho", "ORIDa AUC", "no fg patch"],
        lines,
    )
    diffs = {
        "DreamBench++ alpha, dilate 32% - default": comparison(
            db, label(0.32), label(0.0), "alpha"
        ),
        "DreamBench++ rho, dilate 32% - default": comparison(db, label(0.32), label(0.0), "rho"),
        "ORIDa AUC, default - dilate 32%": comparison(od, label(0.0), label(0.32)),
        "DreamBench++ alpha, dilate 32% - GPT-4o": comparison(gpt, label(0.32), "GPT-4o", "alpha"),
        "DreamBench++ rho, dilate 32% - GPT-4o": comparison(gpt, label(0.32), "GPT-4o", "rho"),
    }
    print_table(
        "Paired differences",
        ["", "difference [95% CI]"],
        [[k, f"{c['diff_point']:+.3f} {ci3(c['ci95'])}"] for k, c in diffs.items()],
    )
    for k, c in diffs.items():
        check.add(k, c["diff_point"], PAPER_DIFFS[k][0])
        check.add_ci(f"{k} CI", c["ci95"], PAPER_DIFFS[k][1])
    print(
        f"\nCP at p = 0 vs stored scores, max |diff|: DreamBench++ "
        f"{db['max_abs_diff_p0_vs_stored_scores']:.2e}, ORIDa {od['max_abs_diff_p0_vs_stored_scores']:.2e}"
    )
    print(f"Wrote {path}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
