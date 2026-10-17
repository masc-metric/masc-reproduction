"""Subject-cluster paired bootstrap for the comparisons the camera-ready text makes.

Existing predictions only (scoring runs under results/, ratings shipped with
DreamBench++); no GPU, no API. Local analysis, not part of the public release.

    python camera_ready_analysis/cluster_ci/cluster_ci.py [--B 10000] [--out DIR]

DreamBench++ (replication key sets, 5% filter): Table 1 (10 metric rows,
CP), Table 3 (8 rows, PF, style excluded), Table 4 (PF pool x strip cells,
7 methods), the boundary comparisons (dilate 32% vs default, vs GPT-4o) and the
stripping comparisons. Every table reports the key bootstrap and the subject
bootstrap (`subject` = one reference image; 63 keys per subject) with one
index set shared by all rows, so differences are paired.

ORIDa (Table 2): pair-level bootstrap (as printed), anchor-subject bootstrap,
and a two-way subject bootstrap: draw 50 subjects with replacement (counts c),
weight a within pair of subject s by c_s and a cross pair (a, b) by c_a * c_b,
and compute the weighted AUC.

Before any table, `self_tests()` checks that (1) a subject drawn k times
contributes all its keys k times, (2) alpha (with min-max normalisation),
rho and the human ranks in each cluster replicate equal an independent
recomputation on the explicitly materialised resample, and (3) the weighted
AUC equals the plain AUC with unit weights and the AUC of explicitly
replicated pairs with integer weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from repro_eval.analysis import masks  # noqa: E402
from repro_eval.analysis.keytable import (  # noqa: E402
    METHODS,
    _scores,
    _shipped,
    key_of,
    load_patch_arrays,
    read_jsonl,
    read_keys,
    ref_mask_path,
    retained,
)
from repro_eval.analysis.stats import (  # noqa: E402
    Dist,
    RatedTable,
    auc,
    bootstrap_auc,
    bootstrap_rated,
    diff_summary,
    table_rng,
)
from repro_eval.analysis.tables import Row, rated_table  # noqa: E402
from repro_eval.data import RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------- columns

EXTRA_RUNS = {  # column -> (signal, model, tag of the unfiltered run)
    "dift": ("geometry", "dift-sdxl", "dreambenchplus_canonical_nofilter"),
    "radio": ("geometry", "C-RADIOv4", "dreambenchplus_summary_nofilter"),
    "vqascore": ("prompt_following", "vqascore", "dreambenchplus_nofilter"),
    "imagereward": ("prompt_following", "imagereward", "dreambenchplus_nofilter"),
    "hpsv3": ("prompt_following", "hpsv3", "dreambenchplus_nofilter"),
    "pf_full_strip": ("prompt_following", "siglip2-so400m", "dreambenchplus_global_nosubj_nofilter"),
    "pf_fg_full": ("prompt_following", "siglip2-so400m", "dreambenchplus_global_fg_nofilter"),
}
EXTRA_SHIPPED = {  # column -> folder under data/dreambench_plus (as the paper scripts load them)
    "dino_i": "data_dino_rating",
    "clip_i": "data_clipi_rating",
    "clip_t": "data_clipt_rating",
    "gpt4o_pf": "data_gpt_rating/prompt_following_wo_internal_thinking",
    "gpt4v_pf": "data_gpt_rating/prompt_following_gpt4v_full",
}

T1_ROWS = [
    Row("GPT-4o", "gpt4o", True, "LLM judge"),
    Row("MaSC", "masc", False, "ours"),
    Row("GPT-4V", "gpt4v", True, "LLM judge"),
    Row("DreamSim", "dreamsim"),
    Row("SigLIP2 global pool", "siglip2_global"),
    Row("DINOv3", "dinov3"),
    Row("DIFT-SDXL", "dift"),
    Row("DINO-I", "dino_i"),
    Row("AM-RADIO", "radio"),
    Row("CLIP-I", "clip_i"),
]
T3_ROWS = [
    Row("GPT-4o", "gpt4o_pf", True, "LLM judge"),
    Row("VQAScore", "vqascore"),
    Row("GPT-4V", "gpt4v_pf", True, "LLM judge"),
    Row("ImageReward", "imagereward"),
    Row("MaSC", "pf_masc", False, "ours"),
    Row("CLIP-T", "clip_t"),
    Row("SigLIP2 global pool", "pf_full_full"),
    Row("HPSv3", "hpsv3"),
]
T4_ROWS = [
    Row("BG x strip (MaSC)", "pf_masc", False, "ours"),
    Row("BG x full", "pf_bg_full", False, "control"),
    Row("Full x full", "pf_full_full", False, "control"),
    Row("Full x strip", "pf_full_strip", False, "control"),
    Row("FG x full", "pf_fg_full", False, "control"),
]
STRIP_ROWS = [
    Row("Regex", "pf_masc", False, "ours"),
    Row("No stripping", "pf_bg_full", False, "control"),
    Row("spaCy", "pf_bg_spacy", False, "control"),
]


def key_table() -> pd.DataFrame:
    df = read_keys()
    for c in [*EXTRA_RUNS, *EXTRA_SHIPPED]:
        df[c] = np.nan
    for m in METHODS:
        sel = df.method == m
        src = {c: _scores(sig, model, tag, m) for c, (sig, model, tag) in EXTRA_RUNS.items()}
        src |= {c: _shipped(folder, m) for c, folder in EXTRA_SHIPPED.items()}
        for c, d in src.items():
            df.loc[sel, c] = df.loc[sel, "key"].map(d).astype(float).to_numpy()
    return df


def matched(df: pd.DataFrame, rows: list[Row], g1: str, g2: str, extra=True) -> pd.DataFrame:
    ok = df[g1].notna() & df[g2].notna() & extra
    for r in rows:
        ok &= df[r.column].notna()
    return df[ok].reset_index(drop=True)


# ---------------------------------------------------------------- self-tests


def _alpha_independent(x: np.ndarray, a: np.ndarray) -> float:
    """Two-coder interval alpha from pairwise disagreement sums (no shared code)."""
    v = np.concatenate([x, a])
    n = len(v)
    d_o = float(np.sum((x - a) ** 2)) / len(x)
    s1, s2 = float(v.sum()), float((v * v).sum())
    d_e = (2 * n * s2 - 2 * s1 * s1) / (n * (n - 1))  # sum_{i != j} (v_i - v_j)^2 / (n (n - 1))
    return 1.0 - d_o / d_e


def _kd_independent(frame: pd.DataFrame, col: str, ordinal: bool, g1: str, g2: str) -> float:
    s = frame[col].to_numpy(float)
    a, b = frame[g1].to_numpy(float), frame[g2].to_numpy(float)
    if not ordinal:
        s = (s - s.min()) / (s.max() - s.min())
        a, b = a / 4.0, b / 4.0
    return (_alpha_independent(s, a) + _alpha_independent(s, b)) / 2.0


def self_tests(df: pd.DataFrame) -> dict:
    sub = matched(df, T1_ROWS[:4], "cp_g1", "cp_g2", retained(df, 0.05))
    rows = T1_ROWS[:4]
    t = RatedTable(
        rows={r.label: sub[r.column].to_numpy(float) for r in rows},
        ordinal={r.label: r.ordinal for r in rows},
        g1=sub.cp_g1.to_numpy(float),
        g2=sub.cp_g2.to_numpy(float),
        clusters=sub.subject.to_numpy(),
    )
    label, n_rep = "selftest_cluster", 4
    res = bootstrap_rated(t, label, B=n_rep, cluster=True)
    rng = table_rng(label + "|cluster")
    uniq = np.unique(t.clusters)
    sizes = sub.groupby("subject").size()
    report = []
    for i in range(n_rep):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        frame = pd.concat([sub[sub.subject == uniq[d]] for d in draw], ignore_index=True)
        counts = pd.Series(uniq[draw]).value_counts()
        # (1) multiplicity: every drawn subject contributes all its keys, once per draw
        assert len(frame) == int(sum(sizes[uniq[d]] for d in draw))
        per = frame.groupby("subject").size()
        for s, k in counts.items():
            assert per[s] == k * sizes[s], (s, k)
        dup = int((counts > 1).sum())
        h = (frame.cp_g1 + frame.cp_g2) / 2.0
        for r in rows:
            a_ind = _kd_independent(frame, r.column, r.ordinal, "cp_g1", "cp_g2")
            r_ind = spearmanr(frame[r.column], h).statistic
            a_b, r_b = res[r.label]["alpha"].reps[i], res[r.label]["rho"].reps[i]
            # (2) per-replicate recomputation agrees with the independent one
            assert abs(a_ind - a_b) < 1e-9, (r.label, i, a_ind, a_b)
            assert abs(r_ind - r_b) < 1e-9, (r.label, i, r_ind, r_b)
        report.append({"replicate": i, "keys": len(frame), "subjects_drawn_more_than_once": dup})
    # (3) weighted AUC
    rs = np.random.default_rng(0)
    w, c = rs.normal(1, 1, 40).round(1), rs.normal(0, 1, 50).round(1)  # rounding forces ties
    assert abs(weighted_auc(w, np.ones(40), c, np.ones(50)) - float(auc(w, c))) < 1e-12
    ww, wc = rs.integers(0, 4, 40), rs.integers(0, 4, 50)
    explicit = float(auc(np.repeat(w, ww), np.repeat(c, wc)))
    assert abs(weighted_auc(w, ww, c, wc) - explicit) < 1e-12
    return {"cluster_replicates_checked": report, "weighted_auc": "ok"}


# ---------------------------------------------------------------- DreamBench++


def dbpp_table(sub, rows, seed_label, g1, g2, comparisons, B):
    out = rated_table(sub, rows, seed_label, g1=g1, g2=g2, comparisons=comparisons, B=B)
    out["n_subjects"] = int(sub.subject.nunique())
    return out


def boundary_frame(df: pd.DataFrame) -> pd.DataFrame:
    rated = df.cp_g1.notna() & df.cp_g2.notna() & df.masc.notna()
    sub = df[rated & retained(df, 0.05)].reset_index(drop=True)
    arrays = load_patch_arrays()
    for p, col in ((0.0, "p0"), (0.32, "dil32")):
        sub[col] = [
            masks.masked_maxcos(arrays[(r.method, r.key)][0],
                                masks.perturbed_grid(str(ref_mask_path(r)), p, *arrays[(r.method, r.key)][1]))
            for r in sub.itertuples()
        ]
    assert float(np.max(np.abs(sub.p0 - sub.masc))) < 1e-5
    return sub


# ---------------------------------------------------------------- ORIDa


def weighted_auc(w, ww, c, wc) -> float:
    """P(within > cross) + 0.5 P(tie) with pair weights (Mann-Whitney)."""
    w, ww, c, wc = (np.asarray(a, dtype=np.float64) for a in (w, ww, c, wc))
    o = np.argsort(c, kind="mergesort")
    cs, cw = c[o], np.concatenate([[0.0], np.cumsum(wc[o])])
    lo = cw[np.searchsorted(cs, w, side="left")]
    hi = cw[np.searchsorted(cs, w, side="right")]
    return float(np.sum(ww * (lo + 0.5 * (hi - lo))) / (ww.sum() * wc.sum()))


def _load_orida():
    spec = importlib.util.spec_from_file_location("orida_auc_ci", _REPO / "scripts/appendix/orida_auc_ci.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.load_pairs()
    sa, sb = {}, {}
    for r in read_jsonl(RESULTS_DIR / mod.METRICS["MaSC"]):
        k = (r["concept_id"], r["extras"]["ref"], r["extras"]["out"], r["gen_method"])
        sa[k], sb[k] = r["extras"]["subject_a"], r["extras"]["subject_b"]
    keys = list(zip(df.concept_id, df.ref, df.out, df.kind))
    df["subject_a"] = [sa[k] for k in keys]
    df["subject_b"] = [sb[k] for k in keys]
    return df, list(mod.METRICS)


def orida_two_way(B: int) -> dict:
    df, metrics = _load_orida()
    w = (df.kind == "orida_within").to_numpy()
    W, C = df[w].reset_index(drop=True), df[~w].reset_index(drop=True)
    assert (W.subject_a == W.subject_b).all() and (C.subject_a != C.subject_b).all()
    subj = np.unique(np.concatenate([W.subject_a, C.subject_a, C.subject_b]))
    pos = {s: i for i, s in enumerate(subj)}
    iw = W.subject_a.map(pos).to_numpy()
    ia, ib = C.subject_a.map(pos).to_numpy(), C.subject_b.map(pos).to_numpy()
    sc = {m: (W[m].to_numpy(float), C[m].to_numpy(float)) for m in metrics}
    res = {m: Dist(float(auc(*sc[m])), np.empty(B)) for m in metrics}
    rng = table_rng("cr_orida_two_way|cluster")
    n_dup = 0
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(subj), size=len(subj)), minlength=len(subj)).astype(float)
        n_dup += int((cnt > 1).any())
        ww, wc = cnt[iw], cnt[ia] * cnt[ib]
        for m in metrics:
            res[m].reps[b] = weighted_auc(sc[m][0], ww, sc[m][1], wc) if wc.sum() > 0 else np.nan
    pair = bootstrap_auc(sc, "table2_orida_auc", B=B)  # the printed (pair-level) intervals
    anchor = bootstrap_auc(sc, "cr_orida_anchor", B=B, clusters=(W.subject_a.to_numpy(), C.subject_a.to_numpy()))
    out = {"n_within": int(w.sum()), "n_cross": int((~w).sum()), "n_subjects": len(subj), "B": B}
    for name, r in (("pair", pair), ("anchor_subject", anchor), ("two_way_subject", res)):
        out[name] = {
            "rows": {m: r[m].summary() for m in metrics},
            "comparisons": [{"a": "MaSC", "b": m, **diff_summary(r["MaSC"], r[m])} for m in metrics if m != "MaSC"],
        }
    return out


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = key_table()
    out = {"B": args.B, "self_tests": self_tests(df)}
    print("self-tests passed", json.dumps(out["self_tests"]), flush=True)

    keep = retained(df, 0.05)
    t1 = matched(df, T1_ROWS, "cp_g1", "cp_g2", keep)
    out["table1"] = dbpp_table(t1, T1_ROWS, "cr_table1", "cp_g1", "cp_g2",
                               [("MaSC", r.label) for r in T1_ROWS if r.label != "MaSC"], args.B)
    print(f"table1 done ({len(t1)} keys, {time.time() - t0:.0f}s)", flush=True)

    pf = keep & ~df.is_style
    t3 = matched(df, T3_ROWS, "pf_g1", "pf_g2", pf)
    out["table3"] = dbpp_table(t3, T3_ROWS, "cr_table3", "pf_g1", "pf_g2",
                               [("MaSC", r.label) for r in T3_ROWS if r.label != "MaSC"], args.B)
    print(f"table3 done ({len(t3)} keys, {time.time() - t0:.0f}s)", flush=True)

    t4 = matched(df, T4_ROWS, "pf_g1", "pf_g2", pf)
    out["table4"] = dbpp_table(t4, T4_ROWS, "cr_table4", "pf_g1", "pf_g2",
                               [("BG x strip (MaSC)", r.label) for r in T4_ROWS[1:]], args.B)
    print(f"table4 done ({len(t4)} keys, {time.time() - t0:.0f}s)", flush=True)

    st = matched(df, STRIP_ROWS, "pf_g1", "pf_g2", pf)
    out["stripping"] = dbpp_table(st, STRIP_ROWS, "cr_stripping", "pf_g1", "pf_g2",
                                  [("Regex", "No stripping"), ("spaCy", "Regex")], args.B)
    print(f"stripping done ({len(st)} keys, {time.time() - t0:.0f}s)", flush=True)

    bd = boundary_frame(df)
    out["boundary"] = dbpp_table(bd, [Row("Dilate 32%", "dil32"), Row("Default", "p0", False, "ours")],
                                 "cr_boundary", "cp_g1", "cp_g2", [("Dilate 32%", "Default")], args.B)
    bg = bd[bd.gpt4o.notna()].reset_index(drop=True)
    out["boundary_vs_gpt4o"] = dbpp_table(bg, [Row("Dilate 32%", "dil32"), Row("GPT-4o", "gpt4o", True, "LLM judge")],
                                          "cr_boundary_gpt4o", "cp_g1", "cp_g2", [("Dilate 32%", "GPT-4o")], args.B)
    print(f"boundary done ({len(bd)} keys, {time.time() - t0:.0f}s)", flush=True)

    out["orida"] = orida_two_way(args.B)
    print(f"orida done ({time.time() - t0:.0f}s)", flush=True)

    path = args.out / "cluster_ci.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
