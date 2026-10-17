"""Machine-readable DreamBench++ comparison bundle for the camera-ready revisions.

Reshapes finished bootstrap outputs; computes no new interval.

    python camera_ready_analysis/cluster_ci/export.py --out DIR

Sources (all B = 10,000):
  - cluster_ci/out/cluster_ci.json   (SLURM 9155078): Tables 1/3/4 replication,
    stripping and boundary subject-cluster intervals
  - report/appendix/mask_area_filtering.json, pf_prompt_strata.json,
    subject_stripping.json, mask_boundaries.json (scripts/appendix, 2026-10-16)

Every key set is re-derived from the key table with the definition the source
used, and stored as a sorted "method|key" list with its sha256, so records on
identical keys are detected and one is marked canonical.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import cluster_ci as cc  # noqa: E402
from repro_eval.analysis.keytable import KEY_TABLE, retained  # noqa: E402
from repro_eval.analysis.stats import SEED_ROOT  # noqa: E402
from repro_eval.data import DREAMBENCHPLUS_DIR, RESULTS_DIR  # noqa: E402

APP = REPO / "report" / "appendix"
CR = HERE / "out" / "cluster_ci.json"

# Original printed values of the camera-ready Overleaf main tables (460ee0f); no interval attaches to these.
ORIGINAL_PRINTED = {
    "table1_cp_7135_keys": {
        "Human inter-rater": (0.658, 0.653), "GPT-4o": (0.499, 0.641), "MaSC": (0.471, 0.611),
        "GPT-4V": (0.432, 0.521), "DreamSim": (0.421, 0.599), "SigLIP2 global pool": (0.369, 0.557),
        "DINOv3": (0.345, 0.479), "DIFT-SDXL": (0.324, 0.502), "DINO-I": (0.311, 0.514),
        "AM-RADIO": (0.226, 0.541), "CLIP-I": (0.135, 0.533)},
    "table3_pf_6607_keys": {
        "Human inter-rater": (0.563, 0.571), "GPT-4o": (0.547, 0.656), "VQAScore": (0.504, 0.640),
        "GPT-4V": (0.485, 0.661), "ImageReward": (0.441, 0.611), "MaSC": (0.354, 0.538),
        "CLIP-T": (0.327, 0.477), "SigLIP2 global pool": (0.326, 0.466), "HPSv3": (0.299, 0.434)},
    "table4_pf_alpha_6607_keys": {
        "BG x full": 0.344, "BG x strip (MaSC)": 0.354, "Full x full": 0.326, "Full x strip": 0.348,
        "FG x full": -0.108},
}
ORIGINAL_NOTES = {
    "table1_cp_7135_keys": "submission-time SAM3 masks and the 7,135-key list were not retained",
    "table3_pf_6607_keys": "submission-time output masks and the 6,607-key list were not retained",
    "table4_pf_alpha_6607_keys": "FG x full -0.108 was computed on 6 methods (BLIP-Diffusion not run), "
                                 "not on the 6,607 keys the caption states",
}

MAF_ROWS = {"GPT-4o": "gpt4o", "MaSC": "masc", "GPT-4V": "gpt4v", "DreamSim": "dreamsim",
            "SigLIP2 global pool": "siglip2_global", "DINOv3": "dinov3", "Human inter-rater": "HUMAN_CP"}
STRATA_ROWS = {"MaSC": "pf_masc", "Full image/full prompt": "pf_full_full"}
STRIP_ROWS = {r.label: r.column for r in cc.STRIP_ROWS} | {"Human inter-rater": "HUMAN_PF"}
CLUSTER_UNIT = ("subject = '<category>/<index>' of the DreamBench++ reference image (one cluster per "
                "reference image; all 7 methods x 9 prompts of a subject share it)")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolved(ci) -> bool:
    return bool(ci[0] > 0 or ci[1] < 0)


# ------------------------------------------------------------ key sets


def keysets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Re-derive every source's key set with that source's own rule."""
    out = {}
    keep5 = retained(df, 0.05)
    pf = keep5 & ~df.is_style
    out["cr_table1"] = cc.matched(df, cc.T1_ROWS, "cp_g1", "cp_g2", keep5)
    out["cr_table3"] = cc.matched(df, cc.T3_ROWS, "pf_g1", "pf_g2", pf)
    out["cr_table4"] = cc.matched(df, cc.T4_ROWS, "pf_g1", "pf_g2", pf)
    out["cr_stripping"] = cc.matched(df, cc.STRIP_ROWS, "pf_g1", "pf_g2", pf)
    rated = df.cp_g1.notna() & df.cp_g2.notna() & df.masc.notna()
    bd = df[rated & keep5].reset_index(drop=True)
    out["cr_boundary"] = bd
    out["cr_boundary_gpt4o"] = bd[bd.gpt4o.notna()].reset_index(drop=True)
    # mask_area_filtering.py: condition, then all six rows and both rating groups non-NaN
    cols = ["gpt4o", "masc", "gpt4v", "dreamsim", "siglip2_global", "dinov3"]
    for name, t in {"0%": 0.0, "2.5%": 0.025, "5%": 0.05, "10%": 0.10}.items():
        d = df[retained(df, t)]
        out[f"maf|{name}"] = d[d.cp_g1.notna() & d.cp_g2.notna() & d[cols].notna().all(axis=1)]
    d = df[df.masc_n_ref_fg_patches > 0]
    out["maf|valid reference"] = d[d.cp_g1.notna() & d.cp_g2.notna() & d[cols].notna().all(axis=1)]
    return out


STRATA = {
    "all": lambda d: pd.Series(True, index=d.index),
    "subject_dependent": lambda d: d.subject_dependent,
    "action_pose": lambda d: d.action_pose,
    "attribute": lambda d: d.attribute,
    "both_action_and_attribute": lambda d: d.action_pose & d.attribute,
    "scene_only": lambda d: d.scene_only,
}  # = scripts/appendix/pf_prompt_strata.py STRATA


def strata_keysets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """pf_prompt_strata.py key sets, rebuilt from its stored labels (no spaCy rerun)."""
    lab = pd.read_csv(APP / "pf_prompt_strata_labels.csv", dtype={"concept_id": str, "prompt_id": str})
    base = df[retained(df, 0.05) & ~df.is_style & df.pf_g1.notna() & df.pf_g2.notna()]
    out = {}
    for rule, cols in (("dependency", {}),
                       ("lexical", {"action_pose_lexical": "action_pose",
                                    "subject_dependent_lexical": "subject_dependent",
                                    "scene_only_lexical": "scene_only"})):
        L = lab.drop(columns=[c for c in ("action_pose", "subject_dependent", "scene_only") if cols])
        L = L.rename(columns=cols)[["concept_id", "prompt_id", "action_pose", "attribute",
                                     "subject_dependent", "scene_only"]]
        d = base.merge(L, on=["concept_id", "prompt_id"])
        d = d[d.pf_masc.notna() & d.pf_full_full.notna()].reset_index(drop=True)
        for name, f in STRATA.items():
            out[f"{rule}|{name}"] = d[f(d).astype(bool)].reset_index(drop=True)
    return out


def keylist(frame: pd.DataFrame) -> list[str]:
    return sorted(f"{m}|{k}" for m, k in zip(frame.method, frame.key))


# ------------------------------------------------------------ records


def rec_rows(analysis_id, rows_key, rows_cluster, colmap):
    out = []
    for lab, r in rows_key.items():
        c = rows_cluster[lab]
        for stat in ("alpha", "rho"):
            assert abs(r[stat]["point"] - c[stat]["point"]) < 1e-9
            out.append({
                "analysis": analysis_id, "metric": lab, "column": colmap.get(lab, lab), "stat": stat,
                "point": r[stat]["point"], "subject_cluster_ci95": c[stat]["ci95"],
                "key_ci95": r[stat]["ci95"], "subject_cluster_se": c[stat]["se"], "key_se": r[stat]["se"],
            })
    return out


def rec_comps(analysis_id, comps_key, comps_cluster, colmap):
    ck = {(c["a"], c["b"], c["stat"]): c for c in comps_key}
    out = []
    for c in comps_cluster:
        k = ck[(c["a"], c["b"], c["stat"])]
        assert abs(k["diff_point"] - c["diff_point"]) < 1e-9, (analysis_id, c["a"], c["b"])
        out.append({
            "analysis": analysis_id, "a": c["a"], "b": c["b"],
            "column_a": colmap.get(c["a"], c["a"]), "column_b": colmap.get(c["b"], c["b"]),
            "stat": c["stat"], "diff_point": c["diff_point"],
            "subject_cluster_ci95": c["ci95"], "subject_cluster_resolved": resolved(c["ci95"]),
            "subject_cluster_p_boot_two_sided": c["p_boot_two_sided"],
            "key_ci95": k["ci95"], "key_resolved": resolved(k["ci95"]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cr = json.load(open(CR))
    maf = json.load(open(APP / "mask_area_filtering.json"))
    strata = json.load(open(APP / "pf_prompt_strata.json"))
    strip = json.load(open(APP / "subject_stripping.json"))
    bnd = json.load(open(APP / "mask_boundaries.json"))

    df = cc.key_table()
    ks = keysets(df)
    sk = strata_keysets(df)

    analyses, rows, comps = [], [], []

    def add_analysis(aid, *, signal, source, key_seed, cluster_seed, frame, definition, n_clusters, target):
        kl = keylist(frame)
        h = hashlib.sha256("\n".join(kl).encode()).hexdigest()
        analyses.append({
            "id": aid, "signal": signal, "source": source,
            "key_bootstrap_seed_label": key_seed, "cluster_bootstrap_seed_label": cluster_seed,
            "n_keys": len(kl), "n_subjects": int(frame.subject.nunique()), "n_clusters_reported": n_clusters,
            "keyset_sha256": h, "keyset_definition": definition, "manuscript_target": target,
        })
        assert len(kl) == len(set(kl))
        if n_clusters is not None:
            assert n_clusters == frame.subject.nunique(), (aid, n_clusters, frame.subject.nunique())
        return kl, h

    keylists = {}

    # --- Tables 1/3/4 replication (cluster_ci run: key and cluster from the same run)
    for aid, name, rowsdef, sig, defin, target in [
        ("table1_replication", "table1", cc.T1_ROWS, "CP",
         "both SAM3 masks > 5% of the image; both CP rating groups and all 10 metric rows non-NaN",
         "new appendix replication table for Table 1"),
        ("table3_replication", "table3", cc.T3_ROWS, "PF",
         "both SAM3 masks > 5%; style subjects excluded; both PF rating groups and all 8 rows non-NaN",
         "new appendix replication table for Table 3"),
        ("table4_replication", "table4", cc.T4_ROWS, "PF",
         "as table3 with the 5 pool x prompt cells non-NaN (FG x full on all 7 methods)",
         "new appendix replication table for Table 4"),
    ]:
        t = cr[name]
        kl, h = add_analysis(aid, signal=sig, source=f"{CR.relative_to(REPO)} (SLURM 9155078)",
                             key_seed=t["seed_label"], cluster_seed=t["seed_label"] + "|cluster",
                             frame=ks[f"cr_{name}"], definition=defin,
                             n_clusters=t["cluster_bootstrap"]["n_clusters"], target=target)
        assert len(kl) == t["n_keys"]
        keylists[h] = kl
        colmap = {r.label: r.column for r in rowsdef} | {"Human inter-rater": f"HUMAN_{sig}"}
        rows += rec_rows(aid, t["rows"], t["cluster_bootstrap"]["rows"], colmap)
        comps += rec_comps(aid, t["comparisons"], t["cluster_bootstrap"]["comparisons"], colmap)

    # --- mask-area filtering (appendix JSON: key and cluster from the same script)
    for cond, c in maf["conditions"].items():
        aid = f"filter|{cond}"
        frame = ks.get(f"maf|{cond}")
        if frame is None:  # scale minimum / raw zero: all rated pairs, empties scored by rule
            frame = df[df.cp_g1.notna() & df.cp_g2.notna()]
        defin = {"scale minimum": "all rated pairs; empty-mask pairs get each metric's observed minimum",
                 "raw zero": "all rated pairs; empty-mask pairs get raw score 0 before normalisation",
                 "valid reference": "pairs with at least one reference foreground patch; no output filter"
                 }.get(cond, f"both SAM3 masks > {cond} of the image (0%: non-empty)") + \
            "; both CP rating groups and the 6 rows non-NaN"
        kl, h = add_analysis(aid, signal="CP", source="report/appendix/mask_area_filtering.json",
                             key_seed=c["seed_label"], cluster_seed=c["seed_label"] + "|cluster",
                             frame=frame, definition=defin, n_clusters=c["cluster_bootstrap"]["n_clusters"],
                             target="appendix mask-area filtering / empty-mask paragraphs")
        assert len(kl) == c["n_keys"], (aid, len(kl), c["n_keys"])
        keylists[h] = kl
        # "scale minimum" and "raw zero" rescore empty-mask pairs, so their score columns differ from the
        # stored ones although the key set is shared: tag the columns with the scoring rule.
        cmap = MAF_ROWS if cond not in ("scale minimum", "raw zero") else \
            {k: (v if v.startswith("HUMAN") else f"{v}@{cond.replace(' ', '_')}") for k, v in MAF_ROWS.items()}
        rows += rec_rows(aid, c["rows"], c["cluster_bootstrap"]["rows"], cmap)
        comps += rec_comps(aid, c["comparisons"], c["cluster_bootstrap"]["comparisons"], cmap)

    # --- PF strata (appendix JSON)
    for rule in ("dependency", "lexical"):
        for name, c in strata[rule].items():
            aid = f"strata|{rule}|{name}"
            kl, h = add_analysis(
                aid, signal="PF", source="report/appendix/pf_prompt_strata.json",
                key_seed=c["seed_label"], cluster_seed=c["seed_label"] + "|cluster", frame=sk[f"{rule}|{name}"],
                definition=f"both SAM3 masks > 5%; style excluded; PF groups, MaSC and full/full non-NaN; {rule} "
                           f"rule, stratum '{name}' (labels: report/appendix/pf_prompt_strata_labels.csv)",
                n_clusters=c["cluster_bootstrap"]["n_clusters"], target="appendix PF prompt-type tables")
            assert len(kl) == c["n_keys"], (aid, len(kl), c["n_keys"])
            keylists[h] = kl
            rows += rec_rows(aid, c["rows"], c["cluster_bootstrap"]["rows"], STRATA_ROWS)
            comps += rec_comps(aid, c["comparisons"], c["cluster_bootstrap"]["comparisons"], STRATA_ROWS)

    # --- stripping: key interval as printed (stripper_ablation), cluster from the cluster_ci run
    t = cr["stripping"]
    kl, h = add_analysis("stripping", signal="PF",
                         source="key: report/appendix/subject_stripping.json; cluster: " + str(CR.relative_to(REPO)),
                         key_seed=strip["seed_label"], cluster_seed=t["seed_label"] + "|cluster",
                         frame=ks["cr_stripping"],
                         definition="both SAM3 masks > 5%; style excluded; PF groups and the 3 stripper rows non-NaN",
                         n_clusters=t["cluster_bootstrap"]["n_clusters"], target="appendix stripping table")
    assert len(kl) == strip["n_keys"] == t["n_keys"]
    keylists[h] = kl
    rows += rec_rows("stripping", strip["rows"], t["cluster_bootstrap"]["rows"], STRIP_ROWS)
    comps += rec_comps("stripping", strip["comparisons"], t["cluster_bootstrap"]["comparisons"], STRIP_ROWS)

    # --- boundary: key interval as printed (perturbation_*), cluster from the cluster_ci run
    bmap = {"p=+0.32": "Dilate 32%", "p=+0.00": "Default", "GPT-4o": "GPT-4o"}
    bcols = {"Dilate 32%": "masc_ref_dilate_0.32", "Default": "masc", "GPT-4o": "gpt4o"}
    for aid, cr_name, e_block in (("boundary", "boundary", bnd["dreambenchplus"]),
                                  ("boundary_vs_gpt4o", "boundary_vs_gpt4o", bnd["dilation_vs_gpt4o"])):
        t = cr[cr_name]
        kl, h = add_analysis(aid, signal="CP",
                             source="key: report/appendix/mask_boundaries.json; cluster: " + str(CR.relative_to(REPO)),
                             key_seed=e_block["seed_label"], cluster_seed=t["seed_label"] + "|cluster",
                             frame=ks["cr_boundary" if cr_name == "boundary" else "cr_boundary_gpt4o"],
                             definition="both SAM3 masks > 5%; CP groups and MaSC non-NaN"
                                        + ("; GPT-4o non-NaN" if aid.endswith("gpt4o") else ""),
                             n_clusters=t["cluster_bootstrap"]["n_clusters"], target="appendix mask-boundary table")
        assert len(kl) == e_block["n_keys"] == t["n_keys"]
        keylists[h] = kl
        krows = {bmap[k]: v for k, v in e_block["rows"].items() if k in bmap}
        crows = t["cluster_bootstrap"]["rows"]
        rows += rec_rows(aid, {k: v for k, v in krows.items() if k in crows}, crows, bcols)
        kcomps = [dict(c, a=bmap.get(c["a"], c["a"]), b=bmap.get(c["b"], c["b"])) for c in e_block["comparisons"]]
        comps += rec_comps(aid, kcomps, t["cluster_bootstrap"]["comparisons"], bcols)

    # --- duplicates: same key set, same columns, same statistic -> one canonical record
    aset = {a["id"]: a for a in analyses}
    priority = ["filter|", "strata|", "stripping", "boundary", "table1_replication", "table3_replication",
                "table4_replication"]

    def prio(aid):
        return next(i for i, p in enumerate(priority) if aid.startswith(p))

    def mark(records, sig):
        groups = {}
        for r in records:
            h = aset[r["analysis"]]["keyset_sha256"] or r["analysis"]
            groups.setdefault((h, *sig(r)), []).append(r)
        dups = []
        for g in groups.values():
            g.sort(key=lambda r: prio(r["analysis"]))
            g[0]["canonical"], g[0]["duplicate_of"] = True, None
            for r in g[1:]:
                r["canonical"] = False
                r["duplicate_of"] = f"{g[0]['analysis']}"
            if len(g) > 1:
                dups.append([r["analysis"] for r in g])
        return dups

    dup_c = mark(comps, lambda r: (r["column_a"], r["column_b"], r["stat"]))
    dup_r = mark(rows, lambda r: (r["column"], r["stat"]))

    for i, c in enumerate(comps):
        c["id"] = f"{c['analysis']}::{c['a']} - {c['b']}::{c['stat']}"

    # --- inputs and environment
    inputs = {str(KEY_TABLE.relative_to(REPO)): sha256(KEY_TABLE)}
    for (sig, model, tag) in [*cc.EXTRA_RUNS.values()]:
        for m in cc.METHODS:
            p = RESULTS_DIR / f"{sig}__{model}__{tag}_{m}.jsonl"
            inputs[str(p.relative_to(REPO))] = sha256(p)
    for folder in cc.EXTRA_SHIPPED.values():
        for m in cc.METHODS:
            p = DREAMBENCHPLUS_DIR / folder / f"{m}.json"
            inputs[f"masc-data/dreambench_plus/{folder}/{m}.json"] = sha256(p)
    for p in sorted((RESULTS_DIR / "aux").glob("*dreambenchplus*patch_arrays.jsonl")):
        inputs[str(p.relative_to(REPO))] = sha256(p)
    for p in [CR, *(APP / f for f in ("mask_area_filtering.json", "pf_prompt_strata.json",
                                       "pf_prompt_strata_labels.csv", "subject_stripping.json",
                                       "mask_boundaries.json"))]:
        inputs[str(p.relative_to(REPO))] = sha256(p)
    code = {str(p.relative_to(REPO)): sha256(p) for p in [
        HERE / "cluster_ci.py", HERE / "export.py", REPO / "repro_eval/analysis/stats.py",
        REPO / "repro_eval/analysis/tables.py", REPO / "repro_eval/analysis/keytable.py",
        REPO / "repro_eval/analysis/masks.py", REPO / "scripts/appendix/mask_area_filtering.py",
        REPO / "scripts/appendix/pf_prompt_strata.py", REPO / "scripts/appendix/subject_stripping.py",
        REPO / "scripts/appendix/mask_boundaries.py"]}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "repro_eval", "scripts"],
                           capture_output=True, text=True).stdout.strip()

    meta = {
        "status": "replication intervals for the approved DreamBench++ revisions; ORIDa excluded (on hold)",
        "created": "2026-10-17",
        "release_commit": head, "release_tracked_code_modified": bool(dirty),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "B": 10_000,
        "interval": "95% percentile: numpy.nanpercentile(replicates, [2.5, 97.5]) (linear interpolation)",
        "resolved_rule": "an interval is 'resolved' when it excludes 0 (lower > 0 or upper < 0)",
        "p_boot_two_sided": "min(1, 2 * min(mean(d <= 0), mean(d >= 0))) over the paired replicates",
        "rng": f"numpy.random.default_rng(numpy.random.SeedSequence([{SEED_ROOT}, zlib.crc32(seed_label.encode())]))",
        "primary_resampling": {
            "unit": CLUSTER_UNIT,
            "procedure": "per replicate: draw = rng.integers(0, n_clusters, size=n_clusters) over the sorted "
                         "unique subjects; the replicate is the concatenation of all keys of each drawn subject, "
                         "once per draw (a subject drawn k times contributes its keys k times)",
            "seed_label": "<analysis key seed label> + '|cluster'",
        },
        "secondary_resampling": {
            "unit": "key (one generated image per method x subject x prompt)",
            "procedure": "index matrices rng.integers(0, n, size=(b, n)) in chunks of b = 250 replicates",
            "caveat": "treats the 63 keys of a subject, which share its reference image and mask, as independent",
        },
        "pairing": "within an analysis, every metric row uses the same index set per replicate; differences are "
                   "computed replicate by replicate",
        "statistics": {
            "alpha": "Kd_o = mean over the two human rating groups g of interval Krippendorff alpha(row, g). "
                     "Continuous rows are min-max normalised to [0, 1] on the rows of the resample (recomputed in "
                     "every replicate) and compared with g / 4; 0-4 rubric rows (GPT-4o, GPT-4V) are used raw "
                     "against raw g",
            "rho": "Spearman (average ranks) between the row and the mean of the two rating groups, ranks "
                   "recomputed in every replicate",
            "human_row": "alpha(g1, g2) on raw ratings and Spearman(g1, g2)",
        },
        "verification": "cluster_ci.py self-tests (passed, SLURM 9155078): multiplicity of duplicated subjects, and "
                        "alpha / rho / human ranks per replicate equal an independent recomputation on the "
                        "materialised resample to 1e-9; all replicates finite",
        "scope": "intervals describe the regenerated replication evaluation, not the printed main-table values",
        "original_printed": {"values": ORIGINAL_PRINTED, "notes": ORIGINAL_NOTES,
                             "format": "table1/table3: (alpha, rho); table4: alpha"},
        "canonical_rule": "records on identical key sets, columns and statistic are the same comparison computed in "
                          "different bootstrap runs; they differ only by Monte Carlo error. One is canonical, by "
                          "source priority: " + " > ".join(priority) + ". Quote the canonical record everywhere. "
                          "A single-run alternative needs a CPU rerun with one shared seed label (not done).",
        "seed_labels_differ_from_published_key_intervals": "stripping and boundary keep the published key intervals "
                          "(seed labels stripper_ablation, perturbation_dbpp_A, perturbation_dilation_vs_gpt4o) and "
                          "take subject-cluster intervals from cluster_ci (cr_stripping|cluster, cr_boundary|cluster, "
                          "cr_boundary_gpt4o|cluster)",
        "inputs_sha256": inputs, "code_sha256": code,
        "slurm": {"cluster_ci": "9155078 (standard, 8 cores, 2026-10-16 17:01-17:17)"},
    }
    bundle = {"meta": meta, "analyses": analyses, "rows": rows, "comparisons": comps,
              "duplicate_groups": {"comparisons": dup_c, "rows": dup_r}}
    (args.out / "dbpp_comparisons.json").write_text(json.dumps(bundle, indent=1) + "\n")

    def flat(r):
        r = dict(r)
        for k in ("subject_cluster_ci95", "key_ci95"):
            v = r.pop(k)
            r[k.replace("ci95", "lo")], r[k.replace("ci95", "hi")] = (v if v else (None, None))
        return r

    pd.DataFrame([flat(c) for c in comps]).to_csv(args.out / "dbpp_comparisons.csv", index=False)
    pd.DataFrame([flat(r) for r in rows]).to_csv(args.out / "dbpp_rows.csv", index=False)
    pd.DataFrame(analyses).to_csv(args.out / "dbpp_analyses.csv", index=False)
    with gzip.open(args.out / "dbpp_keysets.json.gz", "wt") as f:
        json.dump(keylists, f)
    print(f"{len(analyses)} analyses, {len(rows)} row records, {len(comps)} comparisons, "
          f"{len(dup_c)} duplicate comparison groups, {len(dup_r)} duplicate row groups")
    for g in dup_c:
        print("dup comparisons:", g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
