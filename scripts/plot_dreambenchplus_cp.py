"""DreamBench++ Concept Preservation analysis.

For every (matcher, generative method) pair, joins our geometry scores
against DreamBench++'s human CP ratings (0-4 ordinal, averaged over
group1+group2). Reports Spearman ρ and AUC(rating=4 vs rating=0).

Reads:
    results/geometry__<matcher>__dreambenchplus_<method>.jsonl
    data/dreambench_plus/data_human_rating/merged_data/{group1,group2}/<method>-cp.json

Writes:
    report/figures/dreambenchplus_cp__heatmap.png
    report/figures/dreambenchplus_cp__distributions.png
    report/figures/dreambenchplus_cp__per_matcher_aggregate.png
    Prints markdown tables to stdout.

Usage:
    python scripts/plot_dreambenchplus_cp.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import DREAMBENCHPLUS_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

RATINGS_ROOT = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"

# Each entry: (display_name, model_id_in_filename, run_tag).
# MaSC candidates — our recall-formula over fg-masked mutual-NN.
MATCHERS: list[tuple[str, str, str]] = [
    ("DINOv3",                         "dinov3-vitb16",        "dreambenchplus"),
    ("DINOv3 (masked-maxcos)",         "dinov3-vitb16",        "dreambenchplus_maskedmaxcos"),
    ("SigLIP2",                        "siglip2-base",         "dreambenchplus"),
    ("SigLIP2 (masked-maxcos)",        "siglip2-base",         "dreambenchplus_maskedmaxcos"),
    ("SigLIP2 so400m-naflex (masked-maxcos)", "siglip2-so400m", "dreambenchplus_maskedmaxcos_naflex"),
    ("LightGlue",                      "superpoint-lightglue", "dreambenchplus"),
    ("LoFTR",                          "loftr-outdoor",        "dreambenchplus"),
    ("RoMa",                           "roma-outdoor",         "dreambenchplus"),
    ("MASt3R",                         "mast3r-vitl",          "dreambenchplus"),
    ("DIFT-SDXL (masked)",             "dift-sdxl",            "dreambenchplus"),
    ("DreamSim (NIGHTS-finetuned ensemble)", "dreamsim-ensemble", "dreambenchplus"),
    ("AM-RADIO C-RADIOv4-SO400M summary cosine", "C-RADIOv4", "dreambenchplus_summary"),
    ("AM-RADIO C-RADIOv4-SO400M (backbone ablation, masked-maxcos)", "C-RADIOv4", "dreambenchplus_maskedmaxcos"),
    # New 2026-04-28 baselines: same-backbone global ablation, modern
    # DINO, canonical DIFT recipe.
    ("SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)", "siglip2-so400m", "dreambenchplus_global_naflex"),
    ("DINOv3 ViT-L/16 CLS cosine (modern DINO baseline)",          "dinov3-vitl16",  "dreambenchplus_dbplus"),
    ("DIFT-SDXL canonical (paper hyperparameters, masked-maxcos)", "dift-sdxl",      "dreambenchplus_canonical"),
]

# External / vanilla baselines — mask-free global feature similarity,
# GPT-as-a-judge variants (DreamBench++ paper, Appendix), and the two
# human rater groups (g1 / g2) whose average is the CP ground truth.
#
# Each entry: (display_name, loader_kind, *kind_args).
#   ("DIFT-SDXL (baseline)",   "ours",       "dift-sdxl",            "dreambenchplus_baseline")
#   ("CLIP-I (baseline)",      "dbplus",     "data_clipi_rating",    "")
#   ("GPT-4o CP (full)",       "dbplus_gpt", "concept_preservation_full", "")
#   ("Human — group1",         "dbplus_human_group", "group1", "")
#
# CLIP-T is included for completeness but it's a prompt-following score,
# so it's not expected to correlate with Concept Preservation. Human
# groups correlate against mean(g1, g2) which is the very quantity
# they're averaged into — those rows are inflated self-reference
# ceilings, not independent baselines. Reported here alongside the
# honest ρ(g1, g2) inter-rater agreement (printed separately).
BASELINES: list[tuple[str, str, str, str]] = [
    ("DIFT-SDXL (baseline)",             "ours",                "dift-sdxl",                                "dreambenchplus_baseline"),
    ("CLIP-I (baseline)",                "dbplus",              "data_clipi_rating",                        ""),
    ("CLIP-T (baseline)",                "dbplus",              "data_clipt_rating",                        ""),
    ("DINO-I (baseline)",                "dbplus",              "data_dino_rating",                         ""),
    # GPT-as-a-judge: keep only the highest-correlating variant we
    # found. All 13 GPT-4o CP variants shipped by DreamBench++ lie in
    # ρ ∈ [+0.56, +0.64] on this subset — see the 2026-04-22 REPORT
    # entry for the full ablation table.
    ("GPT-4o CP (no internal thinking)", "dbplus_gpt",          "concept_preservation_wo_internal_thinking",""),
    ("GPT-4V CP (full)",                 "dbplus_gpt",          "concept_preservation_gpt4v_full",          ""),
    # Human rater groups. These are inflated self-reference ceilings
    # (g_i is inside mean(g1, g2)); see printed inter-rater ρ below.
    ("Human — group1 (self-ref)",        "dbplus_human_group",  "group1",                                   ""),
    ("Human — group2 (self-ref)",        "dbplus_human_group",  "group2",                                   ""),
]

METHODS: list[str] = [
    "dreambooth_sd",
    "dreambooth_lora_sdxl",
    "textual_inversion_sd",
    "blip_diffusion",
    "emu2",
    "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_g_sdxl",
]

RATING_COLORS = {
    0: "tab:red", 1: "tab:orange", 2: "tab:olive", 3: "tab:green", 4: "tab:blue",
}


def load_ratings(method: str) -> dict[str, float]:
    """Average CP rating over group1 and group2 per instance key."""
    g1 = json.load((RATINGS_ROOT / "group1" / f"{method}-cp.json").open())
    g2 = json.load((RATINGS_ROOT / "group2" / f"{method}-cp.json").open())
    keys = set(g1) & set(g2)
    return {k: (g1[k] + g2[k]) / 2 for k in keys}


def load_group_ratings(method: str, group: str) -> dict[str, float]:
    """Raw per-group 0-4 CP ratings (for α computation, paper formula)."""
    return {k: float(v) for k, v in
            json.load((RATINGS_ROOT / group / f"{method}-cp.json").open()).items()}


def is_ordinal(kind: str) -> bool:
    """Rows natively on the human 0-4 integer scale. α is computed on
    raw values for these; continuous rows are global-min-max normalized
    to [0, 1] with humans scaled to [0, 1] by /4."""
    return kind in ("dbplus_gpt", "dbplus_human_group")


def alpha_interval_arr(x: np.ndarray, y: np.ndarray) -> float:
    """2-rater interval Krippendorff's α, closed-form O(N). Matches
    the `krippendorff.alpha(level='interval')` package to floating
    precision; avoids the V² coincidence matrix that blows up on
    continuous scores.
        D_o = mean((x - y)²);  D_e = 2 M Var(v) / (M − 1);  α = 1 − D_o/D_e
    """
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


def load_scores(matcher_id: str, run_tag: str, method: str) -> dict[str, float]:
    path = RESULTS_DIR / f"geometry__{matcher_id}__{run_tag}_{method}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[f"{r['concept_id']}-{r['prompt_id']}"] = r["score"]
    return out


def load_dbplus_baseline(folder: str, suffix: str, method: str) -> dict[str, float]:
    """Load a DreamBench++-shipped per-method rating JSON (CLIP-I, CLIP-T, DINO, DINOv2).

    The folder name (e.g. `data_clipi_rating`) lives under
    `data/dreambench_plus/`; `suffix` is "" for the primary variant,
    "_v2" for the secondary where applicable.
    """
    path = DREAMBENCHPLUS_DIR / folder / f"{method}{suffix}.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def load_dbplus_gpt(variant: str, _unused: str, method: str) -> dict[str, float]:
    """Load a DreamBench++-shipped GPT-judge CP rating JSON.

    `variant` is a folder name under `data_gpt_rating/`, e.g.
    `concept_preservation_full`. Values are integer 0-4 ratings; we
    cast to float for uniform downstream handling.
    """
    path = DREAMBENCHPLUS_DIR / "data_gpt_rating" / variant / f"{method}.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def load_dbplus_human_group(group: str, _unused: str, method: str) -> dict[str, float]:
    """Load a single human rater group's CP rating JSON (group1 or group2).

    These rows are inflated self-reference ceilings (the loaded group
    is averaged into the `mean(g1, g2)` target used as ground truth).
    The honest ceiling is ρ(g1, g2), printed separately.
    """
    path = RATINGS_ROOT / group / f"{method}-cp.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def load_row_scores(kind: str, arg_a: str, arg_b: str, method: str) -> dict[str, float]:
    if kind == "ours":
        return load_scores(arg_a, arg_b, method)  # matcher_id, run_tag
    if kind == "dbplus":
        return load_dbplus_baseline(arg_a, arg_b, method)  # folder, suffix
    if kind == "dbplus_gpt":
        return load_dbplus_gpt(arg_a, arg_b, method)  # variant, _unused
    if kind == "dbplus_human_group":
        return load_dbplus_human_group(arg_a, arg_b, method)  # group, _unused
    raise ValueError(f"Unknown row kind: {kind!r}")


def joined_arrays(
    scores: dict[str, float], ratings: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(scores) & set(ratings))
    xs = np.asarray([scores[k] for k in keys], dtype=float)
    ys = np.asarray([ratings[k] for k in keys], dtype=float)
    return xs, ys


def compute_cell_stats(xs: np.ndarray, ys: np.ndarray) -> dict:
    if len(xs) < 3:
        return {"n": len(xs), "rho": float("nan"), "pearson": float("nan"),
                "auc_4_vs_0": float("nan"), "n_pos": 0, "n_neg": 0,
                "alpha": float("nan")}
    rho, _ = spearmanr(xs, ys)
    pear, _ = pearsonr(xs, ys)
    pos_mask = ys >= 3.5  # averaged-rating 4 endpoint (avg of two 4s = 4.0)
    neg_mask = ys <= 0.5  # averaged-rating 0 endpoint
    auc = float("nan")
    if pos_mask.sum() >= 2 and neg_mask.sum() >= 2:
        y_all = np.concatenate([np.ones(int(pos_mask.sum())),
                                np.zeros(int(neg_mask.sum()))])
        s_all = np.concatenate([xs[pos_mask], xs[neg_mask]])
        auc = float(roc_auc_score(y_all, s_all))
    return {
        "n": int(len(xs)),
        "rho": float(rho),
        "pearson": float(pear),
        "auc_4_vs_0": auc,
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
        "alpha": float("nan"),  # filled in by collect_grid using g1/g2 raw.
    }


def alpha_cell(row_vals: np.ndarray, g1_vals: np.ndarray, g2_vals: np.ndarray,
               ordinal: bool, row_range: tuple[float, float]) -> float:
    """Paper-exact Kd_o for one (row, method) cell.

    Formula (DreamBench++ `kd_alpha`):
        a_j = round(α(row, g_j), 3)   for j ∈ {1, 2}
        Kd_o = round((a_1 + a_2) / 2, 3)
    For ordinal (0-4 integer) rows: raw values. For continuous rows:
    global min-max to [0, 1] on `row_vals` and /4 on `g_j`, so every
    rater lives on a [0, 1] scale before α.
    """
    if len(row_vals) < 2:
        return float("nan")
    if ordinal:
        x = row_vals; g1 = g1_vals; g2 = g2_vals
    else:
        lo, hi = row_range
        if hi == lo:
            return float("nan")
        x = (row_vals - lo) / (hi - lo)
        g1 = g1_vals / 4.0
        g2 = g2_vals / 4.0
    a1 = round(alpha_interval_arr(x, g1), 3)
    a2 = round(alpha_interval_arr(x, g2), 3)
    return round((a1 + a2) / 2.0, 3)


def all_rows() -> list[tuple[str, str, str, str, str]]:
    """Return a unified list of (display, kind, arg_a, arg_b, bucket) rows.

    `bucket` ∈ {"matcher", "baseline"} so downstream rendering can tell
    our MaSC candidates from the external baselines.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for name, mid, run_tag in MATCHERS:
        rows.append((name, "ours", mid, run_tag, "matcher"))
    for name, kind, a, b in BASELINES:
        rows.append((name, kind, a, b, "baseline"))
    return rows


def compute_matcher_keysets() -> dict[str, set[str]]:
    """Per method, the intersection of keys present across all MaSC matcher
    result files. Used as the apples-to-apples reference sample set: every
    row (matcher or baseline) is restricted to these keys before we compute
    correlations. This removes the 5% mask-coverage filter asymmetry —
    without it, DB++-shipped baselines see ~9441 samples while our matchers
    see ~7304, which mechanically inflates baseline AUCs (more low-quality
    samples survive in n_neg).

    Only matchers with complete coverage (results for all 7 methods)
    contribute to the intersection. Partial-coverage ablations (e.g.
    masked-maxcos variants that have only run for a subset of methods)
    would otherwise arbitrarily shrink the reference keyset; they still
    appear as table rows, just restricted to whatever keys they share
    with the fully-covered intersection.
    """
    fully_covered: list[tuple[str, str, str]] = []
    for name, mid, run_tag in MATCHERS:
        if all(load_scores(mid, run_tag, m) for m in METHODS):
            fully_covered.append((name, mid, run_tag))
    keysets: dict[str, set[str]] = {}
    for method in METHODS:
        per_matcher = [set(load_scores(mid, run_tag, method))
                       for _name, mid, run_tag in fully_covered]
        keysets[method] = set.intersection(*per_matcher) if per_matcher else set()
    return keysets


def collect_grid() -> dict:
    """Return {grid, pool, row_kinds, inter_rater}.

    Every row is restricted to the same per-method key subset — the
    intersection of all fully-covered MaSC-matcher result files.
    DB++-shipped baselines thus see exactly the samples our matchers
    see. Each cell carries both Spearman ρ and Krippendorff's α (paper
    Kd_o); `inter_rater` holds the honest human ceiling (α(g1, g2) and
    ρ(g1, g2)) on the same apples-to-apples keyset.
    """
    grid: dict[tuple[str, str], dict] = {}
    pool: dict[str, dict] = {}
    row_kinds: dict[str, str] = {}
    row_ordinality: dict[str, bool] = {}
    ratings_by_method = {m: load_ratings(m) for m in METHODS}
    g1_by_method = {m: load_group_ratings(m, "group1") for m in METHODS}
    g2_by_method = {m: load_group_ratings(m, "group2") for m in METHODS}
    keysets = compute_matcher_keysets()

    # Global min/max per continuous row, pooled across all methods on
    # the apples-to-apples keyset. Used for [0,1] normalization before
    # α. Integer 0-4 rows skip this step.
    row_ranges: dict[str, tuple[float, float]] = {}
    rows_meta: list[tuple[str, str, str, str, str]] = list(all_rows())
    for name, kind, arg_a, arg_b, bucket in rows_meta:
        if is_ordinal(kind):
            row_ranges[name] = (0.0, 4.0)
            continue
        vals: list[float] = []
        for method in METHODS:
            scores = load_row_scores(kind, arg_a, arg_b, method)
            keep = keysets.get(method, set())
            for k, v in scores.items():
                if k in keep:
                    vals.append(v)
        row_ranges[name] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    for name, kind, arg_a, arg_b, bucket in rows_meta:
        row_kinds[name] = bucket
        row_ordinality[name] = is_ordinal(kind)
        all_xs, all_ys, all_g1, all_g2 = [], [], [], []
        for method in METHODS:
            scores = load_row_scores(kind, arg_a, arg_b, method)
            keep = keysets.get(method, set())
            if keep:
                scores = {k: v for k, v in scores.items() if k in keep}
            xs, ys = joined_arrays(scores, ratings_by_method[method])
            # Pair row scores with both raw group ratings on the same keys.
            shared_keys = sorted(set(scores) & set(g1_by_method[method]) & set(g2_by_method[method]))
            row_arr = np.asarray([scores[k] for k in shared_keys], dtype=float)
            g1_arr = np.asarray([g1_by_method[method][k] for k in shared_keys], dtype=float)
            g2_arr = np.asarray([g2_by_method[method][k] for k in shared_keys], dtype=float)
            stats = compute_cell_stats(xs, ys)
            stats["alpha"] = alpha_cell(row_arr, g1_arr, g2_arr,
                                        ordinal=is_ordinal(kind),
                                        row_range=row_ranges[name])
            grid[(name, method)] = stats
            all_xs.append(xs)
            all_ys.append(ys)
            all_g1.append(g1_arr)
            all_g2.append(g2_arr)
        if all_xs and any(len(x) for x in all_xs):
            xs_cat = np.concatenate(all_xs)
            ys_cat = np.concatenate(all_ys)
            pooled = compute_cell_stats(xs_cat, ys_cat)
            row_cat = np.concatenate(all_xs)  # same as xs_cat (row vs hmean ordered identically)
            g1_cat = np.concatenate(all_g1)
            g2_cat = np.concatenate(all_g2)
            pooled["alpha"] = alpha_cell(row_cat, g1_cat, g2_cat,
                                         ordinal=is_ordinal(kind),
                                         row_range=row_ranges[name])
            pool[name] = pooled
        else:
            pool[name] = {"n": 0, "rho": float("nan"), "pearson": float("nan"),
                          "auc_4_vs_0": float("nan"), "n_pos": 0, "n_neg": 0,
                          "alpha": float("nan")}

    # Honest human inter-rater ceiling — ρ(g1, g2) and α(g1, g2) per
    # method + pooled, on the apples-to-apples keyset.
    inter_rater = {"per_method": {}, "pooled": {}}
    all_g1p, all_g2p = [], []
    for method in METHODS:
        keys = sorted(set(g1_by_method[method]) & set(g2_by_method[method]) & keysets.get(method, set()))
        if not keys:
            inter_rater["per_method"][method] = {"rho": float("nan"),
                                                 "alpha": float("nan"), "n": 0}
            continue
        g1a = np.asarray([g1_by_method[method][k] for k in keys], dtype=float)
        g2a = np.asarray([g2_by_method[method][k] for k in keys], dtype=float)
        rho, _ = spearmanr(g1a, g2a)
        a = round(alpha_interval_arr(g1a, g2a), 3)
        inter_rater["per_method"][method] = {"rho": float(rho), "alpha": a,
                                              "n": len(keys)}
        all_g1p.append(g1a); all_g2p.append(g2a)
    if all_g1p:
        g1c = np.concatenate(all_g1p); g2c = np.concatenate(all_g2p)
        rho, _ = spearmanr(g1c, g2c)
        a = round(alpha_interval_arr(g1c, g2c), 3)
        inter_rater["pooled"] = {"rho": float(rho), "alpha": a, "n": int(len(g1c))}
    else:
        inter_rater["pooled"] = {"rho": float("nan"), "alpha": float("nan"), "n": 0}

    return {"grid": grid, "pool": pool, "row_kinds": row_kinds,
            "row_ordinality": row_ordinality, "row_ranges": row_ranges,
            "inter_rater": inter_rater}


def plot_heatmap(grid: dict, row_kinds: dict, pool: dict, out_path: Path) -> None:
    # Sort rows by pooled α (primary CP metric). NaN α (missing) last.
    rows = sorted(row_kinds.keys(),
                  key=lambda r: (np.isnan(pool[r]["alpha"]),
                                 -pool[r]["alpha"] if not np.isnan(pool[r]["alpha"]) else 1.0))
    matrix = np.full((len(rows), len(METHODS)), np.nan)
    for i, rname in enumerate(rows):
        for j, method in enumerate(METHODS):
            matrix[i, j] = grid[(rname, method)]["alpha"]

    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(rows) + 2))
    vmax = max(0.4, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    # Italicize baseline labels so they're visually distinct.
    labels = [(f"{r}" if row_kinds[r] == "matcher" else r) for r in rows]
    ax.set_yticklabels(labels)
    for i in range(len(rows)):
        for j in range(len(METHODS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.4 * vmax else "black",
                        fontsize=9)
    ax.set_title("Krippendorff α (Kd_o): score vs human CP rating — sorted by pooled α")
    fig.colorbar(im, ax=ax, shrink=0.8, label="α")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_distributions(out_path: Path) -> None:
    ratings_by_method = {m: load_ratings(m) for m in METHODS}
    rows = all_rows()
    fig = plt.figure(figsize=(14, 3.6 * ((len(rows) + 1) // 2)))
    gs = fig.add_gridspec((len(rows) + 1) // 2, 2, hspace=0.5, wspace=0.22)

    for i, (name, kind, arg_a, arg_b, _bucket) in enumerate(rows):
        buckets: dict[int, list[float]] = {r: [] for r in range(5)}
        total_xs: list[float] = []
        total_ys: list[float] = []
        for method in METHODS:
            scores = load_row_scores(kind, arg_a, arg_b, method)
            xs, ys = joined_arrays(scores, ratings_by_method[method])
            total_xs.append(xs)
            total_ys.append(ys)
            for r in range(5):
                # Round avg to nearest integer for bucketing the histograms.
                mask = np.round(ys).astype(int) == r
                if mask.any():
                    buckets[r].extend(xs[mask].tolist())

        ax = fig.add_subplot(gs[i // 2, i % 2])
        nonempty = [np.asarray(v) for v in buckets.values() if v]
        if not nonempty:
            ax.set_title(f"{name}  (no data yet)", fontsize=10)
            ax.set_axis_off()
            continue
        all_scores = np.concatenate(nonempty)
        lo, hi = float(all_scores.min()), float(all_scores.max())
        bins = np.linspace(lo - 1e-3, hi + 1e-3, 32)
        for r in sorted(buckets):
            v = np.asarray(buckets[r])
            if not len(v):
                continue
            ax.hist(v, bins=bins, alpha=0.55, color=RATING_COLORS[r],
                    edgecolor="black", linewidth=0.35,
                    label=f"rating {r} (n={len(v)}, μ={v.mean():.3f})")
        stats = compute_cell_stats(np.concatenate(total_xs), np.concatenate(total_ys))
        ax.set_title(f"{name}  (pooled across 7 methods; ρ={stats['rho']:+.3f}, n={stats['n']})",
                     fontsize=10)
        ax.set_xlabel("geometry score")
        ax.set_ylabel("count")
        ax.legend(loc="best", fontsize=7)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("DreamBench++ CP — our score vs human rating (bucketed, pooled across methods)",
                 fontsize=13, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_matcher_aggregate(pool: dict, row_kinds: dict, out_path: Path) -> None:
    names = list(row_kinds.keys())
    alphas = np.asarray([pool[n]["alpha"] for n in names])
    rhos = np.asarray([pool[n]["rho"] for n in names])
    aucs = np.asarray([pool[n]["auc_4_vs_0"] for n in names])
    n_obs = np.asarray([pool[n]["n"] for n in names])

    # Primary sort: descending Kd_o α (paper's headline metric).
    def rank_key(i: int) -> tuple:
        a = alphas[i]
        return (np.isnan(a), -a if not np.isnan(a) else 1.0)
    order = np.array(sorted(range(len(names)), key=rank_key))

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.5 * len(names) * 1.5)))

    ys = np.arange(len(names))

    ax = axes[0]
    ax.barh(ys, alphas[order], color="tab:orange", alpha=0.85)
    for y, a, n in zip(ys, alphas[order], n_obs[order]):
        if not np.isnan(a):
            ax.text(a + 0.005 * (1 if a >= 0 else -1), y,
                    f"{a:+.3f} (n={n})", va="center",
                    ha="left" if a >= 0 else "right", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Krippendorff's α (Kd_o, paper's metric)")
    ax.set_title("Sorted by α — pooled across 7 gen methods")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1]
    ax.barh(ys, rhos[order], color="tab:blue", alpha=0.85)
    for y, r in zip(ys, rhos[order]):
        if not np.isnan(r):
            ax.text(r + 0.005 * (1 if r >= 0 else -1), y, f"{r:+.3f}", va="center",
                    ha="left" if r >= 0 else "right", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Rank correlation")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[2]
    ax.barh(ys, aucs[order], color="tab:green", alpha=0.85)
    for y, a in zip(ys, aucs[order]):
        if not np.isnan(a):
            ax.text(a + 0.005, y, f"{a:.3f}", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC (rating=4 vs rating=0)")
    ax.set_title("Endpoint discrimination")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _fmt_rho(v: float) -> str:
    return "  n/a" if not np.isfinite(v) else f"{v:+.3f}"


def _fmt_auc(v: float) -> str:
    return " n/a" if not np.isfinite(v) else f"{v:.3f}"


# Display rows that appear in the paper's `tab:cp_dbplus` (Tab. 1).
# (paper_label, internal_name_in_grid). Order matches the LaTeX table.
# `internal_name` must exist in MATCHERS or BASELINES — collect_grid()
# keys both into the same dicts.
PAPER_CP_ROWS: list[tuple[str, str]] = [
    ("GPT-4o",                            "GPT-4o CP (no internal thinking)"),
    ("MaSC",                              "SigLIP2 so400m-naflex (masked-maxcos)"),
    ("GPT-4V",                            "GPT-4V CP (full)"),
    ("DreamSim",                          "DreamSim (NIGHTS-finetuned ensemble)"),
    ("SigLIP2 SO400M-NaFlex global pool", "SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)"),
    ("DINOv3",                            "DINOv3 ViT-L/16 CLS cosine (modern DINO baseline)"),
    ("DIFT-SDXL",                         "DIFT-SDXL canonical (paper hyperparameters, masked-maxcos)"),
    ("DINO-I",                            "DINO-I (baseline)"),
    ("AM-RADIO C-RADIOv4-SO400M",         "AM-RADIO C-RADIOv4-SO400M summary cosine"),
    ("CLIP-I",                            "CLIP-I (baseline)"),
]

# Compact column headers for the per-method α table (paper format).
METHOD_DISPLAY: dict[str, str] = {
    "dreambooth_sd":               "DB-SD",
    "dreambooth_lora_sdxl":        "DB-LoRA-XL",
    "textual_inversion_sd":        "TI-SD",
    "blip_diffusion":              "BLIP-D",
    "emu2":                        "Emu2",
    "ip_adapter_plus_vit_h_sdxl":  "IP-Plus-XL",
    "ip_adapter_vit_g_sdxl":       "IP-G-XL",
}


def _count_alpha_positive(grid: dict, internal_name: str) -> tuple[int, int]:
    """Per-method α-positive count for a paper row. Returns (n_pos, n_total).

    Used to back the §5.1 prose claim that MaSC is α-positive on all 7
    DB++ gen methods, DreamSim on 6, AM-RADIO summary cosine on 3.
    """
    n_pos = 0
    n_total = 0
    for m in METHODS:
        a = grid.get((internal_name, m), {}).get("alpha", float("nan"))
        if np.isfinite(a):
            n_total += 1
            if a > 0.0:
                n_pos += 1
    return n_pos, n_total


def print_per_method_alpha_table(grid: dict, pool: dict, inter_rater: dict) -> None:
    """Markdown per-method α breakdown for the paper's `tab:cp_dbplus` rows.

    Backs the contribution-(4) claim ("MaSC α-positive on all 7 methods;
    DreamSim 6/7; AM-RADIO summary cosine 3/7") with a printable table.
    """
    print("\n### Per-method Krippendorff α (Kd_o) — paper-row breakdown\n")
    print("Same row set as `tab:cp_dbplus`; same apples-to-apples 7,135-key keyset. "
          "**+/7** is the count of methods where α is positive (§5.1 prose support).\n")
    cols = [METHOD_DISPLAY[m] for m in METHODS]
    print("| metric | " + " | ".join(cols) + " | **pooled** | +/7 |")
    print("|---|" + ":---:|" * len(METHODS) + "---:|---:|")
    # Honest ceiling row first (inter_rater key for "α-positive count" is undefined; report `7/7`).
    cells_h = [_fmt_rho(inter_rater["per_method"][m]["alpha"]) for m in METHODS]
    n_pos_h = sum(1 for m in METHODS
                  if np.isfinite(inter_rater["per_method"][m]["alpha"])
                  and inter_rater["per_method"][m]["alpha"] > 0.0)
    print(f"| *Human inter-rater (ceiling)* | " + " | ".join(cells_h)
          + f" | ***{_fmt_rho(inter_rater['pooled']['alpha']).strip()}*** | {n_pos_h}/7 |")
    for paper_label, internal_name in PAPER_CP_ROWS:
        if internal_name not in pool:
            continue
        cells = [_fmt_rho(grid[(internal_name, m)]["alpha"]) for m in METHODS]
        pooled = pool[internal_name]["alpha"]
        n_pos, n_total = _count_alpha_positive(grid, internal_name)
        print(f"| {paper_label} | " + " | ".join(cells)
              + f" | **{_fmt_rho(pooled).strip()}** | {n_pos}/{n_total} |")


def write_per_method_alpha_latex(grid: dict, pool: dict, inter_rater: dict,
                                  out_path: Path) -> None:
    """Emit Tab. S1 (per-method α breakdown) as a LaTeX `tabular` block.

    Direct paste-in for the paper supplementary; cells use the same
    +0.000 / -0.000 formatting as the headline tables.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_cell(v: float) -> str:
        if not np.isfinite(v):
            return "---"
        # Render negatives as $-$ for LaTeX so the minus sign is the math one.
        s = f"{v:+.3f}"
        return s.replace("-", "$-$")

    cols = [METHOD_DISPLAY[m] for m in METHODS]
    n_pooled = inter_rater["pooled"]["n"]

    lines = [
        r"% Auto-generated by scripts/plot_dreambenchplus_cp.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"    \centering",
        rf"    \caption{{Per-method Krippendorff $\alpha$ (Kd$_o$) on DreamBench++~Concept Preservation. "
        rf"Same lineup as Tab.~\ref{{tab:cp_dbplus}}; same apples-to-apples ${n_pooled:,}$-key subset. "
        rf"Last column counts the gen methods where $\alpha>0$. "
        rf"\textbf{{Bold pooled $\alpha$}} matches the headline table; this supplementary view backs "
        rf"the per-method-asymmetry claim in \S5.1 (\textbf{{MaSC}} is $\alpha$-positive on all 7 methods).}}",
        r"    \label{tab:per_method_alpha}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{l" + "r" * len(METHODS) + r"rc}",
        r"        \toprule",
        r"        Metric & " + " & ".join(cols) + r" & \textbf{pooled} & +/7 \\",
        r"        \midrule",
    ]

    cells_h = [fmt_cell(inter_rater["per_method"][m]["alpha"]) for m in METHODS]
    pooled_h = fmt_cell(inter_rater["pooled"]["alpha"]).replace("$-$", "-")
    pooled_h_tex = fmt_cell(inter_rater["pooled"]["alpha"])
    n_pos_h = sum(1 for m in METHODS
                  if np.isfinite(inter_rater["per_method"][m]["alpha"])
                  and inter_rater["per_method"][m]["alpha"] > 0.0)
    lines.append(r"        \emph{Human inter-rater} & "
                 + " & ".join(rf"\emph{{{c}}}" for c in cells_h)
                 + rf" & \emph{{\textbf{{{pooled_h_tex}}}}} & \emph{{{n_pos_h}/7}} \\")
    lines.append(r"        \midrule")

    for paper_label, internal_name in PAPER_CP_ROWS:
        if internal_name not in pool:
            continue
        cells = [fmt_cell(grid[(internal_name, m)]["alpha"]) for m in METHODS]
        pooled = pool[internal_name]["alpha"]
        pooled_tex = fmt_cell(pooled)
        n_pos, n_total = _count_alpha_positive(grid, internal_name)
        if paper_label == "MaSC":
            label = r"\textbf{MaSC}"
            cells = [rf"\textbf{{{c}}}" for c in cells]
            pooled_tex = rf"\textbf{{{pooled_tex}}}"
            n_field = rf"\textbf{{{n_pos}/{n_total}}}"
        else:
            label = paper_label
            n_field = f"{n_pos}/{n_total}"
        lines.append(rf"        {label} & " + " & ".join(cells)
                     + rf" & {pooled_tex} & {n_field} \\")

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table}",
        "",
    ])
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def print_tables(grid: dict, pool: dict, row_kinds: dict) -> None:
    matchers = [n for n, b in row_kinds.items() if b == "matcher"]
    baselines = [n for n, b in row_kinds.items() if b == "baseline"]

    def dump_bucket(title: str, names: list[str], stat_key: str) -> None:
        ordered = sorted(
            names,
            key=lambda n: (np.isnan(pool[n][stat_key]), -pool[n][stat_key]),
        )
        print(f"\n### {title}\n")
        header = "| name | " + " | ".join(METHODS) + " | **all** |"
        sep = "|---|" + ":---:|" * len(METHODS) + "---:|"
        print(header)
        print(sep)
        for name in ordered:
            cells = [_fmt_rho(grid[(name, m)][stat_key]) for m in METHODS]
            pooled = f"**{_fmt_rho(pool[name][stat_key]).strip()}**"
            print(f"| {name} | " + " | ".join(cells) + f" | {pooled} |")

    # α is the paper's headline metric — print those tables first.
    dump_bucket("Krippendorff α (Kd_o) vs human CP — MaSC matchers", matchers, "alpha")
    dump_bucket("Krippendorff α (Kd_o) vs human CP — baselines", baselines, "alpha")
    dump_bucket("Spearman ρ vs human CP — MaSC matchers", matchers, "rho")
    dump_bucket("Spearman ρ vs human CP — baselines", baselines, "rho")

    print("\n### Pooled aggregate (all 7 gen methods) — sorted by Kd_o α\n")
    print("| name | kind | n | Kd_o α | Spearman ρ | Pearson r | AUC(4 vs 0) | n_pos | n_neg |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    ordered_all = sorted(
        row_kinds.keys(),
        key=lambda n: (np.isnan(pool[n]["alpha"]), -pool[n]["alpha"]),
    )
    for name in ordered_all:
        s = pool[name]
        print(f"| {name} | {row_kinds[name]} | {s['n']} | "
              f"{_fmt_rho(s['alpha'])} | {_fmt_rho(s['rho'])} | {_fmt_rho(s['pearson'])} | "
              f"{_fmt_auc(s['auc_4_vs_0'])} | {s['n_pos']} | {s['n_neg']} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "report/figures")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "report/tables",
                    help="Where to write LaTeX `tabular` files for the paper.")
    args = ap.parse_args()

    data = collect_grid()
    grid, pool, row_kinds = data["grid"], data["pool"], data["row_kinds"]
    inter_rater = data["inter_rater"]

    plot_heatmap(grid, row_kinds, pool, args.out_dir / "dreambenchplus_cp__heatmap.png")
    plot_distributions(args.out_dir / "dreambenchplus_cp__distributions.png")
    plot_per_matcher_aggregate(pool, row_kinds,
                               args.out_dir / "dreambenchplus_cp__per_matcher_aggregate.png")
    print_tables(grid, pool, row_kinds)
    print_per_method_alpha_table(grid, pool, inter_rater)
    write_per_method_alpha_latex(grid, pool, inter_rater,
                                 args.tables_dir / "per_method_alpha_cp.tex")

    print("\n### Honest human ceiling — ρ(group1, group2) and α(group1, group2)\n")
    print("Computed on the same apples-to-apples key subset as every other row.\n")
    header = "| stat | " + " | ".join(METHODS) + " | **all** |"
    sep = "|---|" + ":---:|" * len(METHODS) + "---:|"
    print(header)
    print(sep)
    cells_rho = [_fmt_rho(inter_rater["per_method"][m]["rho"]) for m in METHODS]
    cells_a = [_fmt_rho(inter_rater["per_method"][m]["alpha"]) for m in METHODS]
    cells_n = [str(inter_rater["per_method"][m]["n"]) for m in METHODS]
    print(f"| ρ(g1, g2) | " + " | ".join(cells_rho)
          + f" | **{_fmt_rho(inter_rater['pooled']['rho']).strip()}** |")
    print(f"| α(g1, g2) | " + " | ".join(cells_a)
          + f" | **{_fmt_rho(inter_rater['pooled']['alpha']).strip()}** |")
    print(f"| n | " + " | ".join(cells_n) + f" | {inter_rater['pooled']['n']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
