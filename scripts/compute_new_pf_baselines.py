"""Pooled α / ρ for the 2026-04-28 PF baselines (VQAScore, ImageReward, HPSv3).

Reads:
    results/prompt_following__<scorer>__dreambenchplus_<method>.jsonl
    data/dreambench_plus/data_human_rating/merged_data/{group1,group2}/<method>-pf.json

Reports:
    - Pooled Kd_o α (paper formula) on the apples-to-apples shared keyset
    - Spearman ρ on the same subset
    - n per method, total n
    - Comparison row with our SigLIP2 BG-pool + nosubj winner

Same α recipe as `plot_dreambenchplus_cp.py` and `compare_nosubj_ablation.py`.

Usage:
    python scripts/compute_new_pf_baselines.py
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

from repro_eval.data import DREAMBENCHPLUS_DIR, RESULTS_DIR  # noqa: E402

METHODS = [
    "dreambooth_sd", "dreambooth_lora_sdxl", "textual_inversion_sd",
    "blip_diffusion", "emu2",
    "ip_adapter_plus_vit_h_sdxl", "ip_adapter_vit_g_sdxl",
]

PF_RATINGS_ROOT = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"

# (display_name, scorer_filename_token, run_tag)
PF_NEW_BASELINES = [
    ("VQAScore (clip-flant5-xxl)",          "vqascore",     "dreambenchplus"),
    ("ImageReward (BLIP, NeurIPS 2023)",    "imagereward",  "dreambenchplus"),
    ("HPSv3 (Qwen2-VL 7B, 2025)",           "hpsv3",        "dreambenchplus"),
]

# Reference rows (already scored, recomputed here on the matched keyset).
PF_REFERENCE_ROWS = [
    # (name, kind, *args).  "ours_pf" → results/prompt_following__siglip2-so400m__<tag>_<method>.jsonl
    # "dbplus" → data/dreambench_plus/data_<folder>/<method>.json
    # "dbplus_gpt" → data/dreambench_plus/data_gpt_rating/<variant>/<method>.json (integer 0-4)
    ("SigLIP2 SO400M-NaFlex BG-pool + nosubj (ours)", "ours_pf", "dreambenchplus_global_bg_nosubj"),
    ("SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)", "ours_pf", "dreambenchplus_naflex"),
    ("CLIP-T (DB++ baseline)", "dbplus", "data_clipt_rating"),
    ("GPT-4o PF (no internal thinking)", "dbplus_gpt", "prompt_following_wo_internal_thinking"),
    ("GPT-4V PF (full)", "dbplus_gpt", "prompt_following_gpt4v_full"),
]


def load_scorer(scorer: str, run_tag: str, method: str) -> dict[str, float]:
    path = RESULTS_DIR / f"prompt_following__{scorer}__{run_tag}_{method}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            s = float(r["score"])
            if s != s:
                continue
            out[f"{r['concept_id']}-{r['prompt_id']}"] = s
    return out


def load_ours_pf(tag: str, method: str) -> dict[str, float]:
    return load_scorer("siglip2-so400m", tag, method)


def load_dbplus(folder: str, _unused: str, method: str) -> dict[str, float]:
    path = DREAMBENCHPLUS_DIR / folder / f"{method}.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def load_dbplus_gpt(variant: str, _unused: str, method: str) -> dict[str, float]:
    """Load a DreamBench++-shipped GPT-judge PF rating JSON (integer 0–4)."""
    path = DREAMBENCHPLUS_DIR / "data_gpt_rating" / variant / f"{method}.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.load(path.open()).items()}


def load_pf_group(method: str, group: str) -> dict[str, float]:
    return {k: float(v) for k, v in json.load(
        (PF_RATINGS_ROOT / group / f"{method}-pf.json").open()).items()}


def alpha_interval(x: np.ndarray, y: np.ndarray) -> float:
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


def alpha_pooled(row: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float:
    """Continuous-row α: global min-max to [0, 1], then average paper Kd_o."""
    lo, hi = float(row.min()), float(row.max())
    if hi == lo:
        return float("nan")
    x = (row - lo) / (hi - lo)
    a1 = round(alpha_interval(x, g1 / 4.0), 3)
    a2 = round(alpha_interval(x, g2 / 4.0), 3)
    return round((a1 + a2) / 2.0, 3)


def main() -> int:
    # Load all rows (new baselines + reference rows). `ordinal[name]` is
    # True for rows already on the integer 0–4 rubric (GPT judges); those
    # use /4 normalization for α instead of min-max — same convention as
    # plot_dreambenchplus_cp.py.
    rows: dict[str, dict[str, dict[str, float]]] = {}
    ordinal: dict[str, bool] = {}
    for name, scorer, tag in PF_NEW_BASELINES:
        rows[name] = {m: load_scorer(scorer, tag, m) for m in METHODS}
        ordinal[name] = False
    for name, kind, arg in PF_REFERENCE_ROWS:
        if kind == "ours_pf":
            rows[name] = {m: load_ours_pf(arg, m) for m in METHODS}
            ordinal[name] = False
        elif kind == "dbplus":
            rows[name] = {m: load_dbplus(arg, "", m) for m in METHODS}
            ordinal[name] = False
        elif kind == "dbplus_gpt":
            rows[name] = {m: load_dbplus_gpt(arg, "", m) for m in METHODS}
            ordinal[name] = True

    # Per-method shared-keys: intersect all rows + both PF rating groups.
    g1_per = {m: load_pf_group(m, "group1") for m in METHODS}
    g2_per = {m: load_pf_group(m, "group2") for m in METHODS}

    per_method_n: list[tuple[str, int]] = []
    arrs_per_row: dict[str, list[np.ndarray]] = {n: [] for n in rows}
    g1_cat_list: list[np.ndarray] = []
    g2_cat_list: list[np.ndarray] = []
    for m in METHODS:
        shared = set(g1_per[m]) & set(g2_per[m])
        for name in rows:
            shared &= rows[name][m].keys()
        shared = sorted(shared)
        per_method_n.append((m, len(shared)))
        if not shared:
            continue
        for name in rows:
            arrs_per_row[name].append(
                np.asarray([rows[name][m][k] for k in shared], dtype=float)
            )
        g1_cat_list.append(np.asarray([g1_per[m][k] for k in shared], dtype=float))
        g2_cat_list.append(np.asarray([g2_per[m][k] for k in shared], dtype=float))

    g1_cat = np.concatenate(g1_cat_list)
    g2_cat = np.concatenate(g2_cat_list)
    mean_cat = (g1_cat + g2_cat) / 2.0

    print("\n=== Per-method shared-key counts (all-row intersection) ===")
    for m, n in per_method_n:
        print(f"  {m:<32} n = {n}")
    total = sum(n for _, n in per_method_n)
    print(f"  total                            n = {total}")

    # Honest human ceiling — α(group1, group2) and ρ(group1, group2) on the
    # same apples-to-apples keyset every other row uses. Both groups are on
    # the integer 0–4 rubric, so /4 to map to [0, 1] before α (paper conv).
    a_human = round(alpha_interval(g1_cat / 4.0, g2_cat / 4.0), 3)
    rho_human = float(spearmanr(g1_cat, g2_cat).statistic)
    print(f"\n=== Honest human ceiling (PF inter-rater) ===")
    print(f"  α(g1, g2) = {a_human:+.3f}   ρ(g1, g2) = {rho_human:+.3f}   n = {len(g1_cat)}")

    print("\n=== Pooled α + ρ vs DB++ PF mean rating, sorted by α ===")
    print(f"{'row':<58} {'α':>8} {'ρ':>8}  n")
    rendered = []
    for name in rows:
        row_cat = np.concatenate(arrs_per_row[name])
        if ordinal[name]:
            # Integer 0–4: scale all three rows to [0, 1] by /4 (paper convention).
            a1 = round(alpha_interval(row_cat / 4.0, g1_cat / 4.0), 3)
            a2 = round(alpha_interval(row_cat / 4.0, g2_cat / 4.0), 3)
            a = round((a1 + a2) / 2.0, 3)
        else:
            a = alpha_pooled(row_cat, g1_cat, g2_cat)
        rho = float(spearmanr(row_cat, mean_cat).statistic)
        rendered.append((name, a, rho, len(row_cat)))
    rendered.sort(key=lambda t: (np.isnan(t[1]), -t[1] if not np.isnan(t[1]) else 1.0))
    for name, a, rho, n in rendered:
        marker = "★" if "(ours)" in name else " "
        print(f"{marker} {name:<56} {a:>+8.3f} {rho:>+8.3f}  {n}")

    # Per-method α breakdown for the new baselines (paper-table format).
    print("\n=== Per-method α for new PF baselines ===")
    print(f"{'method':<32} " + " ".join(f"{name[:18]:>18}" for name, _, _ in PF_NEW_BASELINES) + "   n")
    for mi, m in enumerate(METHODS):
        cells = []
        for name, _, _ in PF_NEW_BASELINES:
            arr = arrs_per_row[name][mi] if mi < len(arrs_per_row[name]) else np.asarray([])
            g1m = g1_cat_list[mi] if mi < len(g1_cat_list) else np.asarray([])
            g2m = g2_cat_list[mi] if mi < len(g2_cat_list) else np.asarray([])
            if len(arr) < 2:
                cells.append("    n/a")
                continue
            a = alpha_pooled(arr, g1m, g2m)
            cells.append(f"{a:>+8.3f}")
        n = per_method_n[mi][1]
        print(f"{m:<32} " + " ".join(f"{c:>18}" for c in cells) + f"   {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
