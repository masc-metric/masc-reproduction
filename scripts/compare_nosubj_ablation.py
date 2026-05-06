"""Subject-stripping ablation on so400m-naflex PF.

Six-cell PF ablation grid: pool ∈ {full, BG, FG} × prompt ∈ {full, strip}.
All variants share the so400m-naflex backbone and forward pass; only the
attention-pool key-padding mask and prompt-side processing change.

    full × full         — full-image cosine, full prompt        (baseline)
    full × strip        — full-image cosine, subject stripped   (prompt-side ablation)
    BG-pool × full      — BG-pool cosine, full prompt           (image-side ablation)
    BG-pool × strip     — BG-pool cosine, subject stripped      (the proposed metric — MaSC PF)
    FG-pool × full      — FG-pool cosine, full prompt           (control: invert the mask)
    FG-pool × strip     — FG-pool cosine, subject stripped      (control × control — not yet run)

Isolates (a) whether the lift in BG-pool×strip comes from the BG-pool ×
nosubj interaction (both needed) or from nosubj alone, and (b) whether
the FG-pool inverse control is α-negative as the §5.3 prose claims.

Reports pooled Kd_o α + ρ on the shared-key subset (4-variant intersection
on all 7 methods); FG-pool rows use the same keyset further restricted
to the 6 methods FG-pool covers (BLIP-D was not run for the FG-pool
config). FG × strip was never run; rendered as "—".

Outputs:
    - markdown stdout
    - LaTeX `tabular` block at report/tables/pf_2x3_ablation.tex (paper-ready)

Uses the same α recipe as plot_dreambenchplus_cp.py / compare_clipt_dbplus.py.

Usage:
    python scripts/compare_nosubj_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import DREAMBENCHPLUS_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

METHODS = [
    "dreambooth_sd", "dreambooth_lora_sdxl", "textual_inversion_sd",
    "blip_diffusion", "emu2",
    "ip_adapter_plus_vit_h_sdxl", "ip_adapter_vit_g_sdxl",
]

PF_RATINGS_ROOT = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"


def load_ours(tag: str, method: str) -> dict[str, float]:
    """Load a so400m-naflex PF run for one (tag, method). Missing-file →
    empty dict (FG-pool was not run on every method)."""
    path = RESULTS_DIR / f"prompt_following__siglip2-so400m__{tag}_{method}.jsonl"
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


def load_pf_group(method: str, group: str) -> dict[str, float]:
    return {k: float(v) for k, v in json.load(
        (PF_RATINGS_ROOT / group / f"{method}-pf.json").open()).items()}


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


def alpha_pooled(row: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float:
    lo, hi = float(row.min()), float(row.max())
    if hi == lo:
        return float("nan")
    x = (row - lo) / (hi - lo)
    a1 = round(alpha_interval_arr(x, g1 / 4.0), 3)
    a2 = round(alpha_interval_arr(x, g2 / 4.0), 3)
    return round((a1 + a2) / 2.0, 3)


# 2×3 ablation grid. (display, tag-or-None). None tag = run not on disk.
# Order: full → BG → FG (rows), full prompt → strip (cols).
GRID_ROWS = [
    ("Full pool", "dreambenchplus_naflex",       "dreambenchplus_global_nosubj"),
    ("BG-pool",   "dreambenchplus_global_bg",    "dreambenchplus_global_bg_nosubj"),
    ("FG-pool",   "dreambenchplus_global_fg",    None),  # FG × strip never run
]


def _row_tags() -> dict[str, str | None]:
    """Flat {row_label → tag} mapping for grid traversal."""
    out: dict[str, str | None] = {}
    for pool_label, tag_full, tag_strip in GRID_ROWS:
        out[f"{pool_label} × full prompt"] = tag_full
        out[f"{pool_label} × strip"] = tag_strip
    return out


def _pool_alpha_rho(per_method: dict[str, dict[str, dict[str, float]]],
                    tag_label: str,
                    g1_per: dict[str, dict[str, float]],
                    g2_per: dict[str, dict[str, float]],
                    keyset_filter: set[str] | None = None,
                    ) -> tuple[float, float, int]:
    """Compute pooled α + ρ for one row across all methods it has data for.

    `keyset_filter` (optional) restricts to a precomputed apples-to-apples
    keyset (e.g. the 4-variant intersection). Returns (α, ρ, n).
    """
    rows: list[np.ndarray] = []
    g1s: list[np.ndarray] = []
    g2s: list[np.ndarray] = []
    for m in METHODS:
        scores = per_method[tag_label][m]
        if not scores:
            continue
        g1 = g1_per[m]
        g2 = g2_per[m]
        keys = set(scores) & set(g1) & set(g2)
        if keyset_filter is not None:
            keys &= keyset_filter
        keys = sorted(keys)
        if not keys:
            continue
        rows.append(np.asarray([scores[k] for k in keys], dtype=float))
        g1s.append(np.asarray([g1[k] for k in keys], dtype=float))
        g2s.append(np.asarray([g2[k] for k in keys], dtype=float))
    if not rows:
        return float("nan"), float("nan"), 0
    row_cat = np.concatenate(rows)
    g1_cat = np.concatenate(g1s)
    g2_cat = np.concatenate(g2s)
    mean_cat = (g1_cat + g2_cat) / 2.0
    a = alpha_pooled(row_cat, g1_cat, g2_cat)
    rho = float(spearmanr(row_cat, mean_cat).statistic)
    return a, rho, int(len(row_cat))


def _fmt_signed(v: float) -> str:
    return "  n/a " if not np.isfinite(v) else f"{v:+.3f}"


def _latex_signed(v: float) -> str:
    return "---" if not np.isfinite(v) else f"{v:+.3f}".replace("-", "$-$")


def write_pf_2x3_latex(cells: dict[str, dict[str, tuple[float, float, int]]],
                        out_path: Path, n_main: int) -> None:
    """Emit Tab. S2 (PF FG/BG/full × strip/no-strip ablation) as a LaTeX block.

    `cells[pool_label][prompt_label] = (α, ρ, n)`. Pool labels: 'Full pool',
    'BG-pool', 'FG-pool'. Prompt labels: 'full prompt', 'strip'. Missing
    runs (FG × strip) are encoded with `α = NaN, n = 0` and rendered as
    `---`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def cellfmt(triple: tuple[float, float, int]) -> str:
        a, rho, n = triple
        if not np.isfinite(a):
            return "---"
        # Bold MaSC = BG-pool × strip cell — handled at row-construction time.
        return f"{_latex_signed(a)}"

    # Cells:
    full_full = cells["Full pool"]["full prompt"]
    full_strip = cells["Full pool"]["strip"]
    bg_full = cells["BG-pool"]["full prompt"]
    bg_strip = cells["BG-pool"]["strip"]
    fg_full = cells["FG-pool"]["full prompt"]
    fg_strip = cells["FG-pool"]["strip"]

    lines = [
        r"% Auto-generated by scripts/compare_nosubj_ablation.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"    \centering",
        rf"    \caption{{Prompt Following ablation on the same SigLIP2~SO400M-NaFlex backbone: "
        rf"image-side pooling ($\{{$full, BG, FG$\}}$) $\times$ prompt-side subject stripping. "
        rf"Pooled Krippendorff $\alpha$ (Kd$_o$) against pooled DB++ human PF on the shared "
        rf"${n_main:,}$-key subset (style excluded). The \textbf{{BG-pool $\times$ strip}} cell "
        rf"is \textbf{{MaSC}} (Tab.~\ref{{tab:pf_dbplus}}). The \textbf{{FG-pool}} control row "
        rf"backs the \S5.3 prose: pooling \emph{{onto}} the subject (the inverse of the proposed "
        rf"recipe) collapses the signal — $\alpha$ is actively negative. FG~$\times$~strip was "
        rf"not run; remaining FG cell uses the 4-variant keyset further restricted to the 6 "
        rf"methods FG-pool covers (BLIP-D omitted).}}",
        r"    \label{tab:pf_2x3_ablation}",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        Pool $\backslash$ Prompt & full prompt $\alpha$ & subject-stripped $\alpha$ \\",
        r"        \midrule",
    ]

    def row_line(pool_label: str, c_full: tuple[float, float, int],
                 c_strip: tuple[float, float, int], emphasize_strip: bool = False) -> str:
        c1 = _latex_signed(c_full[0])
        if emphasize_strip and np.isfinite(c_strip[0]):
            c2 = rf"\textbf{{{_latex_signed(c_strip[0])}}}"
        else:
            c2 = _latex_signed(c_strip[0])
        return rf"        {pool_label} & {c1} & {c2} \\"

    lines.append(row_line("Full pool (no mask)", full_full, full_strip))
    lines.append(row_line(r"BG-pool \,(\textbf{MaSC})", bg_full, bg_strip,
                          emphasize_strip=True))
    lines.append(row_line("FG-pool (inverse-mask control)", fg_full, fg_strip))
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "report/tables",
                    help="Where to write LaTeX `tabular` files for the paper.")
    args = ap.parse_args()

    row_tags = _row_tags()

    # Load every cell that has a tag (skip None — those are unrun cells).
    per_method: dict[str, dict[str, dict[str, float]]] = {}
    for name, tag in row_tags.items():
        if tag is None:
            per_method[name] = {m: {} for m in METHODS}
            continue
        per_method[name] = {m: load_ours(tag, m) for m in METHODS}

    g1_per = {m: load_pf_group(m, "group1") for m in METHODS}
    g2_per = {m: load_pf_group(m, "group2") for m in METHODS}

    # Apples-to-apples keyset for the 4 fully-covered cells (full × *, BG × *):
    # per-method intersection of those 4 variants ∩ (g1 ∩ g2).
    main_4 = ["Full pool × full prompt", "Full pool × strip",
              "BG-pool × full prompt", "BG-pool × strip"]
    main_keyset_per_method: dict[str, set[str]] = {}
    n_per_method_main: list[tuple[str, int]] = []
    for m in METHODS:
        shared = set(g1_per[m]) & set(g2_per[m])
        for n in main_4:
            shared &= per_method[n][m].keys()
        main_keyset_per_method[m] = shared
        n_per_method_main.append((m, len(shared)))
    main_keyset = {f"{m}|{k}" for m, ks in main_keyset_per_method.items() for k in ks}

    # `_pool_alpha_rho` operates on per-method dicts; build a thin keyset filter
    # by passing per-method sets through the keyset_filter argument indirectly.
    # Simpler: precompute the filtered scores in-place for the 4 main cells.
    def _restricted(name: str) -> dict[str, dict[str, float]]:
        out = {}
        for m in METHODS:
            ks = main_keyset_per_method[m]
            out[m] = {k: per_method[name][m][k] for k in per_method[name][m] if k in ks}
        return out

    restricted: dict[str, dict[str, dict[str, float]]] = {}
    for n in main_4:
        restricted[n] = _restricted(n)

    # FG × full prompt: same per-method keyset, but BLIP-D drops out (FG never
    # ran for it). FG × strip: never run, leave as (NaN, NaN, 0).
    restricted["FG-pool × full prompt"] = _restricted("FG-pool × full prompt")
    restricted["FG-pool × strip"] = {m: {} for m in METHODS}

    # ---- Reporting -----------------------------------------------------

    print("\n=== Per-method shared-key counts (style excluded, 4-variant intersection) ===")
    for m, n in n_per_method_main:
        print(f"  {m:<32} n = {n}")
    total_main = sum(n for _, n in n_per_method_main)
    print(f"  total (4-variant keyset)         n = {total_main}")

    # Pooled α / ρ per cell.
    cell_stats: dict[str, dict[str, tuple[float, float, int]]] = {
        "Full pool": {}, "BG-pool": {}, "FG-pool": {},
    }
    label_to_pool_prompt = {
        "Full pool × full prompt": ("Full pool", "full prompt"),
        "Full pool × strip":       ("Full pool", "strip"),
        "BG-pool × full prompt":   ("BG-pool",   "full prompt"),
        "BG-pool × strip":         ("BG-pool",   "strip"),
        "FG-pool × full prompt":   ("FG-pool",   "full prompt"),
        "FG-pool × strip":         ("FG-pool",   "strip"),
    }
    for cell_label, (pool, prompt) in label_to_pool_prompt.items():
        if cell_label == "FG-pool × strip":
            cell_stats[pool][prompt] = (float("nan"), float("nan"), 0)
            continue
        a, rho, n = _pool_alpha_rho(restricted, cell_label, g1_per, g2_per)
        cell_stats[pool][prompt] = (a, rho, n)

    print("\n=== Pooled α + ρ — 2×3 ablation (FG × strip not yet run) ===")
    print(f"{'pool':<14} {'prompt':<14} {'α':>8} {'ρ':>8} {'n':>7}")
    for pool in ["Full pool", "BG-pool", "FG-pool"]:
        for prompt in ["full prompt", "strip"]:
            a, rho, n = cell_stats[pool][prompt]
            print(f"{pool:<14} {prompt:<14} {_fmt_signed(a):>8} {_fmt_signed(rho):>8} {n:>7}")

    # Markdown table — easier to drop into REPORT.md.
    print("\n### PF ablation grid (pooled α): pool × prompt-stripping\n")
    print("| pool \\ prompt | full prompt | subject-stripped |")
    print("|---|---:|---:|")
    for pool in ["Full pool (no mask)", "BG-pool (MaSC if + strip)", "FG-pool (inverse control)"]:
        key = pool.split(" ")[0] if pool.startswith("Full") else pool.split("-pool")[0] + "-pool"
        # Map display label → cell key.
        if pool.startswith("Full"):
            k = "Full pool"
        elif pool.startswith("BG"):
            k = "BG-pool"
        else:
            k = "FG-pool"
        a_full = cell_stats[k]["full prompt"][0]
        a_strip = cell_stats[k]["strip"][0]
        n_full = cell_stats[k]["full prompt"][2]
        n_strip = cell_stats[k]["strip"][2]
        c1 = f"{a_full:+.3f} (n={n_full})" if np.isfinite(a_full) else "—"
        c2 = f"{a_strip:+.3f} (n={n_strip})" if np.isfinite(a_strip) else "— (not run)"
        if k == "BG-pool":
            c2 = f"**{c2}**"
        print(f"| {pool} | {c1} | {c2} |")

    # Interaction tests on the original 2×2 (full × BG × prompt × strip).
    a_ff = cell_stats["Full pool"]["full prompt"][0]
    a_fs = cell_stats["Full pool"]["strip"][0]
    a_bf = cell_stats["BG-pool"]["full prompt"][0]
    a_bs = cell_stats["BG-pool"]["strip"][0]
    print("\n=== Interaction test (BG × strip vs the simpler ablations) ===")
    print(f"  nosubj lift on full pool: Δα = {a_fs - a_ff:+.3f}")
    print(f"  nosubj lift on BG-pool:   Δα = {a_bs - a_bf:+.3f}")
    print(f"  interaction (BG-specific lift): Δα = {(a_bs - a_bf) - (a_fs - a_ff):+.3f}")

    # FG-pool sanity: §5.3 prose claims FG-pool is α-negative. Print the gap.
    a_fg = cell_stats["FG-pool"]["full prompt"][0]
    n_fg = cell_stats["FG-pool"]["full prompt"][2]
    print(f"\n=== FG-pool inverse-control check ({n_fg} keys, 6 methods — BLIP-D not run) ===")
    print(f"  α(FG-pool × full prompt) = {a_fg:+.3f}   "
          f"(§5.3 prose: 'actively α-negative' — {'CONFIRMED' if a_fg < 0 else 'NOT CONFIRMED'})")
    print(f"  Δα(FG-pool − full pool) = {a_fg - a_ff:+.3f}")
    print(f"  Δα(FG-pool − BG-pool)   = {a_fg - a_bf:+.3f}")

    # LaTeX table for the paper supplementary.
    write_pf_2x3_latex(cell_stats, args.tables_dir / "pf_2x3_ablation.tex",
                       n_main=total_main)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
