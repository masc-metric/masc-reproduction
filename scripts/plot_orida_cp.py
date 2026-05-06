"""Aggregate CP-metric runs on ORIDa train (50 subjects × 10 backgrounds × 1 photo).

ORIDa (https://arxiv.org/abs/2506.08964) is a real-captured object-
compositing dataset; we repurpose it as the citable real-photo CP
benchmark (replacing the iPhone-photo `data/real_data/` set used in
prior REPORT entries). Same statistic (AUC within > cross), bigger n,
published-and-citable.

Pipeline: `build_orida_diffbg_samples` picks 1 photo per (subject,
background) — alphabetically first cam angle, alphabetically first
factual placement (`_1`). Within-pairs are all C(10, 2) = 45 per
subject (each pair guaranteed to span different physical environments).
Cross-pairs are random uniform over (a < b) × pool_a × pool_b, matched
to within count.

Loads `results/geometry__*__orida_train_50x10.jsonl`, separates rows by
`gen_method` (`orida_within` / `orida_cross`), reports mean/std, Δ,
Δ_norm (per-metric min-max → [0, 1]), and AUC(within > cross).

Writes:
    report/figures/orida_cp_table.md
    report/figures/orida_cp_bars.png
    report/figures/orida_cp_violins.png

Usage:
    python scripts/plot_orida_cp.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
FIGURES_DIR = _REPO_ROOT / "report" / "figures"

# Display label, result-file stem, and (optional) score divisor per metric.
# Per-metric min-max in summarize() handles Δ_norm so different score
# ranges (e.g. cosine ∈ [-1, 1] vs DINO-I ∈ [-0.15, +0.98]) become
# directly comparable.
METRICS = [
    ("Our metric (SigLIP2 SO400M-NaFlex masked-maxcos)",
     "geometry__siglip2-so400m__orida_train_50x10.jsonl", 1.0),
    ("AM-RADIO C-RADIOv4-SO400M summary cosine (canonical baseline)",
     "geometry__C-RADIOv4__orida_train_50x10.jsonl", 1.0),
    ("DreamSim (DINO+CLIP+OpenCLIP ensemble, NIGHTS-finetuned)",
     "geometry__dreamsim-ensemble__orida_train_50x10.jsonl", 1.0),
    ("SigLIP2 SO400M-NaFlex global pool (same-backbone ablation)",
     "geometry__siglip2-so400m__orida_train_50x10_global_naflex.jsonl", 1.0),
    ("DINOv3 ViT-L/16 CLS cosine (modern DINO baseline)",
     "geometry__dinov3-vitl16__orida_train_50x10_dbplus.jsonl", 1.0),
    ("DIFT-SDXL canonical (paper hyperparameters, masked-maxcos)",
     "geometry__dift-sdxl__orida_train_50x10_canonical.jsonl", 1.0),
    ("DINO-I (DB++ orig: dino_vits8 CLS cosine)",
     "geometry__dino-vits8__orida_train_50x10.jsonl", 1.0),
    ("CLIP-I (DB++ orig: ViT-B/32 proj-head cosine)",
     "geometry__clip-vit__orida_train_50x10.jsonl", 1.0),
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def auc_within_over_cross(within: np.ndarray, cross: np.ndarray) -> float:
    """P(within > cross) over random (w, c) pairs; ties count as 0.5."""
    if within.size == 0 or cross.size == 0:
        return float("nan")
    all_scores = np.concatenate([within, cross])
    labels = np.concatenate([np.ones(within.size), np.zeros(cross.size)])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, all_scores.size + 1, dtype=float)
    _, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(counts, dtype=float)
    np.add.at(sum_ranks, inverse, ranks)
    mean_ranks = sum_ranks / counts
    ranks = mean_ranks[inverse]

    r_pos = ranks[labels == 1].sum()
    n_pos = within.size
    n_neg = cross.size
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(rows: list[dict], scale: float = 1.0) -> dict:
    within = np.array([r["score"] for r in rows if r["gen_method"] == "orida_within"]) / scale
    cross = np.array([r["score"] for r in rows if r["gen_method"] == "orida_cross"]) / scale
    all_scores = np.concatenate([within, cross]) if within.size and cross.size else within
    lo, hi = (float(all_scores.min()), float(all_scores.max())) if all_scores.size else (0.0, 1.0)
    span = max(hi - lo, 1e-12)
    within_norm = (within - lo) / span
    cross_norm = (cross - lo) / span
    return {
        "n_within": within.size,
        "n_cross": cross.size,
        "mean_within": float(within.mean()) if within.size else float("nan"),
        "std_within": float(within.std()) if within.size else float("nan"),
        "mean_cross": float(cross.mean()) if cross.size else float("nan"),
        "std_cross": float(cross.std()) if cross.size else float("nan"),
        "delta": (float(within.mean()) - float(cross.mean()))
                 if within.size and cross.size else float("nan"),
        "delta_norm": (float(within_norm.mean()) - float(cross_norm.mean()))
                      if within.size and cross.size else float("nan"),
        "score_min": lo,
        "score_max": hi,
        "auc": auc_within_over_cross(within, cross),
        "within_raw": within,
        "cross_raw": cross,
    }


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, dict]] = []
    for label, fname, scale in METRICS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[skip] missing: {path.name}")
            continue
        summaries.append((label, summarize(load_rows(path), scale=scale)))

    if not summaries:
        print("No result files found")
        return 1

    header = (
        "| metric | n_within | n_cross | mean_within | mean_cross "
        "| Δ (within−cross) | Δ_norm (min-max) | AUC(within>cross) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows_md = []
    for label, s in summaries:
        rows_md.append(
            f"| {label} | {s['n_within']} | {s['n_cross']} "
            f"| {s['mean_within']:.3f} ± {s['std_within']:.3f} "
            f"| {s['mean_cross']:.3f} ± {s['std_cross']:.3f} "
            f"| **{s['delta']:+.3f}** | **{s['delta_norm']:+.3f}** "
            f"| **{s['auc']:.3f}** |"
        )
    table_md = header + "\n".join(rows_md) + "\n"

    (FIGURES_DIR / "orida_cp_table.md").write_text(table_md)
    print(table_md)

    # Footnote on rescaling for any non-1.0 divisor in METRICS.
    rescaled = [(lbl, sc) for (lbl, _, sc), (_, _) in zip(METRICS, summaries) if sc != 1.0]
    if rescaled:
        note = ("\n*Note: " + "; ".join(
            f"{lbl.split(' (')[0]} rescaled by 1/{sc:g} (rubric / divisor) "
            f"to map native scale to [0, 1]; AUC and ρ are invariant under "
            f"monotonic transformations."
            for lbl, sc in rescaled
        ) + "*\n")
        (FIGURES_DIR / "orida_cp_table.md").write_text(table_md + note)
        print(note)

    labels_short = [lbl.split(" (")[0] for lbl, _ in summaries]
    labels_short = [
        l.replace(" summary cosine", "").replace(" canonical", "")
        for l in labels_short
    ]
    means_within = [s["mean_within"] for _, s in summaries]
    means_cross = [s["mean_cross"] for _, s in summaries]
    stds_within = [s["std_within"] for _, s in summaries]
    stds_cross = [s["std_cross"] for _, s in summaries]

    n_w = summaries[0][1]["n_within"]
    n_c = summaries[0][1]["n_cross"]

    x = np.arange(len(summaries))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, means_within, w, yerr=stds_within, capsize=3,
           label="within-subject (same object)", color="#4E79A7")
    ax.bar(x + w / 2, means_cross, w, yerr=stds_cross, capsize=3,
           label="cross-subject (different objects)", color="#F28E2B")
    for i, s in enumerate(summaries):
        ax.text(i, max(means_within[i], means_cross[i]) + 0.03,
                f"Δ={s[1]['delta']:+.3f}\nAUC={s[1]['auc']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("mean score")
    ax.set_title(f"ORIDa CP (train, 50 subjects × 10 backgrounds × 1 photo)\n"
                 f"within (same object, different bg) vs cross (different objects): "
                 f"{n_w} within, {n_c} cross pairs")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "orida_cp_bars.png", dpi=130)
    plt.close(fig)

    # ---- Paper-grade violin plot ----
    # Keep original violin styling (mean line, classic blue/orange); paper polish
    # is in font, DPI, PDF output, no in-figure title, and an external legend.
    BLUE = "#4E79A7"
    ORANGE = "#F28E2B"

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(10, 4.2))

        ax.axvspan(-0.5, 0.5, color="#FFF3D6", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

        positions: list[float] = []
        all_data: list[np.ndarray] = []
        colors: list[str] = []
        for i, (_, s) in enumerate(summaries):
            positions.extend([i - 0.2, i + 0.2])
            all_data.extend([s["within_raw"], s["cross_raw"]])
            colors.extend([BLUE, ORANGE])

        parts = ax.violinplot(all_data, positions=positions, widths=0.35,
                              showmeans=True, showmedians=False, showextrema=False)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_edgecolor("black")
            body.set_alpha(0.75)

        ax.set_xticks(np.arange(len(summaries)))
        ax.set_xticklabels(labels_short, rotation=20, ha="right")
        ax.set_ylabel("CP score")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

        legend_handles = [
            Patch(facecolor=BLUE, edgecolor="black", alpha=0.75,
                  label="within-subject (same object, different background)"),
            Patch(facecolor=ORANGE, edgecolor="black", alpha=0.75,
                  label="cross-subject (different objects)"),
        ]
        ax.legend(handles=legend_handles,
                  loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=2, frameon=False)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "orida_cp_violins.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "orida_cp_violins.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"\nWrote:")
    print(f"  {FIGURES_DIR / 'orida_cp_table.md'}")
    print(f"  {FIGURES_DIR / 'orida_cp_bars.png'}")
    print(f"  {FIGURES_DIR / 'orida_cp_violins.png'}")
    print(f"  {FIGURES_DIR / 'orida_cp_violins.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
