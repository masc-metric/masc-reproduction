"""Reproduce DreamBench++ Table 3 (Human Alignment Degree, Kd_o).

Goal: numerical match with the paper's CP / PF Kd_o columns (H-H, G-H,
D-H, C-H) for every rating shipped in `data/dreambench_plus/`. No MaSC
matchers here — this is a pure reproduction check.

Formulas (from the DreamBench++ eval code + inferred preprocessing for
D-H / C-H, which is not in the shared snippet):

  H-H = α(g1, g2)                                       — 2-rater, raw 0-4.
  G-H = mean_i[ mean_j[ α(gpt_i, g_j) ] ] ± std_i       — i ∈ {full, full2, full3},
                                                          j ∈ {g1, g2}.
  D-H = mean_j[ α(dino_mm, g_j /4) ]                    — global min-max on DINO.
  C-H = mean_j[ α(clip_mm, g_j /4) ]                    — global min-max on CLIP.

All α calls use interval distance, 2-rater form, closed-form O(N) via
`D_o = mean((x - y)²)`, `D_e = 2 M Var(v) / (M − 1)`, `α = 1 − D_o / D_e`.
This avoids the krippendorff package's V×V coincidence matrix — O(V²)
memory, catastrophic on continuous scores (V ≈ N).

Min/max for D-H / C-H is computed *globally* across all 7 methods
(pooled DINO and CLIP-I / CLIP-T distributions), matching the paper's
per-cell numbers to within ~0.02. Per-method min/max gives a larger
residual; raw (no scaling) gives ~−0.9 everywhere.

Usage:
    python scripts/reproduce_dbplus_kdo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import DREAMBENCHPLUS_DIR  # noqa: E402

HUMAN_ROOT = DREAMBENCHPLUS_DIR / "data_human_rating" / "merged_data"
GPT_ROOT = DREAMBENCHPLUS_DIR / "data_gpt_rating"

METHODS: list[str] = [
    "textual_inversion_sd",
    "dreambooth_sd",
    "dreambooth_lora_sdxl",
    "blip_diffusion",
    "emu2",
    "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_g_sdxl",
]

# DreamBench++ Table 3 values for numerical comparison.
PAPER_CP = {
    "textual_inversion_sd":        {"H-H": 0.685, "G-H": (0.544, 0.014), "D-H":  0.262, "C-H": -0.030},
    "dreambooth_sd":               {"H-H": 0.647, "G-H": (0.596, 0.003), "D-H":  0.408, "C-H":  0.229},
    "dreambooth_lora_sdxl":        {"H-H": 0.656, "G-H": (0.641, 0.007), "D-H":  0.371, "C-H":  0.321},
    "blip_diffusion":              {"H-H": 0.613, "G-H": (0.362, 0.017), "D-H": -0.078, "C-H": -0.186},
    "emu2":                        {"H-H": 0.746, "G-H": (0.669, 0.005), "D-H":  0.518, "C-H":  0.258},
    "ip_adapter_plus_vit_h_sdxl":  {"H-H": 0.602, "G-H": (0.366, 0.017), "D-H": -0.141, "C-H": -0.150},
    "ip_adapter_vit_g_sdxl":       {"H-H": 0.591, "G-H": (0.458, 0.002), "D-H": -0.073, "C-H": -0.212},
}
PAPER_PF = {
    "textual_inversion_sd":        {"H-H": 0.475, "G-H": (0.461, 0.007), "C-H":  0.267},
    "dreambooth_sd":               {"H-H": 0.516, "G-H": (0.506, 0.002), "C-H":  0.185},
    "dreambooth_lora_sdxl":        {"H-H": 0.469, "G-H": (0.402, 0.001), "C-H":  0.022},
    "blip_diffusion":              {"H-H": 0.619, "G-H": (0.541, 0.003), "C-H":  0.319},
    "emu2":                        {"H-H": 0.441, "G-H": (0.422, 0.011), "C-H":  0.230},
    "ip_adapter_plus_vit_h_sdxl":  {"H-H": 0.576, "G-H": (0.484, 0.006), "C-H":  0.256},
    "ip_adapter_vit_g_sdxl":       {"H-H": 0.509, "G-H": (0.531, 0.006), "C-H":  0.196},
}


def _load_dict(path: Path) -> dict[str, float]:
    return {k: float(v) for k, v in json.loads(path.read_text()).items()}


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b))
    return (np.asarray([a[k] for k in keys], dtype=float),
            np.asarray([b[k] for k in keys], dtype=float))


def alpha_interval_arr(x: np.ndarray, y: np.ndarray) -> float:
    """2-rater interval Krippendorff's α, closed-form O(N)."""
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


def alpha_interval(a: dict[str, float], b: dict[str, float]) -> float:
    x, y = paired(a, b)
    return alpha_interval_arr(x, y)


def rating_path(method: str, signal: str, group: str) -> Path:
    suffix = "-cp.json" if signal == "cp" else "-pf.json"
    return HUMAN_ROOT / group / f"{method}{suffix}"


def gpt_path(variant: str, method: str) -> Path:
    return GPT_ROOT / variant / f"{method}.json"


def gpt_run_folders(signal: str) -> list[str]:
    base = "concept_preservation" if signal == "cp" else "prompt_following"
    return [f"{base}_full", f"{base}_full2", f"{base}_full3"]


def compute_global_minmax(signal: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pooled (across all 7 methods) min/max for DINO-I and CLIP-I / CLIP-T."""
    dino_vals: list[float] = []
    clip_vals: list[float] = []
    for m in METHODS:
        if signal == "cp":
            dino_vals += list(_load_dict(DREAMBENCHPLUS_DIR / "data_dino_rating" / f"{m}.json").values())
            clip_vals += list(_load_dict(DREAMBENCHPLUS_DIR / "data_clipi_rating" / f"{m}.json").values())
        else:
            clip_vals += list(_load_dict(DREAMBENCHPLUS_DIR / "data_clipt_rating" / f"{m}.json").values())
    dino_range = (min(dino_vals), max(dino_vals)) if dino_vals else (0.0, 1.0)
    clip_range = (min(clip_vals), max(clip_vals))
    return dino_range, clip_range


def compute_row(method: str, signal: str,
                dino_range: tuple[float, float],
                clip_range: tuple[float, float]) -> dict:
    """Compute H-H, G-H (mean±std across 3 GPT runs), D-H (CP only), C-H.

    Each is rounded to 3 decimals before averaging, matching the paper's
    `round(..., 3)` inside `kd_alpha` and again inside the group-averaging
    step.
    """
    g1 = _load_dict(rating_path(method, signal, "group1"))
    g2 = _load_dict(rating_path(method, signal, "group2"))

    # H-H: 2-rater α on raw 0-4 group means.
    hh = round(alpha_interval(g1, g2), 3)

    # G-H: for each of 3 GPT runs, mean over groups of α(gpt, g_j).
    gh_per_run: list[float] = []
    for variant in gpt_run_folders(signal):
        gpt = _load_dict(gpt_path(variant, method))
        a1 = round(alpha_interval(gpt, g1), 3)
        a2 = round(alpha_interval(gpt, g2), 3)
        gh_per_run.append(round((a1 + a2) / 2, 3))
    gh_mean = float(np.mean(gh_per_run))
    gh_std = float(np.std(gh_per_run, ddof=0))

    # Continuous baselines. Global min-max normalize, then human / 4.
    row = {
        "method": method,
        "H-H": hh, "G-H_mean": gh_mean, "G-H_std": gh_std,
        "gh_per_run": gh_per_run,
    }
    # Human scaled to [0, 1] by /4.
    g1_keys = sorted(g1); g2_keys = sorted(g2)
    g1s = {k: g1[k] / 4.0 for k in g1_keys}
    g2s = {k: g2[k] / 4.0 for k in g2_keys}

    if signal == "cp":
        dino = _load_dict(DREAMBENCHPLUS_DIR / "data_dino_rating" / f"{method}.json")
        lo, hi = dino_range
        dino_n = {k: (v - lo) / (hi - lo) for k, v in dino.items()}
        dh = round((round(alpha_interval(dino_n, g1s), 3)
                    + round(alpha_interval(dino_n, g2s), 3)) / 2, 3)
        row["D-H"] = dh

    clip_folder = "data_clipi_rating" if signal == "cp" else "data_clipt_rating"
    clip = _load_dict(DREAMBENCHPLUS_DIR / clip_folder / f"{method}.json")
    lo, hi = clip_range
    clip_n = {k: (v - lo) / (hi - lo) for k, v in clip.items()}
    ch = round((round(alpha_interval(clip_n, g1s), 3)
                + round(alpha_interval(clip_n, g2s), 3)) / 2, 3)
    row["C-H"] = ch
    return row


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:+.3f}±{std:.3f}"


def fmt(v: float) -> str:
    return "   n/a" if not np.isfinite(v) else f"{v:+.3f}"


def print_cp_table(rows: list[dict]) -> None:
    print("\n### CP reproduction vs DreamBench++ Table 3\n")
    print("| method | H-H ours | paper | G-H ours | paper | D-H ours | paper | C-H ours | paper |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    err_tot = {"H-H": 0.0, "G-H": 0.0, "D-H": 0.0, "C-H": 0.0}
    for row in rows:
        m = row["method"]; p = PAPER_CP[m]
        gh_ours = fmt_pm(row["G-H_mean"], row["G-H_std"])
        gh_paper = fmt_pm(p["G-H"][0], p["G-H"][1])
        print(f"| {m} | {fmt(row['H-H'])} | {p['H-H']:+.3f} | {gh_ours} | {gh_paper} "
              f"| {fmt(row['D-H'])} | {p['D-H']:+.3f} | {fmt(row['C-H'])} | {p['C-H']:+.3f} |")
        err_tot["H-H"] += abs(row["H-H"] - p["H-H"])
        err_tot["G-H"] += abs(row["G-H_mean"] - p["G-H"][0])
        err_tot["D-H"] += abs(row["D-H"] - p["D-H"])
        err_tot["C-H"] += abs(row["C-H"] - p["C-H"])
    print(f"\nTotal |error|: H-H={err_tot['H-H']:.3f}  "
          f"G-H={err_tot['G-H']:.3f}  D-H={err_tot['D-H']:.3f}  C-H={err_tot['C-H']:.3f}")


def print_pf_table(rows: list[dict]) -> None:
    print("\n### PF reproduction vs DreamBench++ Table 3\n")
    print("| method | H-H ours | paper | G-H ours | paper | C-H ours | paper |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    err_tot = {"H-H": 0.0, "G-H": 0.0, "C-H": 0.0}
    for row in rows:
        m = row["method"]; p = PAPER_PF[m]
        gh_ours = fmt_pm(row["G-H_mean"], row["G-H_std"])
        gh_paper = fmt_pm(p["G-H"][0], p["G-H"][1])
        print(f"| {m} | {fmt(row['H-H'])} | {p['H-H']:+.3f} | {gh_ours} | {gh_paper} "
              f"| {fmt(row['C-H'])} | {p['C-H']:+.3f} |")
        err_tot["H-H"] += abs(row["H-H"] - p["H-H"])
        err_tot["G-H"] += abs(row["G-H_mean"] - p["G-H"][0])
        err_tot["C-H"] += abs(row["C-H"] - p["C-H"])
    print(f"\nTotal |error|: H-H={err_tot['H-H']:.3f}  "
          f"G-H={err_tot['G-H']:.3f}  C-H={err_tot['C-H']:.3f}")


def main() -> int:
    cp_dino_range, cp_clip_range = compute_global_minmax("cp")
    pf_clip_range = compute_global_minmax("pf")[1]
    cp_rows = [compute_row(m, "cp", cp_dino_range, cp_clip_range) for m in METHODS]
    pf_rows = [compute_row(m, "pf", (0.0, 1.0), pf_clip_range) for m in METHODS]
    print_cp_table(cp_rows)
    print_pf_table(pf_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
