"""5×5 contact sheet of ORIDa training subjects across environments.

Picks the first 5 train subjects (by integer id) that have ≥5 unique
factual_counterfactual backgrounds. For each, picks the 5 alphabetically
first background prefixes; within each prefix picks the alphabetically
first scene_id (camera angle 0); within that scene picks the
alphabetically second image — `_0` is the counterfactual (no object),
`_1` is the first factual placement.

Writes:
    report/figures/orida_subjects_grid.png

Usage:
    python scripts/plot_orida_subjects_grid.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = _REPO_ROOT / "data" / "ORIDa" / "ORIDa_v1.0" / "train"
FIGURES_DIR = _REPO_ROOT / "report" / "figures"

N_SUBJECTS = 5
N_ENVIRONMENTS = 5


def select_subjects(n: int) -> list[Path]:
    out = []
    for subj_dir in sorted(TRAIN_ROOT.iterdir(),
                           key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
        if not subj_dir.is_dir():
            continue
        fc = subj_dir / "factual_counterfactual"
        if not fc.is_dir():
            continue
        prefixes = Counter(p.name[:-1] for p in fc.iterdir())
        if len(prefixes) >= N_ENVIRONMENTS:
            out.append(subj_dir)
        if len(out) >= n:
            break
    return out


def pick_image(subj_dir: Path, prefix: str) -> Path | None:
    """First scene by prefix (alphabetical → cam_idx '0'), second image
    in that scene's images dir (alphabetical → first factual placement)."""
    fc = subj_dir / "factual_counterfactual"
    matching_scenes = sorted(p for p in fc.iterdir() if p.name.startswith(prefix))
    if not matching_scenes:
        return None
    imgs = sorted((matching_scenes[0] / "images").iterdir())
    return imgs[1] if len(imgs) >= 2 else None


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subjects = select_subjects(N_SUBJECTS)
    print(f"Subjects: {[s.name for s in subjects]}")

    fig, axes = plt.subplots(N_SUBJECTS, N_ENVIRONMENTS,
                             figsize=(2.4 * N_ENVIRONMENTS, 2.4 * N_SUBJECTS))
    for r, subj_dir in enumerate(subjects):
        fc = subj_dir / "factual_counterfactual"
        prefixes = sorted({p.name[:-1] for p in fc.iterdir()})[:N_ENVIRONMENTS]
        print(f"  subj {subj_dir.name}: prefixes {prefixes}")
        for c, prefix in enumerate(prefixes):
            ax = axes[r, c] if N_SUBJECTS > 1 else axes[c]
            img_path = pick_image(subj_dir, prefix)
            if img_path is None:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            img = Image.open(img_path)
            ax.imshow(np.asarray(img))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"env {prefix}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"subj {subj_dir.name}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=20)

    fig.suptitle("ORIDa train: 5 subjects × 5 environments\n"
                 "(first cam angle per environment, first factual placement)",
                 fontsize=11)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    out_path = FIGURES_DIR / "orida_subjects_grid.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
