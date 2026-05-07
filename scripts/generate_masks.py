"""Generate per-image binary masks via a text-prompted segmenter.

Reads a YAML config that specifies the segmenter, input image root,
and where to write masks. The text prompt per image depends on
`input_kind`:

- `dreambooth`:           class from `prompts_and_classes.txt`.
- `dreambooth_outputs`:   same lookup, one extra path level.
- `dreambenchplus_refs`:  keyword read from first line of
                          `captions/<category>/<XX>.txt`.
- `dreambenchplus_samples`: keyword parsed from the per-subject directory
                          name (`<cat>_<subcat>_<XX>_<keyword>`); method
                          filter via `--method` CLI or `method:` in YAML.

Usage:
    python scripts/generate_masks.py --config configs/masks/sam3.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import (  # noqa: E402
    DREAMBENCHPLUS_METHOD_ALIASES, MASKS_DIR, REPO_ROOT,
    parse_dreambenchplus_subject,
)


def iter_images(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_segmenter_from_cfg(cfg: dict):
    name = cfg["segmenter"]
    if name == "sam3":
        from repro_eval.masks.sam3 import load_segmenter
        return load_segmenter(
            confidence_threshold=cfg.get("confidence_threshold", 0.5),
            union_threshold=cfg.get("union_threshold", 0.5),
        )
    if name == "clipseg":
        from repro_eval.masks.clipseg import load_segmenter
        return load_segmenter(
            model_id=cfg.get("model_id", "CIDAS/clipseg-rd64-refined"),
            threshold=cfg.get("threshold", 0.5),
        )
    if name == "grounded_sam2":
        from repro_eval.masks.grounded_sam2 import load_segmenter
        return load_segmenter(
            gdino_model_id=cfg.get("gdino_model_id", "IDEA-Research/grounding-dino-base"),
            sam_model_id=cfg.get("sam_model_id", "facebook/sam2.1-hiera-large"),
            box_threshold=cfg.get("box_threshold", 0.3),
            text_threshold=cfg.get("text_threshold", 0.25),
            mask_threshold=cfg.get("mask_threshold", 0.0),
        )
    if name == "owlv2_sam2":
        from repro_eval.masks.owlv2_sam2 import load_segmenter
        return load_segmenter(
            owl_model_id=cfg.get("owl_model_id", "google/owlv2-base-patch16-ensemble"),
            sam_model_id=cfg.get("sam_model_id", "facebook/sam2.1-hiera-large"),
            score_threshold=cfg.get("score_threshold", 0.2),
            mask_threshold=cfg.get("mask_threshold", 0.0),
        )
    raise ValueError(f"Unknown segmenter: {name}")


def load_dreambenchplus_keywords(captions_root: Path) -> dict[Path, str]:
    """Return `{<cat_path>/<idx>: keyword}` read from first line of each
    `captions/<category>/<XX>.txt` under DreamBench++.

    Used for `dreambenchplus_refs` where each ref image is at
    `<category>/<XX>.jpg` and the keyword comes from the matching
    caption file's first line.
    """
    mapping: dict[Path, str] = {}
    for txt in sorted(captions_root.rglob("*.txt")):
        rel = txt.relative_to(captions_root).with_suffix("")  # <cat>/<idx>
        first_line = txt.read_text().splitlines()[0].strip()
        if first_line:
            mapping[rel] = first_line
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", type=str, default=None,
                    help="DreamBench++ method short alias (overrides config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    input_root = (REPO_ROOT / cfg["input_root"]).resolve()
    output_root = MASKS_DIR / cfg["output_subdir"]
    exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}
    overwrite = bool(cfg.get("overwrite", False))

    input_kind = cfg["input_kind"]

    # Build `jobs = [(image_path, out_mask_path, text_prompt), ...]` for
    # every input_kind; one unified processing loop below.
    jobs: list[tuple[Path, Path, str]] = []

    if input_kind == "dreambenchplus_refs":
        # input_root = data/dreambench_plus/images
        captions_root = input_root.parent / "captions"
        kw_by_path = load_dreambenchplus_keywords(captions_root)
        for img in iter_images(input_root, exts):
            rel = img.relative_to(input_root).with_suffix("")  # <cat>/<idx>
            keyword = kw_by_path.get(rel)
            if keyword is None:
                continue
            jobs.append((
                img,
                (output_root / rel).with_suffix(".png"),
                keyword,
            ))

    elif input_kind == "dreambenchplus_samples":
        # input_root = data/dreambench_plus/samples
        method = args.method or cfg.get("method")
        if method is None:
            raise ValueError("dreambenchplus_samples requires --method or `method:` in cfg")
        method_full = DREAMBENCHPLUS_METHOD_ALIASES.get(method, method)
        method_dir = input_root / method_full / "tgt_image"
        if not method_dir.exists():
            raise FileNotFoundError(f"No such method dir: {method_dir}")
        # Redirect output under the method name so methods don't collide.
        output_root = output_root / method_full
        for subject_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            _, _, keyword = parse_dreambenchplus_subject(subject_dir.name)
            # Replace underscores that were part of the keyword's original
            # spelling; e.g. "ice cream" was stored verbatim by the dataset.
            # Keywords appear unmodified in the text captions we saw.
            for img in sorted(subject_dir.iterdir()):
                if not img.is_file() or img.suffix.lower() not in exts:
                    continue
                out_rel = Path(subject_dir.name) / img.name
                jobs.append((
                    img,
                    (output_root / out_rel).with_suffix(".png"),
                    keyword,
                ))
        print(f"DreamBench++ method: {method} ({method_full})")

    else:
        raise NotImplementedError(f"input_kind={input_kind!r}")

    total = len(jobs)
    print(f"Found {total} images to segment")
    print(f"Writing masks to {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    segmenter = load_segmenter_from_cfg(cfg)

    n_done = n_skipped = 0
    t0 = time.time()
    for img_path, out_path, prompt in jobs:
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(img_path)
        mask = segmenter.segment(image, prompt)
        Image.fromarray(mask).save(out_path)
        pos_pct = 100.0 * (mask > 0).sum() / mask.size
        n_done += 1
        print(
            f"  [{n_done:>5}/{total}] {out_path.relative_to(output_root)}"
            f"  prompt={prompt!r}  pos={pos_pct:5.1f}%"
        )

    dt = time.time() - t0
    print(f"Done: wrote {n_done}, skipped {n_skipped} (existing) — {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
