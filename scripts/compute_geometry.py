"""Compute geometry (concept-fidelity) signal over (ref, out, ref_mask, out_mask).

Writes one JSONL row per sample to
`results/geometry__<model>__<tag>.jsonl`.

Usage:
    python scripts/compute_geometry.py --config configs/geometry/dinov3.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repro_eval.data import REPO_ROOT, RESULTS_DIR, SAMPLE_BUILDERS  # noqa: E402


def load_matcher(cfg: dict):
    name = cfg["matcher"]
    if name == "dinov3":
        from repro_eval.geometry.dinov3 import load_matcher
        return load_matcher(
            model_id=cfg["model_id"], input_size=cfg["input_size"],
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "recall_fg"),
        )
    if name == "siglip2":
        from repro_eval.geometry.siglip2 import load_matcher
        return load_matcher(
            model_id=cfg["model_id"], input_size=cfg["input_size"],
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "recall_fg"),
            naflex_max_patches=cfg.get("naflex_max_patches", 1024),
            mask_blur_sigma=cfg.get("mask_blur_sigma", 0.0),
        )
    if name == "clip":
        from repro_eval.geometry.clip import load_matcher
        return load_matcher(
            model_id=cfg["model_id"], input_size=cfg["input_size"],
            fg_threshold=cfg.get("fg_threshold", 0.5),
        )
    if name == "clipi_dbplus":
        from repro_eval.geometry.clipi_dbplus import load_matcher
        return load_matcher(model_id=cfg["model_id"])
    if name == "dinoi_dbplus":
        from repro_eval.geometry.dinoi_dbplus import load_matcher
        return load_matcher(
            hub_repo=cfg.get("hub_repo", "facebookresearch/dino:main"),
            hub_model=cfg.get("hub_model", "dino_vits8"),
        )
    if name == "dinov3_dbplus":
        from repro_eval.geometry.dinov3_dbplus import load_matcher
        return load_matcher(
            model_id=cfg["model_id"],
            input_size=cfg.get("input_size", 224),
        )
    if name == "siglip2_global":
        from repro_eval.geometry.siglip2_global import load_matcher
        return load_matcher(
            model_id=cfg["model_id"],
            naflex_max_patches=cfg.get("naflex_max_patches", 1024),
        )
    if name == "dreamsim":
        from repro_eval.geometry.dreamsim import load_matcher
        return load_matcher(
            dreamsim_type=cfg.get("dreamsim_type", "ensemble"),
            cache_dir=cfg.get("cache_dir", ".cache/dreamsim"),
        )
    if name == "radio":
        from repro_eval.geometry.radio import load_matcher
        return load_matcher(
            model_id=cfg["model_id"], input_size=cfg["input_size"],
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "masked_maxcos"),
        )
    if name == "radio_summary":
        from repro_eval.geometry.radio_summary import load_matcher
        return load_matcher(
            model_id=cfg["model_id"],
            input_size=cfg.get("input_size", 512),
        )
    if name == "lightglue":
        from repro_eval.geometry.lightglue import load_matcher
        return load_matcher(
            max_num_keypoints=cfg.get("max_num_keypoints", 2048),
            fg_threshold=cfg.get("fg_threshold", 0.5),
        )
    if name == "loftr":
        from repro_eval.geometry.loftr import load_matcher
        return load_matcher(
            pretrained=cfg.get("pretrained", "outdoor"),
            input_size=cfg["input_size"],
            fg_threshold=cfg.get("fg_threshold", 0.5),
            confidence_threshold=cfg.get("confidence_threshold", 0.2),
            score_type=cfg.get("score_type", "recall_fg"),
        )
    if name == "roma":
        from repro_eval.geometry.roma import load_matcher
        return load_matcher(
            n_sample=cfg.get("n_sample", 5000),
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "recall_fg"),
        )
    if name == "mast3r":
        from repro_eval.geometry.mast3r import load_matcher
        return load_matcher(
            checkpoint=cfg["checkpoint"],
            input_size=cfg.get("input_size", 512),
            patch_size=cfg.get("patch_size", 16),
            subsample_or_initxy1=cfg.get("subsample_or_initxy1", 8),
            border=cfg.get("border", 3),
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "recall_fg"),
        )
    if name == "dift":
        from repro_eval.geometry.dift import load_matcher
        return load_matcher(
            model_id=cfg["checkpoint"],
            input_size=cfg["input_size"],
            timestep=cfg.get("timestep", 101),
            up_ft_index=cfg.get("up_ft_index", 1),
            prompt=cfg.get("prompt", ""),
            seed=cfg.get("seed", 0),
            fg_threshold=cfg.get("fg_threshold", 0.5),
            score_type=cfg.get("score_type", "recall_fg"),
        )
    raise ValueError(f"Unknown matcher: {name}")


def short_model_tag(model_id: str) -> str:
    """`facebook/dinov3-vitb16-pretrain-lvd1689m` -> `dinov3-vitb16`."""
    name = model_id.split("/")[-1]
    parts = name.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", type=str, default=None,
                    help="DreamBench++ method short alias (overrides config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    images_root = (REPO_ROOT / cfg["images_root"]).resolve()
    masks_root = (REPO_ROOT / cfg["masks_root"]).resolve()
    exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}
    signals = set(cfg.get("signals", ["geometry"]))
    if signals != {"geometry"}:
        raise NotImplementedError(f"only geometry supported right now, got {signals}")

    sample_source = cfg["sample_source"]
    builder = SAMPLE_BUILDERS[sample_source]
    method = args.method or cfg.get("method")
    builder_kwargs = cfg.get("builder_kwargs", {}) or {}
    if sample_source == "dreambenchplus":
        samples = builder(images_root, masks_root, exts, method=method, **builder_kwargs)
    else:
        samples = builder(images_root, masks_root, exts, **builder_kwargs)
    print(f"Built {len(samples)} samples from {sample_source}"
          + (f" (method={method})" if method else ""))

    model_tag = short_model_tag(cfg["model_id"])
    tag = cfg["tag"]
    if method:
        tag = f"{tag}_{method}"
    out_path = RESULTS_DIR / f"geometry__{model_tag}__{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_path}")

    matcher = load_matcher(cfg)
    use_ref_mask_on_out = bool(cfg.get("use_ref_mask_on_out", False))

    t0 = time.time()
    with out_path.open("w") as f:
        for i, s in enumerate(samples, 1):
            ref_img = Image.open(s["ref_image"])
            out_img = Image.open(s["out_image"])
            ref_mask_pil = Image.open(s["ref_mask"])
            ref_mask = np.array(ref_mask_pil)
            if use_ref_mask_on_out:
                # Resize the REF mask to the out image's native size, nearest-
                # neighbor to preserve the binary values.
                out_mask = np.array(ref_mask_pil.resize(out_img.size, Image.NEAREST))
            else:
                out_mask = np.array(Image.open(s["out_mask"]))

            res = matcher.fg_match(ref_img, ref_mask, out_img, out_mask)

            extras = {
                **s.get("extras", {}),
                "ref": s["ref_image"].name,
                "out": s["out_image"].name,
                "fg_threshold": cfg.get("fg_threshold", 0.5),
                "use_ref_mask_on_out": use_ref_mask_on_out,
                **res["extras"],
            }
            if "input_size" in cfg:
                extras["input_size"] = cfg["input_size"]

            row = {
                "concept_id": s["concept_id"],
                "prompt_id": s["prompt_id"],
                "gen_method": s["gen_method"],
                "seed": s["seed"],
                "signal": "geometry",
                "model": model_tag,
                "score": res["score"],
                "extras": extras,
            }
            f.write(json.dumps(row) + "\n")
            print(
                f"  [{i:>3}/{len(samples)}] {s['concept_id']}"
                f"  {s['ref_image'].name}<->{s['out_image'].name}"
                f"  score={res['score']:.3f}"
            )
    print(f"Done: wrote {len(samples)} rows to {out_path} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
