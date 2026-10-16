"""Compute geometry (concept-fidelity) signal over (ref, out, ref_mask, out_mask).

Writes one JSONL row per sample to
`results/geometry__<model>__<tag>.jsonl`. Resumable: rows already in the
output file (same concept, prompt, method, seed, ref, out) are skipped and new
rows are appended.

Options used by the appendix analyses (README, "Appendix"):
    --method a,b,c           several DreamBench++ methods, model loaded once
    --min-fg-fraction none   score every DreamBench++ pair (no mask-area
                             filter); thresholds are applied afterwards from
                             the foreground fractions stored in each row
    --tag-suffix _nofilter   keep such runs apart from the default files
    --dump-patch-arrays      (patch-feature matchers) also write each
                             reference patch's max cosine and the patch grid
                             to results/aux/<result>.patch_arrays.jsonl, from
                             which CP under any other reference mask follows
    --max-samples N          smoke tests

Usage:
    python scripts/compute_geometry.py --config configs/geometry/dinov3.yaml
"""
from __future__ import annotations

import argparse
import base64
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
from repro_eval.runs import (  # noqa: E402
    BUILDER_DEFAULT,
    load_once,
    method_list,
    min_fg_fraction,
    read_done_keys,
    terminate_partial_line,
    write_run_meta,
)


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


def _row_key(r: dict) -> tuple:
    ex = r.get("extras", {})
    return (r["concept_id"], str(r["prompt_id"]), r["gen_method"], r["seed"],
            ex.get("ref"), ex.get("out"))


def _aux_key(r: dict) -> tuple:
    return (r["concept_id"], str(r["prompt_id"]), r["gen_method"], r["seed"], r["ref"], r["out"])


def run(args: argparse.Namespace, method: str | None) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    images_root = (REPO_ROOT / cfg["images_root"]).resolve()
    masks_root = (REPO_ROOT / cfg["masks_root"]).resolve()
    exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}
    signals = set(cfg.get("signals", ["geometry"]))
    if signals != {"geometry"}:
        raise NotImplementedError(f"only geometry supported right now, got {signals}")

    sample_source = cfg["sample_source"]
    builder = SAMPLE_BUILDERS[sample_source]
    method = method or cfg.get("method")
    builder_kwargs = dict(cfg.get("builder_kwargs", {}) or {})
    if args.min_fg_fraction is not BUILDER_DEFAULT:
        if sample_source != "dreambenchplus":
            raise ValueError("--min-fg-fraction applies to DreamBench++ only")
        builder_kwargs["min_fg_fraction"] = args.min_fg_fraction
    if sample_source == "dreambenchplus":
        samples = builder(images_root, masks_root, exts, method=method, **builder_kwargs)
        keys = [f"{s['concept_id']}-{s['prompt_id']}" for s in samples]
        if len(keys) != len(set(keys)):
            raise RuntimeError("duplicate concept-prompt keys: several seeds per prompt "
                               "would collide in the rating join")
    else:
        samples = builder(images_root, masks_root, exts, **builder_kwargs)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Built {len(samples)} samples from {sample_source}"
          + (f" (method={method})" if method else ""))

    model_tag = short_model_tag(cfg["model_id"])
    tag = cfg["tag"] + args.tag_suffix
    if method:
        tag = f"{tag}_{method}"
    out_path = RESULTS_DIR / f"geometry__{model_tag}__{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aux_path = RESULTS_DIR / "aux" / f"{out_path.stem}.patch_arrays.jsonl"
    print(f"Writing to {out_path}")
    terminate_partial_line(out_path)
    terminate_partial_line(aux_path)
    done = read_done_keys(out_path, _row_key)
    if args.dump_patch_arrays and done:
        n_arrays = len(read_done_keys(aux_path, _aux_key))
        if n_arrays < len(done):
            raise RuntimeError(
                f"{out_path} has {len(done)} rows but only {n_arrays} have patch arrays; resuming "
                "would skip them. Remove the result file or pass another --tag-suffix.")
    if done:
        print(f"  resume: {len(done)} rows already present, skipping them")

    matcher = load_once(json.dumps(cfg, sort_keys=True), lambda: load_matcher(cfg))
    if args.dump_patch_arrays:
        if not hasattr(matcher, "return_arrays"):
            raise ValueError(f"--dump-patch-arrays needs a PatchFgMatcher, got {type(matcher)}")
        matcher.return_arrays = True
        aux_path.parent.mkdir(parents=True, exist_ok=True)
    use_ref_mask_on_out = bool(cfg.get("use_ref_mask_on_out", False))
    write_run_meta(out_path, config=cfg, config_path=args.config.resolve(),
                   model_ids=[cfg.get("model_id", ""), cfg.get("checkpoint", "")],
                   argv=sys.argv, extra={"n_samples": len(samples), "n_done_before": len(done)})

    t0 = time.time()
    n_new = 0
    aux_f = aux_path.open("a") if args.dump_patch_arrays else None
    with out_path.open("a") as f:
        for i, s in enumerate(samples, 1):
            if (s["concept_id"], str(s["prompt_id"]), s["gen_method"], s["seed"],
                    s["ref_image"].name, s["out_image"].name) in done:
                continue
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
            if aux_f is not None:
                arr = res["arrays"]["nn_cos_per_ref"].astype("<f4")
                aux_f.write(json.dumps({
                    "concept_id": s["concept_id"], "prompt_id": s["prompt_id"],
                    "gen_method": s["gen_method"], "seed": s["seed"],
                    "ref": s["ref_image"].name, "out": s["out_image"].name,
                    "ref_grid": res["arrays"]["ref_grid"],
                    "nn_cos_per_ref_f32_b64": base64.b64encode(arr.tobytes()).decode(),
                }) + "\n")
                aux_f.flush()
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_new += 1
            if n_new % 50 == 0 or i == len(samples):
                print(
                    f"  [{i:>4}/{len(samples)}] {s['concept_id']}"
                    f"  {s['ref_image'].name}<->{s['out_image'].name}"
                    f"  score={res['score']:.3f}"
                    f"  ({(time.time() - t0) / n_new:.3f} s/pair)"
                )
    if aux_f is not None:
        aux_f.close()
    print(f"Done: wrote {n_new} new rows to {out_path} in {time.time()-t0:.1f}s")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", type=method_list, default=[None],
                    help="DreamBench++ method short alias(es), comma-separated "
                         "(overrides the config)")
    ap.add_argument("--min-fg-fraction", type=min_fg_fraction, default=BUILDER_DEFAULT,
                    help="DreamBench++ mask-area filter; 'none' scores every pair "
                         "(default: the builder's 0.05)")
    ap.add_argument("--tag-suffix", type=str, default="")
    ap.add_argument("--dump-patch-arrays", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    return ap


def main() -> int:
    args = parser().parse_args()
    rc = 0
    for method in args.method:
        rc |= run(args, method)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
