"""Prompt-following via external (non-CLIP-T) scorers.

Sister script to `compute_prompt_following.py` for metrics that don't fit
the simple text-image-cosine pattern. Dispatches on `cfg["scorer"]` to a
module under `repro_eval/pf/<scorer>.py` exposing a uniform interface:

    class Scorer:
        def score(self, image: PIL.Image, prompt_text: str) -> dict:
            return {"score": float, "extras": dict}

Writes one JSONL row per (sample, signal) to
`results/prompt_following__<scorer>__<tag>.jsonl`. Resumable: skips rows
whose `(concept_id, prompt_id, gen_method, seed)` keys already appear in
the output file.

Currently supports:
  - vqascore     (Lin et al., ECCV 2024)            — `pip install t2v-metrics`
  - imagereward  (Xu et al., NeurIPS 2023)          — `pip install image-reward`
  - hpsv3        (Ma et al., 2025)                  — vendored or HF model

Usage:
    python scripts/compute_prompt_following_external.py \\
        --config configs/prompt_following/vqascore_dreambenchplus.yaml \\
        --method dreambooth_sd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import repro_eval  # noqa: F401  — triggers .env loading
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


def load_scorer(cfg: dict):
    name = cfg["scorer"]
    if name == "vqascore":
        from repro_eval.pf.vqascore import load_scorer
        return load_scorer(
            model_id=cfg.get("model_id", "clip-flant5-xxl"),
            device=cfg.get("device", "cuda"),
        )
    if name == "imagereward":
        from repro_eval.pf.imagereward import load_scorer
        return load_scorer(
            model_id=cfg.get("model_id", "ImageReward-v1.0"),
            device=cfg.get("device", "cuda"),
        )
    if name == "hpsv3":
        from repro_eval.pf.hpsv3 import load_scorer
        return load_scorer(
            model_id=cfg.get("model_id", "MizzenAI/HPSv3"),
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown scorer: {name}")


def _row_key(r: dict) -> tuple:
    return (r["concept_id"], r["prompt_id"], r["gen_method"], r["seed"])


def run(args: argparse.Namespace, method: str | None) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    images_root = (REPO_ROOT / cfg["images_root"]).resolve()
    masks_root = (REPO_ROOT / cfg["masks_root"]).resolve()
    exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}

    sample_source = cfg["sample_source"]
    builder = SAMPLE_BUILDERS[sample_source]
    method = method or cfg.get("method")
    builder_kwargs = {}
    if args.min_fg_fraction is not BUILDER_DEFAULT:
        builder_kwargs["min_fg_fraction"] = args.min_fg_fraction
    if sample_source == "dreambenchplus":
        samples = builder(images_root, masks_root, exts, method=method, **builder_kwargs)
    else:
        samples = builder(images_root, masks_root, exts, **builder_kwargs)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Built {len(samples)} samples from {sample_source}"
          + (f" (method={method})" if method else ""))

    scorer_name = cfg["scorer"]
    tag = cfg["tag"] + args.tag_suffix + (f"_{method}" if method else "")
    out_path = RESULTS_DIR / f"prompt_following__{scorer_name}__{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_path}")

    terminate_partial_line(out_path)
    done = read_done_keys(out_path, _row_key)
    if done:
        print(f"  resume: {len(done)} keys already present, will skip")

    write_run_meta(out_path, config=cfg, config_path=args.config.resolve(),
                   model_ids=[cfg.get("model_id", "")], argv=sys.argv,
                   extra={"n_samples": len(samples), "n_done_before": len(done)})
    print(f"Loading scorer={scorer_name} ...")
    t0 = time.time()
    scorer = load_once(json.dumps(cfg, sort_keys=True), lambda: load_scorer(cfg))
    print(f"  loaded in {time.time()-t0:.1f}s")

    skip_style = bool(cfg.get("skip_style", False))
    if skip_style:
        print("  skip_style=True  (drops style_* subjects)")

    t0 = time.time()
    n_skipped = n_done = 0
    # Append mode so resumes don't clobber prior rows.
    with out_path.open("a") as f:
        for i, s in enumerate(samples, 1):
            key = (s["concept_id"], s["prompt_id"], s["gen_method"], s["seed"])
            if key in done:
                continue
            extras_in = s.get("extras", {})
            prompt_text = extras_in.get("prompt_text", "")
            if not prompt_text:
                n_skipped += 1
                continue
            if skip_style and str(extras_in.get("category", "")).startswith("style"):
                n_skipped += 1
                continue

            out_img = Image.open(s["out_image"]).convert("RGB")
            try:
                res = scorer.score(out_img, prompt_text)
            except Exception as e:
                print(f"  [{i}] error on {s['out_image'].name}: {e!r}")
                continue

            extras = {
                **extras_in,
                "out": s["out_image"].name,
                "scorer": scorer_name,
                **res.get("extras", {}),
            }
            row = {
                "concept_id": s["concept_id"],
                "prompt_id": s["prompt_id"],
                "gen_method": s["gen_method"],
                "seed": s["seed"],
                "signal": "prompt_following",
                "model": scorer_name,
                "score": res["score"],
                "extras": extras,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_done += 1
            if i % 50 == 0 or i == len(samples):
                print(f"  [{i:>4}/{len(samples)}]  score={res['score']:.4f}")
    print(f"Done: wrote {n_done} new rows (skipped {n_skipped} prompt-empty/style) "
          f"in {time.time()-t0:.1f}s")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", type=method_list, default=[None],
                    help="DreamBench++ method short alias(es), comma-separated")
    ap.add_argument("--min-fg-fraction", type=min_fg_fraction, default=BUILDER_DEFAULT,
                    help="DreamBench++ mask-area filter; 'none' scores every pair")
    ap.add_argument("--tag-suffix", type=str, default="")
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
