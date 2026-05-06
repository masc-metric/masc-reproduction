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


def _load_done_keys(out_path: Path) -> set[tuple]:
    if not out_path.exists():
        return set()
    done: set[tuple] = set()
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["concept_id"], r["prompt_id"], r["gen_method"], r["seed"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    images_root = (REPO_ROOT / cfg["images_root"]).resolve()
    masks_root = (REPO_ROOT / cfg["masks_root"]).resolve()
    exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}

    sample_source = cfg["sample_source"]
    builder = SAMPLE_BUILDERS[sample_source]
    method = args.method or cfg.get("method")
    if sample_source == "dreambenchplus":
        samples = builder(images_root, masks_root, exts, method=method)
    else:
        samples = builder(images_root, masks_root, exts)
    print(f"Built {len(samples)} samples from {sample_source}"
          + (f" (method={method})" if method else ""))

    scorer_name = cfg["scorer"]
    tag = cfg["tag"] + (f"_{method}" if method else "")
    out_path = RESULTS_DIR / f"prompt_following__{scorer_name}__{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_path}")

    done = _load_done_keys(out_path)
    if done:
        print(f"  resume: {len(done)} keys already present, will skip")

    print(f"Loading scorer={scorer_name} ...")
    t0 = time.time()
    scorer = load_scorer(cfg)
    print(f"  loaded in {time.time()-t0:.1f}s")

    skip_style = bool(cfg.get("skip_style", False))
    if skip_style:
        print(f"  skip_style=True  (drops style_* subjects)")

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


if __name__ == "__main__":
    raise SystemExit(main())
