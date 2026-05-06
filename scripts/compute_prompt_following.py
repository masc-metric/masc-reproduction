"""Prompt-following via text-image cosine (CLIP-T / SigLIP2-T).

Per-sample score = cos(image_embed(out_image), text_embed(prompt_text)).
No masking — this is the direct drop-in analog of the CLIP-T baseline
the DreamBench++ paper reports.

Backend is dispatched on `model_id`:
- `openai/clip-*`  → `CLIPModel` + `CLIPProcessor`, `forward(**inputs)
  → {image,text}_embeds`, `padding=True` tokenizer — exact mirror of
  DreamBench++'s `CLIPScore.clipt_score`.
- `google/siglip2-*` → `AutoModel` + `AutoProcessor`. Two variants:
    - `variant: global` (default) — pooled `forward` cosine, the
      baseline SigLIP2-T reported in prior REPORT entries.
    - `variant: maxsim` — FILIP-style token-wise late interaction:
      per-text-token max cosine over all image patches (using raw
      pre-projection `last_hidden_state` at 768-dim on both towers),
      averaged over non-pad / non-eos text tokens.

Usage:
    python scripts/compute_prompt_following.py \\
        --config configs/prompt_following/{clipt,siglip2t}_<...>.yaml \\
        --method <dreambenchplus_method>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repro_eval.data import RESULTS_DIR, SAMPLE_BUILDERS, REPO_ROOT  # noqa: E402


def is_clip_model(model_id: str) -> bool:
    return model_id.startswith("openai/clip-")


def strip_subject_from_prompt(prompt: str, object_name: str) -> str:
    """Remove the subject/object name from a DB++ prompt so only the
    scene description survives. "A photo of a kitten chasing a butterfly
    in a sunny garden" + object='kitten' → "A photo of chasing a butterfly
    in a sunny garden". Tries "(a|an|the) <object>" first, then bare
    "<object>". re.escape handles hyphens and multi-word objects like
    'piggy bank' / 't-shirt'. Collapses whitespace at the end.
    """
    obj = re.escape(object_name.strip())
    for pat in (rf"\b(?:a|an|the)\s+{obj}\b", rf"\b{obj}\b"):
        if re.search(pat, prompt, flags=re.IGNORECASE):
            prompt = re.sub(pat, "", prompt, flags=re.IGNORECASE)
            break
    return re.sub(r"\s+", " ", prompt).strip()


def short_model_tag(model_id: str) -> str:
    """`google/siglip2-base-patch16-512` -> `siglip2-base`;
    `openai/clip-vit-base-patch32` -> `clip-vit`."""
    name = model_id.split("/")[-1]
    parts = name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else name


@torch.inference_mode()
def score_pair_siglip2(model, processor, out_image: Image.Image, prompt_text: str,
                       naflex_max_patches: int, device: str) -> float:
    """SigLIP2 text-image cosine for one (out_image, prompt) pair."""
    kwargs = dict(images=[out_image.convert("RGB")], text=[prompt_text],
                  return_tensors="pt", padding="max_length")
    if "naflex" in processor.image_processor.__class__.__name__.lower() or \
       hasattr(processor.image_processor, "max_num_patches"):
        # NaFlex processors accept max_num_patches; standard ones silently ignore it.
        kwargs["max_num_patches"] = naflex_max_patches
    try:
        inputs = processor(**kwargs)
    except TypeError:
        kwargs.pop("max_num_patches", None)
        inputs = processor(**kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    img_n = F.normalize(out.image_embeds, dim=-1)
    txt_n = F.normalize(out.text_embeds, dim=-1)
    return float((img_n * txt_n).sum(-1)[0].cpu())


@torch.inference_mode()
def score_pair_siglip2_maxsim(model, processor, out_image: Image.Image,
                              prompt_text: str, naflex_max_patches: int,
                              device: str, pad_id: int, eos_id: int,
                              project_text: bool = False) -> tuple[float, int, int]:
    """FILIP-style token-wise late interaction on frozen SigLIP2.

    For each non-pad/non-eos text token, take the max cosine similarity
    against any image patch, then average.

    If `project_text` is False (default `maxsim` variant): use raw
    pre-projection `last_hidden_state` on both sides (768-dim for
    SigLIP2-base).

    If `project_text` is True (`maxsim_textproj` variant): apply
    `text_model.head` (a 768→768 Linear, position-agnostic) to each
    token's hidden state before normalizing. Image patches stay raw —
    the vision pooler is an attention-collapse with a learned query and
    has no position-wise analog.

    Score ∈ roughly [−1, 1]. Returns (score, n_valid_text_tokens, n_patches).
    """
    kwargs = dict(images=[out_image.convert("RGB")], text=[prompt_text],
                  return_tensors="pt", padding="max_length")
    if "naflex" in processor.image_processor.__class__.__name__.lower() or \
       hasattr(processor.image_processor, "max_num_patches"):
        kwargs["max_num_patches"] = naflex_max_patches
    try:
        inputs = processor(**kwargs)
    except TypeError:
        kwargs.pop("max_num_patches", None)
        inputs = processor(**kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
    txt_out = model.text_model(input_ids=inputs["input_ids"])
    txt_hidden = txt_out.last_hidden_state[0]                      # [seq_len, D]
    if project_text:
        txt_hidden = model.text_model.head(txt_hidden)
    img_feats = F.normalize(vis_out.last_hidden_state[0], dim=-1)  # [n_patches, D]
    txt_feats = F.normalize(txt_hidden, dim=-1)                    # [seq_len, D]

    ids = inputs["input_ids"][0]
    valid = (ids != pad_id) & (ids != eos_id)
    if int(valid.sum()) == 0:
        return float("nan"), 0, int(img_feats.shape[0])
    txt_valid = txt_feats[valid]                                   # [n_tok, D]
    cos = txt_valid @ img_feats.T                                  # [n_tok, n_patches]
    per_token_max = cos.max(dim=-1).values                         # [n_tok]
    score = float(per_token_max.mean().cpu())
    return score, int(valid.sum()), int(img_feats.shape[0])


def _out_mask_to_patch_mask(mask_path: Path, grid, device: str) -> torch.Tensor:
    """Load an out mask, bilinear-downsample to the patch grid, threshold at 0.5.
    Returns a flat [Hp*Wp] bool tensor where True = patch is foreground.

    `grid` may be an int (square Hp=Wp=grid) or a 2-tuple (Hp, Wp) for
    NaFlex / non-square grids."""
    Hp, Wp = (grid, grid) if isinstance(grid, int) else grid
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    m = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(Hp, Wp), mode="bilinear", align_corners=False)
    fg = (m[0, 0].clamp(0.0, 1.0) >= 0.5)
    return fg.reshape(-1).to(device)


def _pool_with_mask(head, patches: torch.Tensor, mask_out_positions: torch.Tensor
                    ) -> torch.Tensor:
    """Run SigLIP2's `SiglipMultiheadAttentionPoolingHead` but with an
    attention key_padding_mask that hides specific patch positions from
    the learned probe query. Mirrors the head's own `forward` exactly,
    just threading the mask through `self.attention(...)`.

    `patches`: [1, N, D]. `mask_out_positions`: [N] bool, True = exclude.
    Returns [1, D].
    """
    probe = head.probe.repeat(patches.shape[0], 1, 1)
    # key_padding_mask: [B, S]. True at a position means ignore it.
    kpm = mask_out_positions.unsqueeze(0)
    attn_out = head.attention(probe, patches, patches, key_padding_mask=kpm)[0]
    residual = attn_out
    hidden = head.layernorm(attn_out)
    hidden = residual + head.mlp(hidden)
    return hidden[:, 0]


@torch.inference_mode()
def score_pair_siglip2_masked_pool(model, processor, out_image: Image.Image,
                                   out_mask_path: Path, prompt_text: str,
                                   naflex_max_patches: int, device: str,
                                   use_bg: bool) -> tuple[float, int, int]:
    """CP's "masked pool" idea transposed to PF: pool image patches
    through SigLIP2's *trained* attention pooler but restrict it to
    either BG (`use_bg=True`) or FG (`use_bg=False`) patches via
    `key_padding_mask`. The resulting pooled image embedding still lives
    in the joint space trained against the global pooled text embedding —
    unlike pre-projection MaxSim, the comparison is in-distribution.

    Returns (cosine, n_fg_patches, n_total_patches).
    """
    is_naflex = ("naflex" in processor.image_processor.__class__.__name__.lower()
                 or hasattr(processor.image_processor, "max_num_patches"))

    kwargs = dict(images=[out_image.convert("RGB")], text=[prompt_text],
                  return_tensors="pt", padding="max_length")
    if is_naflex:
        kwargs["max_num_patches"] = naflex_max_patches
    try:
        inputs = processor(**kwargs)
    except TypeError:
        kwargs.pop("max_num_patches", None)
        inputs = processor(**kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Vision forward. NaFlex needs pixel_attention_mask + spatial_shapes;
    # padded positions in last_hidden_state must be excluded from the pool.
    if is_naflex:
        # transformers 5.6+: outer Siglip2VisionModel.forward requires
        # `pixel_attention_mask` (not the old looser `attention_mask`).
        vis_out = model.vision_model(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
        )
        Hp, Wp = [int(x) for x in inputs["spatial_shapes"][0].tolist()]
        valid_mask = inputs["pixel_attention_mask"][0].bool()  # [N_max]
    else:
        vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
        N = vis_out.last_hidden_state.shape[1]
        Hp = Wp = int(round(N ** 0.5))
        if Hp * Wp != N:
            raise ValueError(f"Non-square (non-NaFlex) patch grid (n={N}).")
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)

    patches = vis_out.last_hidden_state  # [1, N_max, D]
    N_max = patches.shape[1]

    fg_real = _out_mask_to_patch_mask(out_mask_path, (Hp, Wp), device)  # [Hp*Wp]
    n_fg = int(fg_real.sum())
    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() != Hp * Wp:
        raise ValueError(f"valid count {valid_idx.numel()} != Hp*Wp {Hp*Wp}")

    # key_padding_mask: True = exclude. Start by excluding everything, then
    # unmask the real positions we want to include (BG if use_bg, else FG).
    kpm = torch.ones(N_max, dtype=torch.bool, device=device)
    include_real = (~fg_real) if use_bg else fg_real
    kpm[valid_idx] = ~include_real
    if int((~kpm).sum()) == 0:
        return float("nan"), n_fg, Hp * Wp

    pooled_img = _pool_with_mask(model.vision_model.head, patches, kpm)  # [1, D]

    txt_out = model.text_model(input_ids=inputs["input_ids"])
    pooled_txt = txt_out.pooler_output  # [1, D]

    img_n = F.normalize(pooled_img, dim=-1)
    txt_n = F.normalize(pooled_txt, dim=-1)
    score = float((img_n * txt_n).sum(-1)[0].cpu())
    return score, n_fg, Hp * Wp


@torch.inference_mode()
def score_pair_clip_masked_pool(model, processor, out_image: Image.Image,
                                out_mask_path: Path, prompt_text: str,
                                device: str, use_bg: bool) -> tuple[float, int, int]:
    """CLIP analog of `score_pair_siglip2_masked_pool`.

    CLIP pools by taking the CLS token (`last_hidden_state[:, 0]`), then
    `post_layernorm` + `visual_projection` to the joint space. We replace
    "take CLS" with "mean-pool the BG (or FG) patch tokens" and run the
    rest of the pooling path as-is. Text side: standard pooled forward.

    Returns (cosine, n_fg_patches, n_total_patches).
    """
    inputs = processor(text=[prompt_text], images=[out_image.convert("RGB")],
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
    # Drop CLS (index 0); keep the N patch tokens.
    patches = vis_out.last_hidden_state[:, 1:, :]  # [1, N, D_vis]
    n_total = patches.shape[1]
    grid = int(round(n_total ** 0.5))
    if grid * grid != n_total:
        raise ValueError(f"Non-square CLIP patch grid (n={n_total}).")

    fg_flat = _out_mask_to_patch_mask(out_mask_path, grid, device)  # True = FG
    n_fg = int(fg_flat.sum())
    keep = fg_flat if not use_bg else (~fg_flat)  # True = include in pool
    if int(keep.sum()) == 0:
        return float("nan"), n_fg, n_total

    # Mean-pool the kept patches → [1, D_vis], then run CLIP's pooled path.
    kept_patches = patches[0][keep]                         # [K, D_vis]
    pooled_hidden = kept_patches.mean(dim=0, keepdim=True)  # [1, D_vis]
    pooled_hidden = model.vision_model.post_layernorm(pooled_hidden)
    img_embed = model.visual_projection(pooled_hidden)      # [1, D_joint]

    # Text: standard pooled forward through full CLIPModel (so we get the
    # joint-space text_embed via text_projection exactly as in the
    # standard CLIP-T path).
    txt_out = model.text_model(input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"])
    text_hidden = txt_out.pooler_output  # [1, D_txt]
    text_embed = model.text_projection(text_hidden)          # [1, D_joint]

    img_n = F.normalize(img_embed, dim=-1)
    txt_n = F.normalize(text_embed, dim=-1)
    score = float((img_n * txt_n).sum(-1)[0].cpu())
    return score, n_fg, n_total


@torch.inference_mode()
def score_pair_clip(model, processor, out_image: Image.Image, prompt_text: str,
                    device: str) -> float:
    """CLIP text-image cosine — mirrors DreamBench++'s `CLIPScore.clipt_score`
    (padding=True, projection-head features, L2-normalize, dot product).
    Uses `forward(**inputs)` → `image_embeds` / `text_embeds` (same projected
    features as `get_{text,image}_features` but version-stable across HF
    transformers refactors). Returns raw cosine ∈ [−1, 1]; DB++ reports ×100."""
    inputs = processor(text=[prompt_text], images=[out_image.convert("RGB")],
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    img_n = F.normalize(out.image_embeds, dim=-1)
    txt_n = F.normalize(out.text_embeds, dim=-1)
    return float((img_n * txt_n).sum(-1)[0].cpu())


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

    model_id = cfg["model_id"]
    model_tag = short_model_tag(model_id)
    tag = cfg["tag"] + (f"_{method}" if method else "")
    out_path = RESULTS_DIR / f"prompt_following__{model_tag}__{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_path}")

    device = cfg.get("device", "cuda")
    naflex_max_patches = int(cfg.get("naflex_max_patches", 1024))
    variant = cfg.get("variant", "global")
    valid_variants = {"global", "maxsim", "maxsim_textproj", "global_bg", "global_fg"}
    if variant not in valid_variants:
        raise ValueError(f"Unknown variant: {variant!r}. Expected one of {valid_variants}.")

    is_clip = is_clip_model(model_id)
    if is_clip and variant not in {"global", "global_bg", "global_fg"}:
        raise ValueError(f"CLIP backend does not support variant={variant!r}.")
    print(f"Loading {model_id} ... (backend={'clip' if is_clip else 'siglip2'}, variant={variant})")
    t0 = time.time()
    if is_clip:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).eval().to(device)
    else:
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval().to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
    eos_id = getattr(processor.tokenizer, "eos_token_id", 1) or 1

    strip_subject = bool(cfg.get("strip_subject", False))
    skip_style = bool(cfg.get("skip_style", False))
    if strip_subject:
        print(f"  strip_subject=True  (removes object name from prompt before scoring)")
    if skip_style:
        print(f"  skip_style=True  (drops style_* subjects)")

    t0 = time.time()
    n_skipped = 0
    with out_path.open("w") as f:
        for i, s in enumerate(samples, 1):
            extras_in = s.get("extras", {})
            prompt_text = extras_in.get("prompt_text", "")
            if not prompt_text:
                n_skipped += 1
                continue
            if skip_style and str(extras_in.get("category", "")).startswith("style"):
                n_skipped += 1
                continue
            orig_prompt = prompt_text
            if strip_subject:
                object_name = extras_in.get("object_name", "")
                if object_name:
                    prompt_text = strip_subject_from_prompt(prompt_text, object_name)
            out_img = Image.open(s["out_image"])
            n_tok = n_patch = None
            extras_mask = None
            if is_clip and variant in {"global_bg", "global_fg"}:
                score, n_fg, n_patch = score_pair_clip_masked_pool(
                    model, processor, out_img, s["out_mask"], prompt_text,
                    device=device, use_bg=(variant == "global_bg"))
                extras_mask = {"n_fg_patches": n_fg, "n_patches": n_patch}
            elif is_clip:
                score = score_pair_clip(model, processor, out_img, prompt_text,
                                        device=device)
            elif variant in {"maxsim", "maxsim_textproj"}:
                score, n_tok, n_patch = score_pair_siglip2_maxsim(
                    model, processor, out_img, prompt_text,
                    naflex_max_patches=naflex_max_patches,
                    device=device, pad_id=pad_id, eos_id=eos_id,
                    project_text=(variant == "maxsim_textproj"))
            elif variant in {"global_bg", "global_fg"}:
                score, n_fg, n_patch = score_pair_siglip2_masked_pool(
                    model, processor, out_img, s["out_mask"], prompt_text,
                    naflex_max_patches=naflex_max_patches,
                    device=device, use_bg=(variant == "global_bg"))
                extras_mask = {"n_fg_patches": n_fg, "n_patches": n_patch}
            else:
                score = score_pair_siglip2(model, processor, out_img, prompt_text,
                                           naflex_max_patches=naflex_max_patches,
                                           device=device)

            extras = {
                **s.get("extras", {}),
                "out": s["out_image"].name,
            }
            if strip_subject:
                extras["prompt_stripped"] = prompt_text
                extras["prompt_orig"] = orig_prompt
            if not is_clip:
                extras["naflex_max_patches"] = naflex_max_patches
            extras["variant"] = variant
            if n_tok is not None:
                extras["n_text_tokens"] = n_tok
                extras["n_image_patches"] = n_patch
            if extras_mask is not None:
                extras.update(extras_mask)
            row = {
                "concept_id": s["concept_id"],
                "prompt_id": s["prompt_id"],
                "gen_method": s["gen_method"],
                "seed": s["seed"],
                "signal": "prompt_following",
                "model": model_tag,
                "score": score,
                "extras": extras,
            }
            f.write(json.dumps(row) + "\n")
            if i % 50 == 0 or i == len(samples):
                print(f"  [{i:>4}/{len(samples)}]  cos={score:.3f}")
    print(f"Done: wrote {len(samples) - n_skipped} rows (skipped {n_skipped} empty prompts) "
          f"in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
