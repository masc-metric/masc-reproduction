# Reproducing the paper

This document walks through reproducing every numbered table and figure
in the paper from raw data. Every script is standalone and driven by a
YAML config — one run = one (matcher, dataset, tag) tuple. Outputs are
JSONL files under `results/`; aggregation scripts read them back.

All paths below are relative to the repo root.

## 0. Setup

```bash
pip install -e ".[repro]"
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu128
```

Optional comparator extras (each comparator only needs the package(s) it
imports — install lazily):

```bash
# CP comparators that aren't covered by the base install
pip install dreamsim                                # DreamSim
pip install kornia                                  # LoFTR (matcher ablation)

# PF comparators (Table 3)
pip install t2v-metrics imageio-ffmpeg              # VQAScore
pip install image-reward                            # ImageReward
# HPSv3 has no pip release; the loader pulls weights from HF on first use

# AM-RADIO and DIFT install per their upstream READMEs.
# RoMa / MASt3R / SuperPoint+LightGlue are pip-installable; consult
# their upstream repos.
```

`.env` at the repo root must contain:

```
HF_TOKEN=hf_...                # gates SigLIP2, SAM3, DINOv3, DreamSim
```

This repository **does not invoke any external API**. GPT-4o / GPT-4V
rows that appear in DB++ Tables 1 + 3 come from the static
`data_gpt_rating/` JSON files shipped with DreamBench++; reading them is
a disk operation, not an API call. There is no GPT row in the ORIDa
table since DreamBench++ does not ship ORIDa GPT ratings.

## 1. Datasets

### DreamBench++ (primary)

Download to `data/dreambench_plus/` per the upstream README:
https://github.com/yuangpeng/dreambench_plus. Required subtrees:

- `images/` — reference images
- `samples/<method_full>/{src_image,tgt_image,text}/` — generations + prompts
- `data_human_rating/`, `data_clip_rating/`, `data_dino_rating/`,
  `data_gpt_rating/` — comparator scores shipped with the dataset
- `captions/` — per-reference object names

### ORIDa

Download to `data/ORIDa/ORIDa_v1.0/` per the dataset's release page
(https://arxiv.org/abs/2506.08964). Required:

- `train/<subj>/factual_only/{images,annotations/masks}/`
- `train/<subj>/factual_counterfactual/<scene>/{images,annotations/masks}/`

ORIDa ships its own segmentation masks — no SAM3 step needed.

## 2. Generate masks (DreamBench++ only)

```bash
python scripts/generate_masks.py --config configs/masks/sam3_dreambenchplus_refs.yaml
for method in dreambooth_sd dreambooth_lora_sdxl textual_inversion_sd \
              blip_diffusion emu2 \
              ip_adapter_plus_vit_h_sdxl ip_adapter_vit_g_sdxl; do
  python scripts/generate_masks.py \
    --config configs/masks/sam3_dreambenchplus_samples.yaml \
    --method $method
done
```

Masks land under `data/masks/dreambench_plus/{refs,samples/<method_full>/}/`.
Full DB++ run is a few hours on an RTX 3090.

## 3. Table 1 — CP on DreamBench++

Run each comparator over all 7 generation methods. Each invocation
writes one JSONL per `(matcher, method)` to `results/`.

**MaSC (primary metric).**

```bash
for m in dreambooth_sd dreambooth_lora_sdxl textual_inversion_sd \
         blip_diffusion emu2 \
         ip_adapter_plus_vit_h_sdxl ip_adapter_vit_g_sdxl; do
  python scripts/compute_geometry.py \
    --config configs/geometry/siglip2_so400m_naflex_dreambenchplus_maskedmaxcos.yaml \
    --method $m
done
```

Repeat the same `compute_geometry.py` loop with each of the comparator
configs:

| Row in Table 1 | Config |
| --- | --- |
| MaSC | `siglip2_so400m_naflex_dreambenchplus_maskedmaxcos.yaml` |
| SigLIP2 SO400M-NaFlex global pool | `siglip2_global_so400m_naflex_dreambenchplus.yaml` |
| DreamSim | `dreamsim_dreambenchplus.yaml` |
| DINOv3 (ViT-L/16 CLS cosine) | `dinov3_dbplus_dreambenchplus.yaml` |
| DIFT-SDXL (canonical) | `dift_canonical_dreambenchplus.yaml` |
| AM-RADIO C-RADIOv4-SO400M | `radio_summary_dreambenchplus.yaml` |

DINO-I and CLIP-I come from `data_dino_rating/` and `data_clip_rating/`
shipped with DreamBench++ — no compute step. GPT-4o and GPT-4V come from
`data_gpt_rating/`.

**Aggregate to α / ρ.** Reads every `results/geometry__*` JSONL plus the
DB++-shipped ratings, intersects on the apples-to-apples keyset, prints
the table:

```bash
python scripts/plot_dreambenchplus_cp.py
```

**Validate the α formula** against DB++'s published Table 3 Kd_o values
(should match within ±0.02):

```bash
python scripts/reproduce_dbplus_kdo.py
```

## 4. Table 2 + Figure 2 — CP on ORIDa

ORIDa pipeline is a single shell script that runs every comparator over
the 50 subjects × 10 backgrounds × 2,250 within / 2,250 cross pair set
used in the paper:

```bash
python scripts/compute_geometry.py \
  --config configs/geometry/siglip2_so400m_naflex_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/siglip2_global_so400m_naflex_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/dreamsim_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/dinov3_dbplus_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/dift_canonical_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/radio_summary_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/dinoi_dbplus_orida_50x10.yaml
python scripts/compute_geometry.py \
  --config configs/geometry/clipi_dbplus_orida_50x10.yaml
```

Aggregate AUC(within > cross) + render Figure 2:

```bash
python scripts/plot_orida_cp.py
```

## 5. Table 3 — PF on DreamBench++

**MaSC PF** (subject-stripped prompt × BG-pool on SigLIP2 SO400M-NaFlex):

```bash
for m in dreambooth_sd dreambooth_lora_sdxl textual_inversion_sd \
         blip_diffusion emu2 \
         ip_adapter_plus_vit_h_sdxl ip_adapter_vit_g_sdxl; do
  python scripts/compute_prompt_following.py \
    --config configs/prompt_following/siglip2t_so400m_naflex_global_bg_nosubj_dreambenchplus.yaml \
    --method $m
done
```

**SigLIP2 global pool baseline** (Table 3 same-backbone control):

```bash
for m in dreambooth_sd dreambooth_lora_sdxl textual_inversion_sd \
         blip_diffusion emu2 \
         ip_adapter_plus_vit_h_sdxl ip_adapter_vit_g_sdxl; do
  python scripts/compute_prompt_following.py \
    --config configs/prompt_following/siglip2t_so400m_naflex_dreambenchplus.yaml \
    --method $m
done
```

**External non-LLM PF baselines** (VQAScore, ImageReward, HPSv3):

```bash
for cfg in vqascore imagereward hpsv3; do
  for m in dreambooth_sd dreambooth_lora_sdxl textual_inversion_sd \
           blip_diffusion emu2 \
           ip_adapter_plus_vit_h_sdxl ip_adapter_vit_g_sdxl; do
    python scripts/compute_prompt_following_external.py \
      --config configs/prompt_following/${cfg}_dreambenchplus.yaml \
      --method $m
  done
done
```

CLIP-T comes from `data_clipt_rating/` shipped with DreamBench++.
GPT-4o / GPT-4V come from `data_gpt_rating/` (PF variant).

Aggregate to α / ρ:

```bash
python scripts/compute_new_pf_baselines.py
```

## 6. Table 4 — PF subject-strip × pool ablation

The 2 × 3 ablation grid (BG-pool / full-pool / FG-pool × stripped /
unstripped prompt) is one script:

```bash
python scripts/compare_nosubj_ablation.py
```

It expects the per-cell PF runs to be on disk under `results/`. Re-run
the relevant `compute_prompt_following.py` configs first if they aren't.

## 7. Section 4.4 — "Aggregator dominates features"

The two same-features-different-aggregator runs:

```bash
# mutual-NN recall_fg aggregator on SigLIP2 patch features
python scripts/compute_geometry.py \
  --config configs/geometry/siglip2_dreambenchplus.yaml      # score_type: recall_fg

# mutual-NN recall_fg aggregator on DINOv3 patch features
python scripts/compute_geometry.py \
  --config configs/geometry/dinov3_dreambenchplus.yaml       # score_type: recall_fg
```

The masked-maxcos versions of the four specialized matchers (LoFTR,
RoMa, MASt3R, SuperPoint+LightGlue) — all land α-negative on DB++ as
reported in the paper:

```bash
python scripts/compute_geometry.py --config configs/geometry/loftr_dreambenchplus_maskedmaxcos.yaml
python scripts/compute_geometry.py --config configs/geometry/roma_dreambenchplus_maskedmaxcos.yaml
python scripts/compute_geometry.py --config configs/geometry/mast3r_dreambenchplus_maskedmaxcos.yaml
python scripts/compute_geometry.py --config configs/geometry/lightglue_dreambenchplus.yaml
```

`plot_dreambenchplus_cp.py` includes these in its full table when their
JSONLs are present under `results/`.

## 8. Table 5 — Compute and runtime

Per-pair latency on a single fixed DreamBench++ pair, each metric in its
own subprocess with `torch.cuda.synchronize()` around every timed call.
Reported numbers were measured on an RTX 3090.

```bash
# CP-side timings (Table 5 upper rows + the MaSC CP entry)
python scripts/benchmark_speed.py

# PF-side timings (VQAScore, ImageReward, HPSv3, MaSC PF, etc.)
python scripts/benchmark_speed_pf_baselines.py
```

## Output JSONL schema

One row per `(sample, signal)`:

```json
{
  "concept_id": "object_00_motorcycle",
  "prompt_id": "0",
  "gen_method": "dreambooth_sd",
  "seed": 0,
  "signal": "geometry",
  "model": "siglip2-so400m-naflex",
  "score": 0.731,
  "extras": {"cp_n_ref_fg": 412, "fg_threshold": 0.5}
}
```

- `signal ∈ {"geometry", "prompt_following"}`.
- One JSONL per `(stage, model, tag, gen_method)` run, named
  `results/<stage>__<model>__<tag>_<gen_method>.jsonl`.
- `extras` holds per-model diagnostics.
