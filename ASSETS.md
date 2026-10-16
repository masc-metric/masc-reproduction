# Assets

This file lists the datasets, checkpoints and third-party code that the reproduction uses, with the revision each was run at and its license or access terms.

The license and terms column follows the paper's appendix (*Licenses, Access Terms, and Attribution*). Rows marked † are not in that appendix table; their entry is taken from the model or package page. Check each asset's official source before redistributing anything derived from it.

The revisions are those of a replication run of every table on 2026-10-10 (NVIDIA H100). `manifests/` holds the archive and checkpoint sha256 sums and one `pip freeze` per environment (`main`: MaSC, the CP baselines and the masks; `vqa`, `ir`, `hps`: VQAScore, ImageReward and HPSv3; `match`: the Section 4.4 matchers). The configs pin no model revision, so the Hugging Face Hub serves the latest one; every scoring run records the snapshot it resolved in `results/<result>.meta.jsonl`.

## Datasets

Neither dataset is redistributed. The scripts rebuild every table after you obtain the data under its original terms.

### DreamBench++

- **Source:** <https://github.com/yuangpeng/dreambench_plus>. Images and human ratings come from the Google Drive release that accompanies the paper.
- **License / terms:** code under Apache 2.0. The image and rating data carry no explicit license. Use them for academic evaluation, consistent with the dataset paper.
- **Archives, sha256** (`manifests/dreambench_archives_sha256.txt`):

  | Archive | sha256 |
  |---|---|
  | `data.zip` | `fabdaac1cbff6826c226e874902193ac7965e29b6c1644e983551c980ec82a33` |
  | `samples.zip` | `0218a8b3de04cc0188a72ce8b77167087603e253be7f4fce7f3ba8cce401174b` |
  | `data_human_rating.zip` | `894a80173c6832c8967ec64cc8e9aa636d41efeb131399c6670a0b5ac348a424` |
  | `data_gpt_rating.zip` | `f7de4b997d559a88ed516f70c3370c8b3d6fd5c66d010b4d328122727a9c8317` |
  | `data_clipi_rating.zip` | `0c17e3198956b0e16c34b09a262b653bf9b7369117f126c9746fc195585fad09` |
  | `data_clipt_rating.zip` | `01b551326bf8d3253358c1f13f2c05c8b30d2d79fe0b8a50ef1f3f15c1abcb07` |
  | `data_dino_rating.zip` | `df583d5eb8b33ca6daf8a73ce1eb85588a796ed42f326f5134cb3a1b45cd08a6` |

### ORIDa

- **Source:** obtained from the ORIDa authors through their access process (Yonsei University). We use the sRGB release and the object captions.
- **License / terms:** no explicit dataset license is published; the paper text is CC BY-NC-SA 4.0. Treat use as non-commercial academic research.
- **Archives, sha256** (`manifests/orida_archive_sha256.txt`):

  | Archive | sha256 |
  |---|---|
  | `ORIDa_sRGB.zip` | `3249e4746ec6dbbb5508b6fe464b89d40b153eb98825f13e60e3ae461db1bc27` |
  | object captions | `206d21c5c67f57579ec7b83aab24e94ef62504d44c4a669c69036f57494926cc` |

## Checkpoints on the Hugging Face Hub

| Asset | Repository | Revision | License / terms | Used for |
|---|---|---|---|---|
| SigLIP2 SO400M-NaFlex | `google/siglip2-so400m-patch16-naflex` | `cc24074f717b612951c2dead130904ab9b65a81e` | Apache 2.0 | MaSC backbone (CP and PF), global-pool controls |
| SigLIP2 B/16 @ 512 † | `google/siglip2-base-patch16-512` | `a89f5c5093f902bf39d3cd4d81d2c09867f0724b` | Apache 2.0 | aggregator analysis (§4.4) |
| SAM 3 | `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` | SAM License (Meta); use of SAM Materials must be acknowledged | concept masks |
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` | Meta DINOv3 License; use of DINO Materials must be acknowledged | DINOv3 CP baseline |
| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` | Meta DINOv3 License | aggregator analysis (§4.4) |
| C-RADIOv4-SO400M | `nvidia/C-RADIOv4-SO400M` | `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8` | NVIDIA Open Model License Agreement | AM-RADIO baseline |
| Stable Diffusion XL base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` | CreativeML Open RAIL++-M | DIFT-SDXL baseline |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | MIT (OpenAI code) | CLIP-I on ORIDa; CLIP-I and CLIP-T in the runtime benchmark |
| CLIPSeg | `CIDAS/clipseg-rd64-refined` | `999e0328d9e10b484360c477313983f9afdd7050` | weights Apache 2.0; code MIT | segmenter sensitivity |
| Grounding DINO base | `IDEA-Research/grounding-dino-base` | `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | Apache 2.0 | Grounded-SAM2 masks |
| SAM 2.1 Hiera-L | `facebook/sam2.1-hiera-large` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` | Apache 2.0 | Grounded-SAM2 and OWLv2+SAM2 masks |
| OWLv2 | `google/owlv2-base-patch16-ensemble` | `cfd3195ba4ea9592eec887ded089f4c08eff231d` | Apache 2.0 | OWLv2+SAM2 masks |
| VQAScore (CLIP-FlanT5-XXL) | `zhiqiulin/clip-flant5-xxl` | `89bad6fffe1126b24d4360c1e1f69145eb6103aa` | Apache 2.0 (t2v_metrics) | PF baseline |
| CLIP ViT-L/14-336 † | `openai/clip-vit-large-patch14-336` | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` | OpenAI CLIP (MIT code); see the model card | VQAScore vision tower |
| FLAN-T5-XXL † | `google/flan-t5-xxl` | `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` | Apache 2.0 | VQAScore tokenizer and config |
| HPSv3 | `MizzenAI/HPSv3` | `4f81e3e09edd82fe3c5f636444c721b592a735ca` | MIT (code) | PF baseline |
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` | Apache 2.0 | HPSv3 backbone |
| BERT base uncased † | `bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | Apache 2.0 | ImageReward text encoder |

## Other checkpoints and code

| Asset | Source and version | License / terms | Used for |
|---|---|---|---|
| ImageReward v1.0 | `image-reward==1.5` (checkpoint `THUDM/ImageReward`) | Apache 2.0 | PF baseline |
| t2v_metrics | `t2v_metrics==1.2` | Apache 2.0 | VQAScore |
| HPSv3 package | `hpsv3==1.0.0` | MIT | HPSv3 |
| DINO ViT-S/8 | `torch.hub` `facebookresearch/dino:main`, `dino_vits8` | Apache 2.0 | DINO-I on ORIDa |
| DreamSim ensemble | `dreamsim==0.2.1`, NIGHTS-finetuned ensemble (DINO + CLIP + OpenCLIP) | MIT | DreamSim baseline |
| LoFTR (outdoor) | `kornia==0.8.3` | Apache 2.0 | §4.4 matcher control |
| RoMa | `romatch==0.1.2` | MIT; bundled DINOv2 Apache 2.0 | §4.4 matcher control |
| SuperPoint + LightGlue | `cvg/LightGlue` at `eb42fee2d71449efb0aa5c10549752b5d75384d8` | Apache 2.0 (LightGlue); MIT (SuperPoint re-implementation) | §4.4 matcher control |
| MASt3R | `naver/mast3r` at `f5209afc300cec36239a7ac992263f36847bbba0`; checkpoint `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`, sha256 `e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2` | CC BY-NC-SA 4.0, non-commercial only | §4.4 matcher control |
| spaCy and its English model † | `spacy==3.8.16`, `en_core_web_sm-3.8.0` | MIT | spaCy stripper option and PF prompt strata |

## LLM judges

| Judge | Source | Terms |
|---|---|---|
| GPT-4o, GPT-4V on DreamBench++ | ratings shipped with the DreamBench++ release (`data_gpt_rating.zip`) | as the dataset |
| GPT-4o on ORIDa | OpenAI API, `gpt-4o-2024-05-13`, the DreamBench++ subject rubric, both images at 512 × 512 with `detail: high`, temperature 1, a random seed per call | OpenAI Terms of Use |

The OpenAI snapshot `gpt-4o-2024-05-13` is scheduled for shutdown on 2026-10-23. The per-pair ORIDa ratings behind the paper's Table 2 were not retained.
