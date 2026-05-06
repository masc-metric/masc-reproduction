# MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation

Anonymous repository accompanying the paper submission. This repo contains
**both** the released `masc-metric` package source and the scripts used
to reproduce every comparator's published numbers in the paper.

![MaSC method diagram](assets/method_diagram.png)

MaSC is a non-LLM evaluation metric for single-concept text-to-image
personalization. One forward pass of `google/siglip2-so400m-patch16-naflex`
per image plus pre-computed segmentation masks yields **two scores** from
the same patch-token tensor:

- **Concept Preservation (CP)** — masked-maxcos over the foreground
  concept region.
- **Prompt Following (PF)** — cosine between a subject-stripped prompt
  embedding and a background-pooled image embedding.

## Repository layout

```
masc-reproduction/
├── src/masc/                    # MaSC package source (vendored, the released artifact)
├── examples/                    # end-to-end MaSC scoring example on a single sample
├── tests/                       # pure-Python smoke tests for the package
├── repro_eval/                  # reproduction helpers: dataset builders, comparator modules
│   ├── data.py                  # DreamBench++ + ORIDa sample builders
│   ├── geometry/                # CP comparators + masked-maxcos matcher ablations
│   ├── pf/                      # PF comparator scorers (VQAScore, ImageReward, HPSv3)
│   └── masks/                   # SAM3 text-prompted segmenter
├── scripts/                     # reproduction CLIs (compute, plot, benchmark)
├── configs/                     # one YAML per (matcher, dataset)
└── assets/                      # method diagram referenced from README
```

## Install

```bash
git clone <this-repo-url>
cd masc-reproduction
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# install masc + reproduction extras (pyyaml, scipy, matplotlib, dotenv)
pip install -e ".[repro]"

# `masc-metric` requires torch >= 2.6 (transformers 5.6 requires it for
# non-safetensors checkpoint loading). Use whichever torch + cuda
# combination matches your GPU; the recipe below is one known-good
# pairing. See https://pytorch.org/get-started/locally/ for others.
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu128
```

You will also need a HuggingFace token with manual-gate access approved
for `google/siglip2-so400m-patch16-naflex` and `facebook/sam3`. Put it in
a `.env` file at the repo root:

```
HF_TOKEN=hf_...
```

`repro_eval/__init__.py` loads `.env` via `python-dotenv` on import, so
scripts that `from repro_eval...` pick up the token automatically.

## Quick start: score one sample with MaSC

```python
import numpy as np
from PIL import Image
from masc import MaSC

masc = MaSC()  # default: siglip2-so400m-patch16-naflex; auto-picks cuda > mps > cpu

ref_image = Image.open("ref.png")
out_image = Image.open("out.png")
ref_mask = np.array(Image.open("ref_mask.png").convert("L"))
out_mask = np.array(Image.open("out_mask.png").convert("L"))

result = masc.score(
    ref_image=ref_image,
    ref_mask=ref_mask,
    out_image=out_image,
    out_mask=out_mask,
    prompt="a photo of a cat on the beach",
    object_name="cat",  # subject stripped from prompt for the PF head
)

print(result.cp, result.pf)
print(result.extras)
```

A runnable end-to-end version is in [examples/score_sample.py](examples/score_sample.py),
which scores a real (reference, output, prompt) sample shipped under
[examples/sample_hat/](examples/sample_hat/).

## Reproducing the paper

Each numbered table/section in the paper has a small set of scripts and
configs that regenerate it from raw data. See
[scripts/REPRODUCE.md](scripts/REPRODUCE.md) for the full step-by-step
recipe (datasets to download, mask generation, per-comparator scoring,
α / ρ / AUC aggregation).

Datasets used:

- **DreamBench++** — primary CP and PF evaluation benchmark
  (https://github.com/yuangpeng/dreambench_plus). Place under
  `data/dreambench_plus/`.
- **ORIDa** (https://arxiv.org/abs/2506.08964) — real-photo identity
  discrimination benchmark. Place under `data/ORIDa/ORIDa_v1.0/`.

## License

Apache-2.0. See [LICENSE](LICENSE).
