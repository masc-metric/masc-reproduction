"""AM-RADIO summary-token cosine baseline.

Canonical use of a foundation model as a global-image-similarity metric:
take the model's CLS-equivalent global descriptor (RADIO calls this
`summary`), L2-normalize, cosine. Direct analog of how DreamBench++ uses
CLIP-I (`visual_projection(vision_model.pooler_output)` cosine) and DINO-I
(`dino_vits8` CLS cosine).

Preprocessing matches the canonical RADIO inference path exactly:
  - shortest-edge resize to 512 (BICUBIC; matches RADIO's CLIPImageProcessor
    `size.shortest_edge=512`, `resample=3`).
  - center-crop to 512x512 (matches `crop_size=(512, 512)`); this is also
    consistent with how CLIP-I and DINO-I produce a square global descriptor
    from arbitrary-aspect-ratio inputs.
  - rescale to [0, 1] only. **Do NOT externally apply mean/std normalization** —
    `radio_model.RADIOModel.forward` documents that input is expected in [0, 1]
    range; the model's internal `InputConditioner` (initialized with OpenAI CLIP
    mean/std per `input_conditioner.get_default_conditioner`) handles
    normalization. Pre-normalizing externally with the same mean/std results in
    double-normalization and silently degrades scores.

Score: cosine between L2-normalized `summary` tensors. NOTE: distinct from the
masked-maxcos backbone-ablation in `radio.py`; that one tests "what if we plug
RADIO patch features into our recipe." This module implements RADIO's natural
baseline use as an image descriptor, with no masks and no patch matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

DEFAULT_MODEL_ID = "nvidia/C-RADIOv4-SO400M"
DEFAULT_INPUT_SIZE = 512  # native preferred resolution per HF model card


@dataclass
class RadioSummaryMatcher:
    model: torch.nn.Module
    transform: T.Compose
    input_size: int
    device: str

    @torch.inference_mode()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        # Match the canonical RADIO inference pipeline:
        #   resize(shortest_edge=input_size, BICUBIC) -> CenterCrop -> ToTensor.
        # No mean/std normalization — the model's InputConditioner does it.
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # bf16 autocast: matches RADIO's training amp_dtype.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(t)
        # RadioOutput namedtuple: (.summary, .features).
        summary = out.summary if hasattr(out, "summary") else out[0]
        return F.normalize(summary.float(), dim=-1)[0]

    def fg_match(self, ref_image, ref_mask, out_image, out_mask) -> dict:
        # Masks unused — global summary descriptor cosine.
        ref = self._embed(ref_image)
        out = self._embed(out_image)
        cos = float((ref * out).sum().item())
        return {
            "score": cos,
            "extras": {
                "score_type": "summary_cosine",
                "cos_x100": cos * 100.0,
            },
        }


def load_matcher(
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    device: str = "cuda",
) -> RadioSummaryMatcher:
    # Load the processor only to confirm the canonical preprocessing
    # parameters (shortest_edge, crop_size, resample) — we don't actually
    # use it in the forward pass because we need to pass [0, 1] tensors
    # to the model and PIL-side transforms compose more cleanly than
    # processor-side toggles.
    proc = CLIPImageProcessor.from_pretrained(model_id)
    if int(proc.size["shortest_edge"]) != input_size:
        raise ValueError(
            f"radio_summary: input_size={input_size} disagrees with processor "
            f"shortest_edge={proc.size['shortest_edge']}"
        )
    if int(proc.crop_size["height"]) != input_size or int(proc.crop_size["width"]) != input_size:
        raise ValueError(
            f"radio_summary: input_size={input_size} disagrees with processor "
            f"crop_size={proc.crop_size}"
        )

    transform = T.Compose([
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),  # rescales [0, 255] -> [0, 1]; no mean/std normalization
    ])

    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True)
        .eval()
        .to(device)
    )
    return RadioSummaryMatcher(
        model=model, transform=transform, input_size=input_size, device=device
    )
