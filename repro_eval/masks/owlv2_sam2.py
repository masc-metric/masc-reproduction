"""OWLv2 + SAM2: OWLv2 (text -> boxes) + SAM2 (boxes -> mask).

Same shape as `grounded_sam2.py`, with the open-vocab detector swapped
for OWLv2 — gives an independent point in the "different detector"
design space without changing the SAM2 head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    Sam2Model,
    Sam2Processor,
)


@dataclass
class Owlv2Sam2Segmenter:
    owl_processor: Owlv2Processor
    owl_model: Owlv2ForObjectDetection
    sam_processor: Sam2Processor
    sam_model: Sam2Model
    device: str
    score_threshold: float = 0.2
    mask_threshold: float = 0.0

    @torch.inference_mode()
    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        image = image.convert("RGB")
        w, h = image.size
        # OWLv2 takes a list of text queries per image. Plural-ish phrasing
        # tends to do better than a bare noun (paper convention).
        queries = [[f"a photo of a {prompt}"]]

        inputs = self.owl_processor(
            text=queries, images=image, return_tensors="pt"
        ).to(self.device)
        outputs = self.owl_model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]
        boxes = results["boxes"]  # (N, 4) xyxy in image coords
        if boxes.numel() == 0:
            return np.zeros((h, w), dtype=np.uint8)

        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes.cpu().tolist()],
            return_tensors="pt",
        ).to(self.device)
        sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            mask_threshold=self.mask_threshold,
            binarize=True,
        )[0]

        m = masks.squeeze(1).numpy().astype(bool)
        union = m.any(axis=0)
        return (union.astype(np.uint8)) * 255


def load_segmenter(
    owl_model_id: str = "google/owlv2-base-patch16-ensemble",
    sam_model_id: str = "facebook/sam2.1-hiera-large",
    score_threshold: float = 0.2,
    mask_threshold: float = 0.0,
    device: str | None = None,
) -> Owlv2Sam2Segmenter:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    owl_processor = Owlv2Processor.from_pretrained(owl_model_id)
    owl_model = Owlv2ForObjectDetection.from_pretrained(owl_model_id).to(device).eval()
    sam_processor = Sam2Processor.from_pretrained(sam_model_id)
    sam_model = Sam2Model.from_pretrained(sam_model_id).to(device).eval()
    return Owlv2Sam2Segmenter(
        owl_processor=owl_processor,
        owl_model=owl_model,
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )
