"""Grounded-SAM2: Grounding-DINO (text -> boxes) + SAM2 (boxes -> mask).

Box-prompted SAM2 with text grounding. Returns a uint8 {0, 255} mask at
the image's native (H, W); union of all SAM2 masks for the boxes whose
Grounding-DINO score crosses `box_threshold`. Empty mask if no detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


@dataclass
class GroundedSam2Segmenter:
    gdino_processor: AutoProcessor
    gdino_model: AutoModelForZeroShotObjectDetection
    sam_processor: Sam2Processor
    sam_model: Sam2Model
    device: str
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    mask_threshold: float = 0.0

    @torch.inference_mode()
    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        image = image.convert("RGB")
        w, h = image.size
        # Grounding-DINO expects lowercase, period-terminated text query.
        text = prompt.lower().strip()
        if not text.endswith("."):
            text = text + "."

        gd_inputs = self.gdino_processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)
        gd_outputs = self.gdino_model(**gd_inputs)
        results = self.gdino_processor.post_process_grounded_object_detection(
            gd_outputs,
            input_ids=gd_inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
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
        )[0]  # (N, 1, H, W) bool tensor

        m = masks.squeeze(1).numpy().astype(bool)
        union = m.any(axis=0)
        return (union.astype(np.uint8)) * 255


def load_segmenter(
    gdino_model_id: str = "IDEA-Research/grounding-dino-base",
    sam_model_id: str = "facebook/sam2.1-hiera-large",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    mask_threshold: float = 0.0,
    device: str | None = None,
) -> GroundedSam2Segmenter:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
    gdino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id)
        .to(device).eval()
    )
    sam_processor = Sam2Processor.from_pretrained(sam_model_id)
    sam_model = Sam2Model.from_pretrained(sam_model_id).to(device).eval()
    return GroundedSam2Segmenter(
        gdino_processor=gdino_processor,
        gdino_model=gdino_model,
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        mask_threshold=mask_threshold,
    )
