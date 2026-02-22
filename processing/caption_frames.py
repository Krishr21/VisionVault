from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image


class FrameCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        # BLIP-2 is heavy for MVP; this default is lighter and open-source.
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def caption_image(self, image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()


def caption_frames(frame_paths: List[Path], model_name: str = "Salesforce/blip-image-captioning-base") -> Dict[str, str]:
    captioner = FrameCaptioner(model_name=model_name)
    results: Dict[str, str] = {}
    for path in frame_paths:
        results[path.name] = captioner.caption_image(path)
    return results
