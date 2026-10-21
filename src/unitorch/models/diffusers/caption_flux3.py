# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""An explicit, separate source-only alternative to the official 9B PUP."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch
from transformers import (
    Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration, Qwen3VLVideoProcessor,
)


ASSETS_CAPTION_SYSTEM = """You write the full target-scene caption consumed by an image generation model.
You receive ONLY a source photograph and an editing instruction. Inspect the source and follow the instruction.
For apparel extraction, describe an overhead studio product photograph on an uninterrupted white background.
Inventory EVERY visible garment and accessory from the source, including footwear, bags, belts, scarves,
hats and jewelry when actually visible. Preserve each item's category, color, fabric, pattern, trim,
construction and proportions. Describe trousers as trousers, never a skirt. Keep a dress or jumpsuit
as one complete garment. Do not invent hidden items or guess logos, brands, identities or unreadable text.
Arrange separate complete items without overlap, centered with space between them, suitable for a tall 2:3 canvas.
State frame-relative positions and preserve fine lace, seams and textures with even diffuse illumination
and sharp focus across all items. Describe the desired output scene, not instructions about editing.
The photograph contains no person, mannequin, body parts, original setting or added decorative objects.
Return ONE self-contained English paragraph, normally 180-300 words. Begin 'Studio photograph, overhead view'.
Use concrete observable descriptions without marketing adjectives, markdown, explanation or alternative guesses.
This is a caption, not a garment segmentation list. Output the caption only."""


class Flux3SourceCaptioner:
    """Qwen3-VL caption stage, loaded separately from the diffusion model.

    Uses the already licensed base text encoder as an alternative VLM, not the
    official 9B PUP. Its identity records this distinction and the prompt hash.
    Production callers persist captions with datasets.flux3_assets.prepare_captions.
    """

    def __init__(self, text_encoder_path: str, device: str = "cuda", max_new_tokens: int = 768,
                 max_pixels: int = 1024 * 1536, system_prompt_path=None):
        root = Path(text_encoder_path)
        self.system_prompt = Path(system_prompt_path).read_text() if system_prompt_path else ASSETS_CAPTION_SYSTEM
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(root, dtype=torch.bfloat16, local_files_only=True).eval().to(device)
        tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True)
        template_path = root / "chat_template.json"
        template = json.loads(template_path.read_text())["chat_template"] if template_path.exists() else tokenizer.chat_template
        image_processor = Qwen2VLImageProcessor.from_pretrained(root, local_files_only=True)
        # Bound only VLM visual tokens; source/target training geometry is unchanged.
        image_processor.size["longest_edge"] = max_pixels
        video_processor = Qwen3VLVideoProcessor.from_pretrained(root, local_files_only=True)
        self.processor = Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                                         video_processor=video_processor, chat_template=template)
        self.max_new_tokens = max_new_tokens
        self.identity = "Qwen3-VL-4B-source-only:" + hashlib.sha256(json.dumps({
            "system": self.system_prompt, "max_new_tokens": max_new_tokens,
            "max_pixels": max_pixels, "text_config": json.loads((root / "config.json").read_text()),
            "base_revision": "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86",
        }, sort_keys=True).encode()).hexdigest()

    @torch.inference_mode()
    def __call__(self, source_path: str, instruction: str):
        with Image.open(source_path) as handle:
            source = ImageOps.exif_transpose(handle).convert("RGB")
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user", "content": [{"type": "image", "image": source},
                                                   {"type": "text", "text": instruction}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        caption_tokens = output[:, inputs["input_ids"].shape[1]:]
        if caption_tokens.shape[1] >= self.max_new_tokens:
            raise ValueError("Caption hit max_new_tokens; refusing an incomplete scene caption")
        return self.processor.tokenizer.decode(caption_tokens[0], skip_special_tokens=True).strip()
