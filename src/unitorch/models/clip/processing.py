# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import inspect
from typing import Optional, Union

from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

from unitorch.models import (
    GenericOutputs,
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
)


class ClipProcessor(HfImageClassificationProcessor, HfTextClassificationProcessor):
    """Multimodal processor for CLIP models."""

    @staticmethod
    def _build_tokenizer(vocab_path: str, merge_path: str) -> CLIPTokenizer:
        init_params = inspect.signature(CLIPTokenizer.__init__).parameters
        if "vocab_file" in init_params and "merges_file" in init_params:
            return CLIPTokenizer(vocab_file=vocab_path, merges_file=merge_path)
        return CLIPTokenizer(vocab=vocab_path, merges=merge_path)

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: int = 128,
        position_start_id: int = 0,
    ) -> None:
        vision_processor = (
            CLIPImageProcessor.from_json_file(vision_config_path)
            if vision_config_path is not None
            else CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        )
        HfImageClassificationProcessor.__init__(self, vision_processor=vision_processor)

        tokenizer = (
            self._build_tokenizer(vocab_path=vocab_path, merge_path=merge_path)
            if vocab_path is not None and merge_path is not None
            else CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        )
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=0,
            target_type_id=0,
            position_start_id=position_start_id,
        )

    def text_classification(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* for text classification."""
        outputs = HfTextClassificationProcessor.classification(
            self, text=text, max_seq_length=max_seq_length
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
        )

    def image_classification(self, image: Union[Image.Image, str]) -> GenericOutputs:
        """Preprocess *image* for image classification."""
        return GenericOutputs(
            pixel_values=HfImageClassificationProcessor.classification(
                self, image=image
            ).pixel_values,
        )

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Preprocess a text-image pair for multimodal classification."""
        text_out = self.text_classification(text=text, max_seq_length=max_seq_length)
        pixel_out = self.image_classification(image=image)
        return GenericOutputs(
            input_ids=text_out.input_ids,
            attention_mask=text_out.attention_mask,
            position_ids=text_out.position_ids,
            pixel_values=pixel_out.pixel_values,
        )
