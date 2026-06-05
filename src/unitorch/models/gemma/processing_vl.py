# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers.models.gemma4 import Gemma4ImageProcessor

from unitorch.models import GenericOutputs
from unitorch.models.gemma.processing import GemmaProcessor


class GemmaVLProcessor(GemmaProcessor):
    """
    Gemma processor for multimodal generation with image inputs.
    """

    def __init__(
        self,
        tokenizer_file: str,
        processor_config_path: str,
        tokenizer_config: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        max_gen_seq_length: Optional[int] = 512,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

        self.vision_processor = Gemma4ImageProcessor.from_json_file(
            processor_config_path
        )
        self.image_token = getattr(self.tokenizer, "image_token", "<|image|>")
        self.boi_token = getattr(self.tokenizer, "boi_token", "<|image>")
        self.eoi_token = getattr(self.tokenizer, "eoi_token", "<image|>")
        self.image_token_id = getattr(
            self.tokenizer,
            "image_token_id",
            self.tokenizer.convert_tokens_to_ids(self.image_token),
        )

    def _create_mm_token_type_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return mm_token_type_ids

    def processing_images(
        self,
        images: Union[Image.Image, str, Sequence[Union[Image.Image, str]]],
    ):
        if isinstance(images, (Image.Image, str)):
            images = [images]
        images = [
            image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
            for image in images
        ]
        return self.vision_processor(images=images, return_tensors="pt")

    def _prepare_text_with_images(
        self,
        text: str,
        num_soft_tokens_per_image: Sequence[int],
    ) -> str:
        text = str(text)
        image_count = len(num_soft_tokens_per_image)
        soft_token_placeholder = "<|gemma_image_soft_token|>"
        if image_count > 0 and self.image_token not in text:
            prefix = " ".join([self.image_token] * image_count)
            text = f"{prefix}\n{text}".strip()

        image_index = 0
        while self.image_token in text:
            if image_index >= image_count:
                raise ValueError(
                    "More image placeholders were found in the prompt than image inputs."
                )
            replacement = (
                f"{self.boi_token}"
                f"{soft_token_placeholder * int(num_soft_tokens_per_image[image_index])}"
                f"{self.eoi_token}"
            )
            text = text.replace(self.image_token, replacement, 1)
            image_index += 1

        if image_index != image_count:
            raise ValueError(
                "The number of image placeholders in the prompt does not match the number of image inputs."
            )

        return text.replace(soft_token_placeholder, self.image_token)

    def generation_inputs(
        self,
        text: str,
        images: Optional[
            Union[Image.Image, str, Sequence[Union[Image.Image, str]]]
        ] = None,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        image_inputs = self.processing_images(images) if images else None
        num_soft_tokens_per_image = (
            image_inputs["num_soft_tokens_per_image"].tolist() if image_inputs else []
        )
        text = self._prepare_text_with_images(text, num_soft_tokens_per_image)
        text_inputs = super().generation_inputs(
            text=text,
            max_seq_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            mm_token_type_ids=self._create_mm_token_type_ids(text_inputs.input_ids),
            pixel_values=(image_inputs["pixel_values"] if image_inputs else None),
            image_position_ids=(
                image_inputs["image_position_ids"] if image_inputs else None
            ),
        )

    def generation(
        self,
        text: str,
        images: Optional[
            Union[Image.Image, str, Sequence[Union[Image.Image, str]]]
        ],
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        image_inputs = self.processing_images(images) if images else None
        num_soft_tokens_per_image = (
            image_inputs["num_soft_tokens_per_image"].tolist() if image_inputs else []
        )
        text = self._prepare_text_with_images(text, num_soft_tokens_per_image)
        text_inputs = super().generation(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenericOutputs(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            mm_token_type_ids=self._create_mm_token_type_ids(text_inputs.input_ids),
            pixel_values=(image_inputs["pixel_values"] if image_inputs else None),
            image_position_ids=(
                image_inputs["image_position_ids"] if image_inputs else None
            ),
            input_ids_label=text_inputs.input_ids_label,
            attention_mask_label=text_inputs.attention_mask_label,
        )

    def messages_generation(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[
            Union[Image.Image, str, Sequence[Union[Image.Image, str]]]
        ] = None,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()

        text = self.chat_template(messages[:-1])
        text_pair = self.chat_template(messages[-1:])
        outputs = self.generation(
            text=text,
            images=images,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            mm_token_type_ids=outputs.mm_token_type_ids,
            pixel_values=outputs.pixel_values,
            image_position_ids=outputs.image_position_ids,
            input_ids_label=outputs.input_ids_label,
            attention_mask_label=outputs.attention_mask_label,
        )
