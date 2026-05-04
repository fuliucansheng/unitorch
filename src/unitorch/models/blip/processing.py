# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union

from PIL import Image
from transformers import BertTokenizer, BlipImageProcessor

from unitorch.models import (
    GenericOutputs,
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
)
from unitorch.utils import pop_value


class BlipProcessor(
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
):
    """Multimodal processor for BLIP models (text classification, image classification, captioning)."""

    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: int = 128,
        max_gen_seq_length: int = 48,
        position_start_id: int = 0,
    ) -> None:
        vision_processor = BlipImageProcessor.from_json_file(vision_config_path)
        HfImageClassificationProcessor.__init__(self, vision_processor=vision_processor)

        tokenizer = BertTokenizer(vocab_file=vocab_path)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.bos_token_id = tokenizer.cls_token_id
        tokenizer.eos_token = tokenizer.sep_token
        tokenizer.eos_token_id = tokenizer.sep_token_id

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=0,
            target_type_id=0,
            position_start_id=position_start_id,
        )
        HfTextGenerationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    def text_classification(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* for text classification."""
        outputs = HfTextClassificationProcessor.classification(self, text=text, max_seq_length=max_seq_length)
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
        )

    def image_classification(self, image: Union[Image.Image, str]) -> GenericOutputs:
        """Preprocess *image* for image classification."""
        return GenericOutputs(
            pixel_values=HfImageClassificationProcessor.classification(self, image=image).pixel_values,
        )

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Preprocess a text-image pair for multimodal classification."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        text_out = self.text_classification(text, max_seq_length)
        pixel_out = self.image_classification(image)
        return GenericOutputs(
            input_ids=text_out.input_ids,
            attention_mask=text_out.attention_mask,
            position_ids=text_out.position_ids,
            pixel_values=pixel_out.pixel_values,
        )

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* as encoder generation inputs."""
        outputs = HfTextGenerationProcessor.generation_inputs(self, text=text, max_seq_length=max_seq_length)
        return GenericOutputs(input_ids=outputs.input_ids, attention_mask=outputs.attention_mask)

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* as generation labels."""
        outputs = HfTextGenerationProcessor.generation_labels(self, text=text, max_gen_seq_length=max_gen_seq_length)
        return GenericOutputs(input_ids=outputs.input_ids, attention_mask=outputs.attention_mask)

    def generation(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Preprocess an image-text pair for captioning (image-to-text generation)."""
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)
        tokens = self.generation_inputs(text, max_gen_seq_length)
        pixels = self.image_classification(image)
        labels = self.generation_labels(text, max_gen_seq_length)
        return GenericOutputs(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            pixel_values=pixels.pixel_values,
            input_ids_label=labels.input_ids,
            attention_mask_label=labels.attention_mask,
        )
