# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union

from PIL import Image
from transformers import BertTokenizer, ChineseCLIPImageProcessor

from unitorch.models import (
    GenericOutputs,
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
)


class ChineseClipProcessor(HfImageClassificationProcessor, HfTextClassificationProcessor):
    """Multimodal processor for Chinese CLIP models."""

    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: int = 128,
        position_start_id: int = 0,
    ) -> None:
        HfImageClassificationProcessor.__init__(
            self,
            vision_processor=ChineseCLIPImageProcessor.from_json_file(vision_config_path),
        )

        tokenizer = BertTokenizer(vocab_file=vocab_path)
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=0,
            target_type_id=1,
            position_start_id=position_start_id,
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
        text_out = self.text_classification(text=text, max_seq_length=max_seq_length)
        pixel_out = self.image_classification(image=image)
        return GenericOutputs(
            input_ids=text_out.input_ids,
            attention_mask=text_out.attention_mask,
            token_type_ids=text_out.token_type_ids,
            position_ids=text_out.position_ids,
            pixel_values=pixel_out.pixel_values,
        )
