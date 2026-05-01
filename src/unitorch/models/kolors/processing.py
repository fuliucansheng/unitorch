# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Optional, Union
from transformers import CLIPImageProcessor, CLIPTokenizer
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    GenericOutputs,
)


class KolorsMPSProcessor(HfImageClassificationProcessor, HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
    ):
        """
        Initializes the KolorsMPSProcessor.

        Args:
            vocab_path (str, optional): Path to the vocabulary file.
            merge_path (str, optional): Path to the merge file.
            vision_config_path (str, optional): Path to the vision processor configuration file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 77.
            position_start_id (int, optional): Starting position ID. Defaults to 0.
        """
        if vision_config_path is not None:
            vision_processor = CLIPImageProcessor.from_json_file(vision_config_path)
        else:
            vision_processor = CLIPImageProcessor.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            )
        HfImageClassificationProcessor.__init__(self, vision_processor=vision_processor)

        if vocab_path is not None and merge_path is not None:
            tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=merge_path)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
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
    ):
        """
        Processes text for classification.

        Args:
            text (str): Input text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed text inputs.
        """
        outputs = HfTextClassificationProcessor.classification(
            self, text=text, max_seq_length=max_seq_length
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
        )

    def image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Processes an image for classification.

        Args:
            image (PIL.Image.Image or str): Input image or path.

        Returns:
            GenericOutputs: Processed image inputs.
        """
        outputs = HfImageClassificationProcessor.classification(self, image=image)
        return GenericOutputs(pixel_values=outputs.pixel_values)

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        condition: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Processes text, image, and condition for multimodal classification.

        Args:
            text (str): Input text.
            image (PIL.Image.Image or str): Input image or path.
            condition (str): Condition text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed text, image, and condition inputs.
        """
        text_outputs = self.text_classification(text=text, max_seq_length=max_seq_length)
        pixel_outputs = self.image_classification(image=image)
        condition_outputs = self.text_classification(text=condition, max_seq_length=max_seq_length)
        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            position_ids=text_outputs.position_ids,
            pixel_values=pixel_outputs.pixel_values,
            condition_input_ids=condition_outputs.input_ids,
            condition_attention_mask=condition_outputs.attention_mask,
            condition_position_ids=condition_outputs.position_ids,
        )
