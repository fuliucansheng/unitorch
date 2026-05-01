# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Optional, Union
from transformers import SiglipImageProcessor, SiglipTokenizer
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    GenericOutputs,
)


class SiglipProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        """
        Initializes the SiglipProcessor.

        Args:
            vocab_path (str, optional): Path to the vocabulary file.
            vision_config_path (str, optional): Path to the vision configuration file.
            max_seq_length (int, optional): Maximum sequence length for text inputs. Defaults to 128.
            position_start_id (int, optional): Starting position ID. Defaults to 0.
        """
        if vocab_path is not None:
            tokenizer = SiglipTokenizer(vocab_file=vocab_path)
        else:
            tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token

        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=0,
            target_type_id=0,
            position_start_id=position_start_id,
        )

        if vision_config_path is not None:
            self.vision_processor = SiglipImageProcessor.from_json_file(vision_config_path)
        else:
            self.vision_processor = SiglipImageProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
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
            self,
            text=text,
            max_seq_length=max_seq_length,
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
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]
        return GenericOutputs(pixel_values=pixel_values)

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        """
        Processes text and image for multimodal classification.

        Args:
            text (str): Input text.
            image (PIL.Image.Image or str): Input image or path.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed text and image inputs.
        """
        text_outputs = self.text_classification(text=text, max_seq_length=max_seq_length)
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]
        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            position_ids=text_outputs.position_ids,
            pixel_values=pixel_values,
        )
