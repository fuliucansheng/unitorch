# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import BlipImageProcessor, BertTokenizer
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class BlipProcessor(
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
):
    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
        position_start_id: Optional[int] = 0,
    ):
        """
        Initializes the BlipProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            vision_config_path (str): The path to the vision configuration file.
            max_seq_length (Optional[int]): The maximum sequence length for text inputs. Defaults to 128.
            max_gen_seq_length (Optional[int]): The maximum sequence length for generated outputs. Defaults to 48.
            position_start_id (Optional[int]): The position start ID. Defaults to 0.
        """
        vision_processor = BlipImageProcessor.from_json_file(vision_config_path)
        HfImageClassificationProcessor.__init__(self, vision_processor=vision_processor)

        tokenizer = BertTokenizer(
            vocab_file=vocab_path,
        )
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
        """
        Performs text classification on the given input text.

        Args:
            text (str): The input text to classify.
            max_seq_length (Optional[int]): The maximum sequence length for the text. If None, the default value from initialization is used.

        Returns:
            GenericOutputs: The outputs of the text classification.
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
    ) -> GenericOutputs:
        """
        Performs image classification on the given input image.

        Args:
            image (PIL.Image.Image): The input image to classify.

        Returns:
            GenericOutputs: The outputs of the image classification.
        """
        outputs = HfImageClassificationProcessor.classification(
            self,
            image=image,
        )

        return GenericOutputs(
            pixel_values=outputs.pixel_values,
        )

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Performs classification using both text and image inputs.

        Args:
            text (str): The input text to classify.
            image (PIL.Image.Image): The input image to classify.
            max_seq_length (Optional[int]): The maximum sequence length for the text. If None, the default value from initialization is used.

        Returns:
            GenericOutputs: The outputs of the classification.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )

        text_outputs = self.text_classification(text, max_seq_length)
        pixel_outputs = self.image_classification(image)

        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            position_ids=text_outputs.position_ids,
            pixel_values=pixel_outputs.pixel_values,
        )

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Generate inputs for text generation.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated input tokens and attention mask.
        """
        outputs = HfTextGenerationProcessor.generation_inputs(
            self,
            text=text,
            max_seq_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Generates labels for text generation based on the given input text.

        Args:
            text (str): The input text for generating labels.
            max_gen_seq_length (Optional[int]): The maximum sequence length for the generated labels. If None, the default value from initialization is used.

        Returns:
            GenericOutputs: The generated labels.
        """
        outputs = HfTextGenerationProcessor.generation_labels(
            self,
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    def generation(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Generate inputs, labels, and tokens for image to text generation.

        Args:
            text (str): The input text.
            image (Image.Image): The input image to caption.
            max_gen_seq_length (int, optional): Maximum generated sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated input tokens, attention masks, label tokens, and attention masks.
        """

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
