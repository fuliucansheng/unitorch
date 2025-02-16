# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import SiglipImageProcessor, SiglipTokenizer
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
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
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            vision_config_path (str): The path to the vision configuration file.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to 128.
            position_start_id (int, optional): The starting position ID for positional embeddings. Defaults to 0.
        """
        if vocab_path is not None:
            tokenizer = SiglipTokenizer(
                vocab_file=vocab_path,
            )
        else:
            tokenizer = SiglipTokenizer.from_pretrained(
                "google/siglip-base-patch16-224"
            )
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
            self.vision_processor = SiglipImageProcessor.from_json_file(
                vision_config_path
            )
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
        Performs text classification.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to None.

        Returns:
            GenericOutputs: An object containing the processed inputs.
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
        Performs image classification.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            GenericOutputs: An object containing the processed inputs.
        """
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        return GenericOutputs(
            pixel_values=pixel_values,
        )

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        """
        Performs classification using text and image inputs.

        Args:
            text (str): The input text.
            image (PIL.Image.Image): The input image.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to None.

        Returns:
            GenericOutputs: An object containing the processed inputs.
        """
        text_outputs = self.text_classification(
            text=text,
            max_seq_length=max_seq_length,
        )
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            position_ids=text_outputs.position_ids,
            pixel_values=pixel_values,
        )
