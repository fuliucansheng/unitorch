# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import CLIPImageProcessor, CLIPTokenizer
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    GenericOutputs,
)


class ClipProcessor(HfImageClassificationProcessor, HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        """
        Initializes the ClipProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            vision_config_path (str): The path to the vision configuration file.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to 128.
            position_start_id (int, optional): The starting position ID for positional embeddings. Defaults to 0.
        """
        if vision_config_path is not None:
            vision_processor = CLIPImageProcessor.from_json_file(vision_config_path)
        else:
            vision_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        HfImageClassificationProcessor.__init__(
            self,
            vision_processor=vision_processor,
        )

        if vocab_path is not None and merge_path is not None:
            tokenizer = CLIPTokenizer(
                vocab_file=vocab_path,
                merges_file=merge_path,
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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
        pixel_outputs = self.image_classification(
            image=image,
        )

        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            position_ids=text_outputs.position_ids,
            pixel_values=pixel_outputs.pixel_values,
        )
