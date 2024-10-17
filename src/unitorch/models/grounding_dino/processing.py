# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from hmac import new
import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import BertTokenizer, GroundingDinoImageProcessor
from unitorch.utils import resize_shortest_edge, pop_value
from unitorch.models import (
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class GroundingDinoProcessor(
    HfTextClassificationProcessor, HfImageClassificationProcessor
):
    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        """
        Args:
            vision_config_path: vision config path to detr processor
            min_size_test: resize shortest edge parameters
            max_size_test: resize shortest edge parameters
        """
        self.bert_tokenizer = BertTokenizer(
            vocab_path, do_basic_tokenize=True, do_lower_case=True
        )
        self.vision_processor = GroundingDinoImageProcessor.from_json_file(
            vision_config_path
        )

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=self.bert_tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=0,
            target_type_id=1,
            position_start_id=position_start_id,
        )

        HfImageClassificationProcessor.__init__(
            self,
            vision_processor=self.vision_processor,
        )

    def detection(
        self,
        text: str,
        image: Union[str, Image.Image],
        bboxes: List[List[float]],
        classes: List[str],
    ):
        """
        Args:
            image: input image
            bboxes: bboxes to image
            classes: class to each bbox
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        org_w, org_h = image.size
        pixel_outputs = HfImageClassificationProcessor.classification(self, image)
        text_outputs = HfTextClassificationProcessor.classification(self, text)

        bboxes = torch.tensor(bboxes).float()
        bboxes[:, 0] = bboxes[:, 0] / org_w
        bboxes[:, 1] = bboxes[:, 1] / org_h
        bboxes[:, 2] = bboxes[:, 2] / org_w
        bboxes[:, 3] = bboxes[:, 3] / org_h

        assert all(c in text for c in classes)
        ground_truth = text_outputs.input_ids.long().tolist()
        class_ids = []
        for c in classes:
            class_tokens = self.tokenizer.tokenize(c)
            class_token_ids = self.tokenizer.convert_tokens_to_ids(class_tokens)
            class_idx = ground_truth.index(class_token_ids[0])
            class_ids.append(class_idx)

        classes = torch.tensor(class_ids)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)

        assert (
            bboxes.size(-1) == 4 and classes.dim() == 1 and len(classes) == len(bboxes)
        )
        return GenericOutputs(
            pixel_values=pixel_outputs.pixel_values,
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            token_type_ids=text_outputs.token_type_ids,
            bboxes=bboxes,
            classes=classes,
        )

    def detection_inputs(
        self,
        text: str,
        image: Union[str, Image.Image],
    ):
        """
        Args:
            image: input image
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        pixel_outputs = HfImageClassificationProcessor.classification(self, image)
        text_outputs = HfTextClassificationProcessor.classification(self, text)
        return GenericOutputs(
            pixel_values=pixel_outputs.pixel_values,
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            token_type_ids=text_outputs.token_type_ids,
        )
