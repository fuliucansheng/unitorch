# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import List, Optional, Union
from transformers import BertTokenizer, GroundingDinoImageProcessor
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
        Initializes the GroundingDinoProcessor.

        Args:
            vocab_path (str): Path to the BERT vocabulary file.
            vision_config_path (str): Path to the GroundingDINO image processor configuration file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
            position_start_id (int, optional): Starting position ID. Defaults to 0.
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
        Processes image and text for training with ground-truth detections.

        Args:
            text (str): Input text describing objects.
            image (str or PIL.Image.Image): Input image or path.
            bboxes (List[List[float]]): Ground-truth bounding boxes in [x1, y1, x2, y2] format.
            classes (List[str]): Class name for each bounding box.

        Returns:
            GenericOutputs: Processed inputs including pixel values, text tokens, boxes, and class IDs.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        org_w, org_h = image.size

        pixel_outputs = HfImageClassificationProcessor.classification(self, image)
        text_outputs = HfTextClassificationProcessor.classification(self, text)

        bboxes = torch.tensor(bboxes).float()
        bboxes[:, 0] /= org_w
        bboxes[:, 1] /= org_h
        bboxes[:, 2] /= org_w
        bboxes[:, 3] /= org_h

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

        assert bboxes.size(-1) == 4 and classes.dim() == 1 and len(classes) == len(bboxes)
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
        Processes image and text for inference.

        Args:
            text (str): Input text describing objects.
            image (str or PIL.Image.Image): Input image or path.

        Returns:
            GenericOutputs: Processed pixel values and text tokens.
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
