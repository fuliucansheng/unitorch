# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.grounding_dino import (
    GroundingDinoForDetection as _GroundingDinoForDetection,
)
from unitorch.models.grounding_dino import GroundingDinoProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.grounding_dino import pretrained_grounding_dino_infos


class GroundingDinoForDetectionPipeline(_GroundingDinoForDetection):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = GroundingDinoProcessor(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/grounding_dino")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "grounding-dino-tiny",
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/grounding_dino")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "config"
            ),
        )
        config_path = cached_path(config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vocab"
            ),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "weight"
            ),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path,
            vision_config_path,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/grounding_dino")
    def __call__(
        self,
        text: str,
        image: Union[Image.Image, str],
        text_threshold: Optional[float] = 0.25,
        box_threshold: Optional[float] = 0.25,
    ):
        inputs = self.processor.detection_inputs(text, image)
        outputs = self.detect(
            inputs.pixel_values.unsqueeze(0).to(self._device),
            inputs.input_ids.unsqueeze(0).to(self._device),
            inputs.token_type_ids.unsqueeze(0).to(self._device),
            inputs.attention_mask.unsqueeze(0).to(self._device),
            norm_bboxes=True,
            text_threshold=text_threshold,
            box_threshold=box_threshold,
        )
        result_image = image.copy()
        bboxes = outputs["bboxes"][0].cpu().numpy()
        scores = outputs["scores"][0].cpu().numpy()
        classes = outputs["classes"][0].cpu().numpy()
        # draw boxes on the result image with class and score
        for bbox, score, tokens in zip(bboxes, scores, classes):
            if score < box_threshold:
                continue
            classid = self.processor.tokenizer.decode(
                [t for t in tokens if t != 0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            bbox = bbox * np.array(
                [
                    result_image.width,
                    result_image.height,
                    result_image.width,
                    result_image.height,
                ]
            )
            bbox = list(map(int, bbox))
            result_image = result_image.copy()
            draw = ImageDraw.Draw(result_image)
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0] + 5, bbox[1] + 5), f"{classid}", fill="blue")
            draw.text((bbox[0] + 5, bbox[1] + 15), f"{score:.2f}", fill="green")

        return result_image
