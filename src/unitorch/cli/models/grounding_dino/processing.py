# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.grounding_dino import (
    GroundingDinoProcessor as _GroundingDinoProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    TensorsInputs,
    ListTensorsInputs,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models.grounding_dino import pretrained_grounding_dino_infos


class GroundingDinoProcessor(_GroundingDinoProcessor):
    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/grounding_dino")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/grounding_dino")
        pretrained_name = config.getoption("pretrained_name", "grounding-dino-tiny")

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vocab"
            ),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/grounding_dino/detection/inputs")
    def _detection_inputs(
        self,
        text: str,
        image: Union[Image.Image, str],
    ):
        outputs = super().detection_inputs(
            text=text,
            image=image,
        )

        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            token_type_ids=outputs.token_type_ids,
        )

    @register_process("core/process/grounding_dino/detection")
    def _detection(
        self,
        text: str,
        image: Union[Image.Image, str],
        bboxes: List[List[float]],
        classes: List[str],
        do_eval: Optional[bool] = False,
    ):
        outputs = super().detection(
            text=text,
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        if do_eval:
            new_h, new_w = outputs.pixel_values.size()[1:]
            bboxes = outputs.bboxes
            bboxes[:, 0] = bboxes[:, 0] * new_w
            bboxes[:, 1] = bboxes[:, 1] * new_h
            bboxes[:, 2] = bboxes[:, 2] * new_w
            bboxes[:, 3] = bboxes[:, 3] * new_h
            return TensorsInputs(
                pixel_values=outputs.pixel_values,
                input_ids=outputs.input_ids,
                attention_mask=outputs.attention_mask,
                token_type_ids=outputs.token_type_ids,
            ), DetectionTargets(
                bboxes=bboxes,
                classes=outputs.classes,
            )

        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            token_type_ids=outputs.token_type_ids,
        ), ListTensorsInputs(
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/postprocess/grounding_dino/detection")
    def _post_dectection(self, outputs: DetectionOutputs):
        results = outputs.to_pandas()
        results["bboxes"] = [bboxes.tolist() for bboxes in outputs.bboxes]
        results["scores"] = [scores.tolist() for scores in outputs.scores]
        results["classes"] = [classes.tolist() for classes in outputs.classes]
        return WriterOutputs(results)
