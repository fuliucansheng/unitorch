# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.detr import DetrProcessor as _DetrProcessor
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli.models import (
    TensorInputs,
    TensorSeqInputs,
    DetectionOutputs,
    DetectionTargets,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models.detr import pretrained_detr_infos


class DetrProcessor(_DetrProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @config_defaults_init("core/process/detr")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/detr")
        pretrained_name = config.getoption("pretrained_name", "detr-resnet-50")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/detr/image")
    def _image(
        self,
        image: Union[Image.Image, str],
    ):
        outputs = super().image(
            image=image,
        )
        return TensorSeqInputs(
            images=outputs.image,
        )

    @register_process("core/process/detr/detection")
    def _detection(
        self,
        image: Union[Image.Image, str],
        bboxes: List[List[float]],
        classes: List[int],
        do_eval: Optional[bool] = False,
    ):
        outputs = super().detection(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )
        if do_eval:
            new_h, new_w = outputs.image.size()[1:]
            bboxes = outputs.bboxes
            bboxes[:, 0] = bboxes[:, 0] * new_w
            bboxes[:, 1] = bboxes[:, 1] * new_h
            bboxes[:, 2] = bboxes[:, 2] * new_w
            bboxes[:, 3] = bboxes[:, 3] * new_h
            return TensorSeqInputs(images=outputs.image), DetectionTargets(
                bboxes=bboxes,
                classes=outputs.classes,
            )

        return TensorSeqInputs(
            images=outputs.image,
            bboxes=outputs.bboxes,
            classes=outputs.classes,
        )

    @register_process("core/postprocess/detr/detection")
    def _post_dectection(self, outputs: DetectionOutputs):
        results = outputs.to_pandas()
        results["bboxes"] = [bboxes.tolist() for bboxes in outputs.bboxes]
        results["scores"] = [scores.tolist() for scores in outputs.scores]
        results["classes"] = [classes.tolist() for classes in outputs.classes]
        return WriterOutputs(results)
