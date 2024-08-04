# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.sam import SamProcessor as _SamProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import SegmentationOutputs, TensorsInputs
from unitorch.cli.models.sam import pretrained_sam_infos


class SamProcessor(_SamProcessor):
    def __init__(
        self,
        vision_config_path: str,
        output_folder: Optional[str] = None,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )
        assert output_folder is not None
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

    @classmethod
    @add_default_section_for_init("core/process/sam")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/sam")
        pretrained_name = config.getoption("pretrained_name", "sam-vit-base")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    def save(self, image: Image.Image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        image.save(f"{self.output_folder}/{name}")
        return name

    @register_process("core/process/sam/segmentation/inputs")
    def _segmentation_inputs(
        self,
        image: Union[Image.Image, str],
        points_per_crop: Optional[int] = 32,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().segmentation_inputs(
            image=image,
            points_per_crop=points_per_crop,
        )
        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            original_sizes=outputs.original_sizes,
            reshaped_input_sizes=outputs.reshaped_input_sizes,
            input_points=outputs.input_points,
            input_labels=outputs.input_labels,
            input_boxes=outputs.input_boxes,
        )

    @register_process("core/postprocess/sam/segmentation")
    def _processing_masks(self, outputs: SegmentationOutputs):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.masks)
        results["mask_image"] = [
            ";".join([self.save(Image.fromarray(_m_.numpy())) for _m_ in m])
            for m in outputs.outputs
        ]
        return WriterOutputs(results)
