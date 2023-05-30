# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import hashlib
from PIL import Image
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models.modeling_utils import TensorsOutputs, TensorsTargets


@dataclass
class DiffusionOutputs(TensorsOutputs, WriterMixin):
    outputs: torch.Tensor


@dataclass
class DiffusionTargets(TensorsTargets):
    targets: torch.Tensor
    masks: Optional[torch.Tensor] = torch.empty(0)


class DiffusionProcessor:
    def __init__(self, image_folder: Optional[str] = None):
        assert image_folder is not None
        self.image_folder = image_folder
        if not os.path.exists(image_folder):
            os.makedirs(self.image_folder, exist_ok=True)

    @classmethod
    @add_default_section_for_init("core/process/diffusion")
    def from_core_configure(cls, config, **kwargs):
        pass

    def save(self, image: Image.Image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        image.save(f"{self.image_folder}/{name}")
        return name

    @register_process("core/postprocess/diffusion")
    def _diffusion(
        self,
        outputs: DiffusionOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        images = outputs.outputs.numpy().transpose(0, 2, 3, 1)
        images = numpy_to_pil(images)
        results["diffusion"] = [self.save(image) for image in images]
        return WriterOutputs(results)
