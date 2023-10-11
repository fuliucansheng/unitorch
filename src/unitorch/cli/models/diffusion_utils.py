# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import hashlib
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import safetensors
from diffusers.utils import numpy_to_pil
from diffusers.utils import export_to_video
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models.modeling_utils import TensorsOutputs, TensorsTargets

from unitorch.cli import cached_path


def load_weight(
    path,
    prefix: Optional[str] = "",
):
    if path.endswith(".safetensors"):
        path = cached_path(path)
        state_dict = safetensors.torch.load_file(path)
    else:
        path = cached_path(path)
        state_dict = torch.load(path, map_location="cpu")
    state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}
    return state_dict


@dataclass
class DiffusionOutputs(TensorsOutputs, WriterMixin):
    outputs: torch.Tensor


@dataclass
class DiffusionTargets(TensorsTargets):
    targets: torch.Tensor
    masks: Optional[torch.Tensor] = torch.empty(0)


class DiffusionProcessor:
    def __init__(self, output_folder: Optional[str] = None):
        assert output_folder is not None
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

    @classmethod
    @add_default_section_for_init("core/process/diffusers")
    def from_core_configure(cls, config, **kwargs):
        pass

    def save_image(self, image: Image.Image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        image.save(f"{self.output_folder}/{name}")
        return name

    def save_video(self, video: List[np.ndarray]):
        md5 = hashlib.md5()
        for image in video:
            md5.update(image.tobytes())
        name = md5.hexdigest() + ".mp4"
        export_to_video(video_frames=video, video_name=f"{self.output_folder}/{name}")
        return name

    @register_process("core/postprocess/diffusion/image")
    def _diffusion_image(
        self,
        outputs: DiffusionOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        images = outputs.outputs.numpy()
        images = numpy_to_pil(images)
        results["diffusion"] = [self.save_image(image) for image in images]
        return WriterOutputs(results)

    @register_process("core/postprocess/diffusion/video")
    def _diffusion_video(
        self,
        outputs: DiffusionOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        videos = outputs.outputs.numpy()
        results["diffusion"] = [self.save_video(video) for video in videos]
        return WriterOutputs(results)


def diffusion_model_decorator(cls):
    class DiffusionModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__diffusion_model__" in kwargs:
                self.model = kwargs.pop("__diffusion_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

            assert hasattr(self.model, "generate")

            self.model.register_forward_hook(self._hook)
            self.__in_training__ = False

        def _hook(self, module, inputs, outputs):
            self.__in_training__ = True

        def forward(self, *args, **kwargs):
            if self.training or self.__in_training__:
                return self.model(*args, **kwargs)
            return self.model.generate(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__diffusion_model__=model)

    return DiffusionModel
