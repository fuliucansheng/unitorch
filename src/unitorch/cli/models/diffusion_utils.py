# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import torch
import torch.nn as nn
import hashlib
import tempfile
import numpy as np
import safetensors
from PIL import Image
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil, pt_to_pil, export_to_gif
from unitorch import is_safetensors_available, is_opencv_available
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
    replace_keys: Optional[Dict] = dict(),
    prefix_keys: Optional[Dict] = dict(),
):
    if isinstance(path, str):
        path = [path]

    state_dict = dict()
    for p in path:
        if p.endswith(".safetensors"):
            p = cached_path(p)
            state_dict = {**state_dict, **safetensors.torch.load_file(p)}
        else:
            p = cached_path(p)
            state_dict = {**state_dict, **torch.load(p, map_location="cpu")}

    results = dict()
    for key, value in list(state_dict.items()):
        for rkey, prefix in prefix_keys.items():
            if re.match(rkey, key):
                key = prefix + key
                break

        for rkey, nkey in replace_keys.items():
            key = re.sub(rkey, nkey, key)

        results[key] = value

    return results


def numpy2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    batch_size, channels, num_frames, height, width = video.shape
    mean = np.array(mean).reshape(1, -1, 1, 1, 1)
    std = np.array(std).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video * std + mean
    video = np.clip(video, 0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = np.transpose(video, (2, 3, 0, 4, 1)).reshape(f, h, i * w, c)
    images = np.split(images, f, axis=0)
    images = [(image.squeeze(0) * 255).astype("uint8") for image in images]  # f h w c
    return images


def export_to_video(
    video_frames: List[np.ndarray], output_video_path: str = None
) -> str:
    assert is_opencv_available(), "Please install python3-opencv first."
    import cv2

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path


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
        output_path = f"{self.output_folder}/{name}"
        image.save(output_path)
        return name

    def save_gif(self, images: List[Image.Image]):
        md5 = hashlib.md5()
        for image in images:
            md5.update(image.tobytes())
        name = md5.hexdigest() + ".gif"
        output_path = f"{self.output_folder}/{name}"
        export_to_gif(images, output_path)
        return name

    def save_video(self, video):
        md5 = hashlib.md5()
        for image in video:
            md5.update(image.tobytes())
        name = md5.hexdigest() + ".mp4"
        output_path = f"{self.output_folder}/{name}"
        export_to_video(video, output_path)
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

    @register_process("core/postprocess/diffusion/gif")
    def _diffusion_gif(
        self,
        outputs: DiffusionOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        images = outputs.outputs.numpy()
        images = [numpy_to_pil(image) for image in images]
        results["diffusion"] = [self.save_gif(image) for image in images]
        return WriterOutputs(results)

    @register_process("core/postprocess/diffusion/video")
    def _diffusion_video(
        self,
        outputs: DiffusionOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        videos = outputs.outputs.numpy()
        videos = [numpy2vid(video) for video in videos]
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
