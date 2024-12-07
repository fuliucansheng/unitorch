# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import torch
import hashlib
import logging
import requests
import numpy as np
import torch.nn as nn
from PIL import Image
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import numpy_to_pil
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models.modeling_utils import ListTensorsOutputs, ListTensorsTargets


@dataclass
class SegmentationOutputs(ListTensorsOutputs, WriterMixin):
    masks: Union[torch.Tensor, List[torch.Tensor]]
    classes: Union[torch.Tensor, List[torch.Tensor]] = torch.empty(0)


@dataclass
class SegmentationTargets(ListTensorsTargets):
    targets: Union[torch.Tensor, List[torch.Tensor]]
    sample_weight: Optional[torch.Tensor] = torch.empty(0)


class SegmentationProcessor:
    def __init__(
        self,
        mask_threshold: float = None,
        output_folder: Optional[str] = None,
        http_url: Optional[str] = None,
    ):
        self.mask_threshold = mask_threshold

        self.output_folder = output_folder
        if self.output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        self.http_url = http_url

    @classmethod
    @add_default_section_for_init("core/process/segmentation")
    def from_core_configure(cls, config, **kwargs):
        pass

    def save_image(self, image: Image.Image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        if self.http_url is not None:
            byte = io.BytesIO()
            image.save(byte, format="JPEG")
            for _ in range(3):
                resp = requests.post(
                    self.http_url.format(name), files={"file": byte.getvalue()}
                )
                if resp.status_code == 200:
                    break
            if resp.status_code != 200:
                logging.error(f"Failed to save image {name} to zip.")
        else:
            output_path = f"{self.output_folder}/{name}"
            image.save(output_path)
        return name

    @register_process("core/postprocess/segmentation")
    def _segmentation(
        self,
        outputs: SegmentationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.masks)
        mask0 = outputs.masks[0].numpy()
        if (
            mask0.shape[-1] == 1 or mask0.ndim == 2
        ) and self.mask_threshold is not None:
            results["pixel_class"] = [
                (m.numpy() > self.mask_threshold).astype(np.int16).reshape(-1).tolist()
                for m in outputs.masks
            ]
        else:
            results["pixel_class"] = [
                m.numpy().argmax(-1).reshape(-1).tolist() for m in outputs.masks
            ]
        return WriterOutputs(results)

    @register_process("core/postprocess/segmentation/mask")
    def _segmentation_mask(
        self,
        outputs: SegmentationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.masks)
        if self.mask_threshold is not None:
            masks = [
                (m.numpy() < self.mask_threshold).astype(np.int16)
                for m in outputs.masks
            ]
        else:
            masks = [m.numpy() for m in outputs.masks]
        results["mask_image"] = [
            self.save_image(numpy_to_pil(mask if mask.shape[0] > 1 else mask[0]))
            for mask in masks
        ]
        return WriterOutputs(results)

    @register_process("core/postprocess/segmentation/class_mask")
    def _segmentation_class_mask(
        self,
        outputs: SegmentationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.masks)
        assert all([m.ndim == 3 for m in outputs.masks])
        if self.mask_threshold is not None:
            masks = [
                (m.numpy() < self.mask_threshold).astype(np.int16)
                for m in outputs.masks
            ]
        else:
            masks = [m.numpy() for m in outputs.masks]
        classes = [c.numpy() for c in outputs.classes]
        classes = [c if c.ndim == 1 else c.argmax(-1) for c in classes]
        results["mask_images"] = [
            ";".join(
                [
                    self.save_image(numpy_to_pil(_mask_image))
                    for _mask_image in _mask_images
                ]
            )
            for _mask_images in masks
        ]
        results["mask_classes"] = [
            ";".join([str(_class) for _class in _classes]) for _classes in classes
        ]
        return WriterOutputs(results)


def segmentation_model_decorator(cls):
    class SegmentationModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__segmentation_model__" in kwargs:
                self.model = kwargs.pop("__segmentation_model__")
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

            assert hasattr(self.model, "segment")

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.segment(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__segmentation_model__=model)

    return SegmentationModel
