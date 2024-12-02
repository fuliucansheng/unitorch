# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers import AutoencoderKL
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


class VAEForDiffusion(GenericModel):
    prefix_keys_in_state_dict = {
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    replace_keys_in_state_dict = {
        "\.query\.": ".to_q.",
        "\.key\.": ".to_k.",
        "\.value\.": ".to_v.",
        "\.proj_attn\.": ".to_out.0.",
    }

    def __init__(
        self,
        config_path: str,
        patch_size: Optional[int] = 32,
        stride: Optional[int] = 16,
    ):
        super().__init__()
        config_dict = json.load(open(config_path))
        self.vae = AutoencoderKL.from_config(config_dict)
        self.patch_size = patch_size
        self.stride = stride

    def patch_mse_loss(self, y_true, y_pred):
        y_true = y_true.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_pred = y_pred.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_true = y_true.contiguous().view(y_true.size(0), -1)
        y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
        return F.mse_loss(y_pred, y_true, reduction="none").mean(1)

    def forward(self, pixel_values):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        outputs = self.vae.decode(latents).sample

        loss = self.patch_mse_loss(pixel_values, outputs).mean()
        return loss
