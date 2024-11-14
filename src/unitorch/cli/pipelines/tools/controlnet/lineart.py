# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from unitorch.models import GenericModel
from unitorch.cli import hf_endpoint_url


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LineartModel(GenericModel):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super().__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartPipeline(LineartModel):
    def __init__(self):
        super().__init__(3, 1, 3)
        self.from_pretrained(
            hf_endpoint_url("/lllyasviel/Annotators/resolve/main/sk_model.pth")
        )
        self.preprocess = ToTensor()
        self.postprocess = ToPILImage()
        self.eval()

    def __call__(self, image: Image.Image):
        pixel_values = self.preprocess(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        line = self.forward(pixel_values)[0][0]
        line = self.postprocess(line.cpu())
        return line


lineart_pipe = None


def lineart(image: Image.Image):
    global lineart_pipe
    if lineart_pipe is None:
        lineart_pipe = LineartPipeline()
        lineart_pipe.to("cpu")
    return lineart_pipe(image)
