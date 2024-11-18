# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from unitorch.models import GenericModel
from unitorch.cli import hf_endpoint_url


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(GenericModel):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(
            input_channel=3, output_channel=64, layer_number=2
        )
        self.block2 = DoubleConvBlock(
            input_channel=64, output_channel=128, layer_number=2
        )
        self.block3 = DoubleConvBlock(
            input_channel=128, output_channel=256, layer_number=3
        )
        self.block4 = DoubleConvBlock(
            input_channel=256, output_channel=512, layer_number=3
        )
        self.block5 = DoubleConvBlock(
            input_channel=512, output_channel=512, layer_number=3
        )

    def forward(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HedPipeline(ControlNetHED_Apache2):
    def __init__(self):
        super().__init__()
        self.from_pretrained(
            hf_endpoint_url("/lllyasviel/Annotators/resolve/main/ControlNetHED.pth")
        )
        self.preprocess = ToTensor()
        self.postprocess = ToPILImage()
        self.eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W, C = image.shape
        pixel_values = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        edges = self.forward(pixel_values)
        edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))


hed_pipe = None


def hed(image: Image.Image):
    global hed_pipe
    if hed_pipe is None:
        hed_pipe = HedPipeline()
        hed_pipe.to("cpu")
    return hed_pipe(image)
