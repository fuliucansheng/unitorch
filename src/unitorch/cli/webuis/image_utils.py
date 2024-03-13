# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image, ImageFilter, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui


@register_webui("core/webui/image_utils")
class ImageUtilsWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._status = "running"
        self._iface = gr.Blocks()
        with self._iface:
            # canny
            with gr.Row(variant="panel"):
                with gr.Column():
                    image = gr.Image(type="pil", label="Input Image")
                    height = gr.Slider(512, 1024, value=512, label="Image Height")
                    width = gr.Slider(512, 1024, value=512, label="Image Width")
                    submit = gr.Button(value="Submit")
                canny_image = gr.Image(type="pil", label="Output Canny Image")

                submit.click(
                    self.canny,
                    inputs=[image, height, width],
                    outputs=[canny_image],
                )

            # blend
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        image1 = gr.Image(type="pil", label="Input Image1")
                        image2 = gr.Image(type="pil", label="Input Image2")
                    height2 = gr.Slider(512, 1024, value=512, label="Image Height")
                    width2 = gr.Slider(512, 1024, value=512, label="Image Width")
                    alpha = gr.Slider(0, 1, value=0.5, label="Alpha")
                    submit2 = gr.Button(value="Submit")
                blend_image = gr.Image(type="pil", label="Output Blend Image")

                submit2.click(
                    self.blend,
                    inputs=[image1, image2, alpha, height2, width2],
                    outputs=[blend_image],
                )

            # invert
            with gr.Row(variant="panel"):
                with gr.Column():
                    image3 = gr.Image(type="pil", label="Input Image")
                    height3 = gr.Slider(512, 1024, value=512, label="Image Height")
                    width3 = gr.Slider(512, 1024, value=512, label="Image Width")
                    submit3 = gr.Button(value="Submit")
                invert_image = gr.Image(type="pil", label="Output Invert Image")

                submit3.click(
                    self.invert,
                    inputs=[image3, height3, width3],
                    outputs=[invert_image],
                )

    @property
    def name(self):
        return "ImageUtils"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        passs

    def canny(
        self,
        image: Image.Image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ):
        image = image.resize((height, width))
        image = image.convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)
        return image

    def blend(
        self,
        image1,
        image2,
        alpha=0.5,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ):
        image1 = image1.resize((height, width))
        image2 = image2.resize((height, width))
        image = Image.blend(image1, image2, alpha)
        return image

    def invert(
        self,
        image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ):
        image = image.resize((height, width))
        image = ImageOps.invert(image)
        return image
