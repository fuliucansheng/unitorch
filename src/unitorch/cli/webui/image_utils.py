# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image, ImageFilter
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
            with gr.Row():
                image = gr.Image(type="pil", label="Input Image")
                canny_image = gr.Image(type="pil", label="Output Canny Image")
            height = gr.Slider(512, 768, value=512, label="Image Height")
            width = gr.Slider(512, 768, value=512, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(
                self.canny,
                inputs=[image, height, width],
                outputs=[canny_image],
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
