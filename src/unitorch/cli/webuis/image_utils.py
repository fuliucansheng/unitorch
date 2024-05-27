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


class ImageCannyWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._status = "running"
        self._iface = gr.Blocks()
        with self._iface:
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

    @property
    def name(self):
        return "Canny"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        pass

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


class ImageBlendWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._status = "running"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        image1 = gr.Image(type="pil", label="Input Image1")
                        image2 = gr.Image(type="pil", label="Input Image2")
                    height = gr.Slider(512, 1024, value=512, label="Image Height")
                    width = gr.Slider(512, 1024, value=512, label="Image Width")
                    alpha = gr.Slider(0, 1, value=0.5, label="Alpha")
                    submit = gr.Button(value="Submit")
                blend_image = gr.Image(type="pil", label="Output Blend Image")

                submit.click(
                    self.blend,
                    inputs=[image1, image2, alpha, height, width],
                    outputs=[blend_image],
                )

    @property
    def name(self):
        return "Blend"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        pass

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


class ImageInvertWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._status = "running"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
                with gr.Column():
                    image = gr.Image(type="pil", label="Input Image")
                    height = gr.Slider(512, 1024, value=512, label="Image Height")
                    width = gr.Slider(512, 1024, value=512, label="Image Width")
                    submit = gr.Button(value="Submit")
                invert_image = gr.Image(type="pil", label="Output Invert Image")

                submit.click(
                    self.invert,
                    inputs=[image, height, width],
                    outputs=[invert_image],
                )

    @property
    def name(self):
        return "Invert"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        pass

    def invert(
        self,
        image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ):
        image = image.resize((height, width))
        image = ImageOps.invert(image)
        return image


@register_webui("core/webui/image")
class ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ImageCannyWebUI(config),
            ImageBlendWebUI(config),
            ImageInvertWebUI(config),
        ]
        self._iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.name for webui in webuis],
        )

    @property
    def name(self):
        return "Image"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        pass
