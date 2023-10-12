# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.stable_xl_refiner import (
    StableXLRefinerForText2ImageGenerationPipeline,
    StableXLRefinerForImage2ImageGenerationPipeline,
    StableXLRefinerForImageInpaintingPipeline,
)


@register_webui("core/webui/stable_xl_refiner/text2image")
class StableXLRefinerText2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl_refiner/text2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7881)

    def start(self, **kwargs):
        self._model = (
            StableXLRefinerForText2ImageGenerationPipeline.from_core_configure(
                self.config
            )
        )
        self.config.set_default_section("core/webui/stable_xl_refiner/text2image")

        iface = gr.Interface(fn=self.serve, inputs="textbox", outputs="image")
        iface.launch(server_name=self.host, server_port=self.port)

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        height: int = 1024,
        width: int = 1024,
    ):
        assert self._model is not None
        image = self._model(text, height, width)
        return image


@register_webui("core/webui/stable_xl_refiner/image2image")
class StableXLRefinerImage2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl_refiner/image2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7882)

    def start(self, **kwargs):
        self._model = (
            StableXLRefinerForImage2ImageGenerationPipeline.from_core_configure(
                self.config
            )
        )
        self.config.set_default_section("core/webui/stable_xl_refiner/image2image")

        iface = gr.Interface(
            fn=self.serve, inputs=["textbox", gr.Image(type="pil")], outputs="image"
        )
        iface.launch(server_name=self.host, server_port=self.port)

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        image: Image.Image,
    ):
        assert self._model is not None
        image = self._model(text, image)
        return image


@register_webui("core/webui/stable_xl_refiner/inpainting")
class StableXLRefinerImageInpaintingWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl_refiner/inpainting")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7883)

    def start(self, **kwargs):
        self._model = StableXLRefinerForImageInpaintingPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable_xl_refiner/inpainting")

        iface = gr.Interface(
            fn=self.serve,
            inputs=["textbox", gr.Image(type="pil"), gr.Image(type="pil")],
            outputs="image",
        )
        iface.launch(server_name=self.host, server_port=self.port)

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
    ):
        assert self._model is not None
        image = self._model(text, image, mask_image)
        return image
