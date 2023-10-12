# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.stable_xl import (
    StableXLForText2ImageGenerationPipeline,
    StableXLForImage2ImageGenerationPipeline,
    StableXLForImageInpaintingPipeline,
)


@register_webui("core/webui/stable_xl/text2image")
class StableXLText2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl/text2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7871)

    def start(self, **kwargs):
        self._model = StableXLForText2ImageGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable_xl/text2image")

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


@register_webui("core/webui/stable_xl/image2image")
class StableXLImage2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl/image2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7872)

    def start(self, **kwargs):
        self._model = StableXLForImage2ImageGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable_xl/image2image")

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


@register_webui("core/webui/stable_xl/inpainting")
class StableXLImageInpaintingWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable_xl/inpainting")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7873)

    def start(self, **kwargs):
        self._model = StableXLForImageInpaintingPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable_xl/inpainting")

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
