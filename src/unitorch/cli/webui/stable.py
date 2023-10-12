# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.stable import (
    StableForText2ImageGenerationPipeline,
    StableForImage2ImageGenerationPipeline,
    StableForImageInpaintingPipeline,
    StableForImageResolutionPipeline,
)


@register_webui("core/webui/stable/text2image")
class StableText2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable/text2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7861)

    def start(self, **kwargs):
        self._model = StableForText2ImageGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable/text2image")

        iface = gr.Interface(fn=self.serve, inputs="textbox", outputs="image")
        iface.launch(server_name=self.host, server_port=self.port)

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        height: int = 512,
        width: int = 512,
    ):
        assert self._model is not None
        image = self._model(text, height, width)
        return image


@register_webui("core/webui/stable/image2image")
class StableImage2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable/image2image")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7862)

    def start(self, **kwargs):
        self._model = StableForImage2ImageGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/stable/image2image")

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


@register_webui("core/webui/stable/inpainting")
class StableImageInpaintingWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable/inpainting")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7863)

    def start(self, **kwargs):
        self._model = StableForImageInpaintingPipeline.from_core_configure(self.config)
        self.config.set_default_section("core/webui/stable/inpainting")

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


@register_webui("core/webui/stable/resolution")
class StableImageResolutionWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/stable/resolution")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7864)

    def start(self, **kwargs):
        self._model = StableForImageResolutionPipeline.from_core_configure(self.config)
        self.config.set_default_section("core/webui/stable/resolution")

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
