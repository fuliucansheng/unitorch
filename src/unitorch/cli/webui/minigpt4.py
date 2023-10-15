# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.minigpt4 import MiniGPT4Blip2LlamaForGenerationPipeline


@register_webui("core/webui/minigpt4")
class MiniGPT4WebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/minigpt4")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7851)

    def start(self, **kwargs):
        self._model = MiniGPT4Blip2LlamaForGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/minigpt4")

        iface = gr.Interface(
            fn=self.serve, inputs=["textbox", gr.Image(type="pil")], outputs="textbox"
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
        result = self._model(text, image)
        return result
