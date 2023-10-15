# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.animate import (
    AnimateForText2VideoGenerationPipeline,
)


@register_webui("core/webui/animate/text2video")
class AnimateText2VideoWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        config.set_default_section("core/webui/animate/text2video")
        self.host = config.getoption("host", "0.0.0.0")
        self.port = config.getoption("port", 7850)

    def start(self, **kwargs):
        self._model = AnimateForText2VideoGenerationPipeline.from_core_configure(
            self.config
        )
        self.config.set_default_section("core/webui/animate/text2video")

        iface = gr.Interface(
            fn=self.serve, inputs=["textbox", "textbox"], outputs=gr.Video()
        )
        iface.launch(server_name=self.host, server_port=self.port)

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        negative_text: str,
        height: int = 512,
        width: int = 512,
    ):
        assert self._model is not None
        video = self._model(text, negative_text, height, width)
        return video
