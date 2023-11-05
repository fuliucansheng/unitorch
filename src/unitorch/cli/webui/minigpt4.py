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
        self._pipe = None
        self._status = "stopped"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["minigpt4-7b", "minigpt4-13b"], label="Pretrain Checkpoint Name"
                )
                status = gr.Textbox(label="Model Status")
                click_start = gr.Button(value="Start")
                click_stop = gr.Button(value="Stop")
                click_start.click(
                    self.start, inputs=[pretrained_name], outputs=[status]
                )
                click_stop.click(self.stop, outputs=[status])
            prompt = gr.Textbox(label="Input Prompt")
            image = gr.Image(type="pil", label="Input Image")
            caption = gr.Textbox(label="Output Caption")
            submit = gr.Button(text="Submit")
            submit.click(self.serve, inputs=[prompt, image], outputs=[caption])

    @property
    def name(self):
        return "MiniGPT4"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self, pretrained_name, **kwargs):
        if self._status == "running":
            self.stop()
        self.config.set("core/pipeline/minigpt4", "pretrained_name", pretrained_name)
        self._pipe = MiniGPT4Blip2LlamaForGenerationPipeline.from_core_configure(
            self.config
        )
        self._status = "running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        self._status = "stopped"
        return self._status

    def serve(
        self,
        text: str,
        image: Image.Image,
    ):
        assert self._pipe is not None
        result = self._pipe(text, image)
        return result
