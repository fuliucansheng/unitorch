# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.animate import (
    AnimateForText2VideoGenerationPipeline,
)


@register_webui("core/webui/animate/text2video")
class AnimateText2VideoWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-realistic-animate-v1.5",
        "stable-v1.5-realistic-animate-v1.5.2",
        "stable-v1.5-realistic-animate-v1.5.2-zoom-in",
        "stable-v1.5-realistic-animate-v1.5.2-zoom-out",
    ]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    self.supported_pretrained_names,
                    value=self._name,
                    label="Pretrain Checkpoint Name",
                )
                status = gr.Textbox(label="Model Status", value=self._status)
                click_start = gr.Button(value="Start")
                click_stop = gr.Button(value="Stop")
                click_start.click(
                    self.start, inputs=[pretrained_name], outputs=[status]
                )
                click_stop.click(self.stop, outputs=[status])
            prompt = gr.Textbox(label="Input Prompt")
            negative_prompt = gr.Textbox(label="Input Negative Prompt")
            video = gr.Video(type="pil", label="Output Video")
            height = gr.Slider(512, 768, value=512, label="Video Height")
            width = gr.Slider(512, 768, value=512, label="Video Width")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve,
                inputs=[prompt, negative_prompt, height, width],
                outputs=[video],
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "StableText2Video"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self, pretrained_name, **kwargs):
        if self._status == "running":
            self.stop()
        self.config.set(
            "core/pipeline/animate/text2video", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = AnimateForText2VideoGenerationPipeline.from_core_configure(
            self.config
        )
        self._status = "running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        return self._status

    def serve(
        self,
        text: str,
        negative_text: str,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
    ):
        assert self._pipe is not None
        video = self._pipe(
            text,
            negative_text,
            height=height,
            width=width,
        )
        return video
