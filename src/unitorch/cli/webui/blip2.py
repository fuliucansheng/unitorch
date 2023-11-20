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
from unitorch.cli.pipelines.blip2 import Blip2ForText2ImageGenerationPipeline
from unitorch.cli.pipelines.blip2controlnet import (
    Blip2ControlNetForText2ImageGenerationPipeline,
)


@register_webui("core/webui/blip2/text2image")
class Blip2Text2ImageWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-blipdiffusion",
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
            refer_prompt = gr.Textbox(label="Input Reference Prompt")
            refer_image = gr.Image(type="pil", label="Input Reference Image")
            image = gr.Image(type="pil", label="Output Image")
            height = gr.Slider(512, 768, value=512, label="Image Height")
            width = gr.Slider(512, 768, value=512, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve,
                inputs=[prompt, refer_prompt, refer_image, height, width],
                outputs=[image],
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "Blip2Text2Image"

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
            "core/pipeline/blip2/text2image", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = Blip2ForText2ImageGenerationPipeline.from_core_configure(
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
        refer_text: str,
        refer_image: Image.Image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            refer_text,
            refer_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        return image


@register_webui("core/webui/blip2controlnet/text2image")
class Blip2ControlNetText2ImageWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-blipdiffusion-controlnet",
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
            cond_image = gr.Image(type="pil", label="Input Condition Image")
            refer_prompt = gr.Textbox(label="Input Reference Prompt")
            refer_image = gr.Image(type="pil", label="Input Reference Image")
            image = gr.Image(type="pil", label="Output Image")
            height = gr.Slider(512, 768, value=512, label="Image Height")
            width = gr.Slider(512, 768, value=512, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve,
                inputs=[prompt, cond_image, refer_prompt, refer_image, height, width],
                outputs=[image],
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "Blip2ControlNetText2Image"

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
            "core/pipeline/blip2/text2image", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = Blip2ControlNetForText2ImageGenerationPipeline.from_core_configure(
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
        cond_image: Image.Image,
        refer_text: str,
        refer_image: Image.Image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            cond_image,
            refer_text,
            refer_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        return image
