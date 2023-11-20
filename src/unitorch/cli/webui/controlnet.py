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
from unitorch.cli.pipelines.controlnet import (
    ControlNetForText2ImageGenerationPipeline,
    ControlNetForImage2ImageGenerationPipeline,
    ControlNetForImageInpaintingPipeline,
)


@register_webui("core/webui/controlnet/text2image")
class ControlNetText2ImageWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-controlnet-canny",
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
            condition_image = gr.Image(type="pil", label="Input Condition Image")
            image = gr.Image(type="pil", label="Output Image")
            height = gr.Slider(512, 768, value=512, label="Image Height")
            width = gr.Slider(512, 768, value=512, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve,
                inputs=[prompt, condition_image, height, width],
                outputs=[image],
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "ControlNetText2Image"

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
            "core/pipeline/controlnet/text2image", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/controlnet/text2image", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = ControlNetForText2ImageGenerationPipeline.from_core_configure(
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
        condition_image: Image.Image,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            condition_image=condition_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        return image


@register_webui("core/webui/controlnet/image2image")
class ControlNetImage2ImageWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-controlnet-canny",
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
            raw_image = gr.Image(type="pil", label="Input Image")
            condition_image = gr.Image(type="pil", label="Input Condition Image")
            image = gr.Image(type="pil", label="Output Image")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve, inputs=[prompt, raw_image, condition_image], outputs=[image]
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "ControlNetImage2Image"

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
            "core/pipeline/controlnet/image2image", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/controlnet/image2image", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = ControlNetForImage2ImageGenerationPipeline.from_core_configure(
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
        image: Image.Image,
        condition_image: Image.Image,
    ):
        assert self._pipe is not None
        image = self._pipe(text, image, condition_image)
        return image


@register_webui("core/webui/controlnet/inpainting")
class ControlNetImageInpaintingWebUI(GenericWebUI):
    supported_pretrained_names = [
        "stable-v1.5-controlnet-canny",
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
            raw_image = gr.Image(type="pil", label="Input Image")
            raw_image_mask = gr.Image(type="pil", label="Input Mask Image")
            condition_image = gr.Image(type="pil", label="Input Condition Image")
            image = gr.Image(type="pil", label="Output Image")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve,
                inputs=[prompt, raw_image, raw_image_mask, condition_image],
                outputs=[image],
            )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "ControlNetImageInpainting"

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
            "core/pipeline/controlnet/inpainting", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/controlnet/inpainting", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = ControlNetForImageInpaintingPipeline.from_core_configure(
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
        image: Image.Image,
        mask_image: Image.Image,
        condition_image: Image.Image,
    ):
        assert self._pipe is not None
        image = self._pipe(text, image, mask_image, condition_image)
        return image
