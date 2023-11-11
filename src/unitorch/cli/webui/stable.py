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
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-v1.5"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-v1.5", "stable-v2", "stable-v2.1"],
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
            image = gr.Image(type="pil", label="Output Image")
            height = gr.Slider(512, 768, value=512, label="Image Height")
            width = gr.Slider(512, 768, value=512, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(self.serve, inputs=[prompt, height, width], outputs=[image])

    @property
    def name(self):
        return "StableText2Image"

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
            "core/pipeline/stable/text2image", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/stable/text2image", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = StableForText2ImageGenerationPipeline.from_core_configure(
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
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        return image


@register_webui("core/webui/stable/image2image")
class StableImage2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-v1.5-nitrosocke-ghibli"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-v1.5-nitrosocke-ghibli"],
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
            image = gr.Image(type="pil", label="Output Image")
            submit = gr.Button(value="Submit")
            submit.click(self.serve, inputs=[prompt, raw_image], outputs=[image])

    @property
    def name(self):
        return "StableImage2Image"

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
            "core/pipeline/stable/image2image", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/stable/image2image", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = StableForImage2ImageGenerationPipeline.from_core_configure(
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
    ):
        assert self._pipe is not None
        image = self._pipe(text, image)
        return image


@register_webui("core/webui/stable/inpainting")
class StableImageInpaintingWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-v1.5-inpainting"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-v1.5-inpainting"],
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
            image = gr.Image(type="pil", label="Output Image")
            submit = gr.Button(value="Submit")
            submit.click(
                self.serve, inputs=[prompt, raw_image, raw_image_mask], outputs=[image]
            )

    @property
    def name(self):
        return "StableImageInpainting"

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
            "core/pipeline/stable/inpainting", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/stable/inpainting", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = StableForImageInpaintingPipeline.from_core_configure(self.config)
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
    ):
        assert self._pipe is not None
        image = self._pipe(text, image, mask_image)
        return image


@register_webui("core/webui/stable/resolution")
class StableImageResolutionWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-v1.5-x4-upscaler"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-v1.5-x4-upscaler"],
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
            image = gr.Image(type="pil", label="Output Image")
            submit = gr.Button(value="Submit")
            submit.click(self.serve, inputs=[prompt, raw_image], outputs=[image])

    @property
    def name(self):
        return "StableImageResolution"

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
            "core/pipeline/stable/resolution", "pretrained_name", pretrained_name
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set("core/pipeline/stable/resolution", "pad_token", "!")
        self._name = pretrained_name
        self._pipe = StableForImageResolutionPipeline.from_core_configure(self.config)
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
    ):
        assert self._pipe is not None
        image = self._pipe(text, image)
        return image
