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
from unitorch.cli.pipelines.stable_xl_refiner import (
    StableXLRefinerForText2ImageGenerationPipeline,
    StableXLRefinerForImage2ImageGenerationPipeline,
    StableXLRefinerForImageInpaintingPipeline,
)


@register_webui("core/webui/stable_xl_refiner/text2image")
class StableXLRefinerText2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-xl-base-refiner-1.0"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-xl-base-refiner-1.0"],
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
            height = gr.Slider(512, 1024, value=1024, label="Image Height")
            width = gr.Slider(512, 1024, value=1024, label="Image Width")
            submit = gr.Button(value="Submit")
            submit.click(self.serve, inputs=[prompt, height, width], outputs=[image])

    @property
    def name(self):
        return "StableXLRefinerText2Image"

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
            "core/pipeline/stable_xl_refiner/text2image",
            "pretrained_name",
            pretrained_name,
        )
        self._name = pretrained_name
        self._pipe = StableXLRefinerForText2ImageGenerationPipeline.from_core_configure(
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
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
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


@register_webui("core/webui/stable_xl_refiner/image2image")
class StableXLRefinerImage2ImageWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-xl-base-refiner-1.0"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-xl-base-refiner-1.0"],
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
        return "StableXLRefinerImage2Image"

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
            "core/pipeline/stable_xl_refiner/image2image",
            "pretrained_name",
            pretrained_name,
        )
        self._name = pretrained_name
        self._pipe = (
            StableXLRefinerForImage2ImageGenerationPipeline.from_core_configure(
                self.config
            )
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


@register_webui("core/webui/stable_xl_refiner/inpainting")
class StableXLRefinerImageInpaintingWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._name = "stable-xl-base-refiner-1.0"
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row():
                pretrained_name = gr.Dropdown(
                    ["stable-xl-base-1.0"],
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
        return "StableXLRefinerImageInpainting"

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
            "core/pipeline/stable_xl_refiner/inpainting",
            "pretrained_name",
            pretrained_name,
        )
        self._name = pretrained_name
        self._pipe = StableXLRefinerForImageInpaintingPipeline.from_core_configure(
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
    ):
        assert self._pipe is not None
        image = self._pipe(text, image, mask_image)
        return image
