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
from unitorch.cli.models.diffusers import pretrained_diffusers_infos
from unitorch.cli.pipelines.stable_xl import (
    StableXLForText2ImageGenerationPipeline,
    StableXLForImage2ImageGenerationPipeline,
    StableXLForImageInpaintingPipeline,
)
from unitorch.cli.webui import matched_pretrained_names


@register_webui("core/webui/stable_xl/text2image")
class StableXLText2ImageWebUI(GenericWebUI):
    match_patterns = ["^stable-xl", "^stable-turbo-xl"]
    block_patterns = [
        ".*controlnet",
        ".*refiner",
        ".*adapter",
    ]
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++", "DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
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

            with gr.Row(variant="panel"):
                with gr.Column():
                    prompt = gr.Textbox(label="Input Prompt")
                    height = gr.Slider(512, 1024, value=1024, label="Image Height")
                    width = gr.Slider(512, 1024, value=1024, label="Image Width")
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    steps = gr.Slider(
                        0, 1000, value=50, label="Diffusion Steps", step=1
                    )
                    seed = gr.Slider(
                        0, 999999999999, value=42, label="Magic Number", step=1
                    )
                    scheduler = gr.Radio(self.supported_schedulers, label="Sampler")
                    with gr.Row(variant="panel"):
                        freeu_s1 = gr.Slider(
                            0, 10, value=0.9, label="FreeU S1", step=0.1
                        )
                        freeu_s2 = gr.Slider(
                            0, 10, value=0.2, label="FreeU S2", step=0.1
                        )
                        freeu_b1 = gr.Slider(
                            0, 10, value=1.2, label="FreeU B1", step=0.1
                        )
                        freeu_b2 = gr.Slider(
                            0, 10, value=1.4, label="FreeU B2", step=0.1
                        )

                    submit = gr.Button(value="Submit")

                image = gr.Image(type="pil", label="Output Image")
                submit.click(
                    self.serve,
                    inputs=[
                        prompt,
                        height,
                        width,
                        guidance_scale,
                        steps,
                        seed,
                        scheduler,
                        freeu_s1,
                        freeu_s2,
                        freeu_b1,
                        freeu_b2,
                    ],
                    outputs=[image],
                )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "StableXLText2Image"

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
            "core/pipeline/stable_xl/text2image", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = StableXLForText2ImageGenerationPipeline.from_core_configure(
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
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image


@register_webui("core/webui/stable_xl/image2image")
class StableXLImage2ImageWebUI(GenericWebUI):
    match_patterns = [
        "^stable-xl",
    ]
    block_patterns = [
        ".*controlnet",
        ".*refiner",
        ".*adapter",
    ]
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++", "DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
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

            with gr.Row(variant="panel"):
                with gr.Column():
                    raw_image = gr.Image(type="pil", label="Input Image")
                    prompt = gr.Textbox(label="Input Prompt")
                    strength = gr.Slider(0, 1, value=0.8, label="Strength", step=0.01)
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    steps = gr.Slider(
                        0, 1000, value=50, label="Diffusion Steps", step=1
                    )
                    seed = gr.Slider(
                        0, 999999999999, value=42, label="Magic Number", step=1
                    )
                    scheduler = gr.Radio(self.supported_schedulers, label="Sampler")
                    with gr.Row(variant="panel"):
                        freeu_s1 = gr.Slider(
                            0, 10, value=0.9, label="FreeU S1", step=0.1
                        )
                        freeu_s2 = gr.Slider(
                            0, 10, value=0.2, label="FreeU S2", step=0.1
                        )
                        freeu_b1 = gr.Slider(
                            0, 10, value=1.2, label="FreeU B1", step=0.1
                        )
                        freeu_b2 = gr.Slider(
                            0, 10, value=1.4, label="FreeU B2", step=0.1
                        )

                    submit = gr.Button(value="Submit")
                image = gr.Image(type="pil", label="Output Image")

                submit.click(
                    self.serve,
                    inputs=[
                        prompt,
                        raw_image,
                        strength,
                        guidance_scale,
                        steps,
                        seed,
                        scheduler,
                        freeu_s1,
                        freeu_s2,
                        freeu_b1,
                        freeu_b2,
                    ],
                    outputs=[image],
                )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "StableXLImage2Image"

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
            "core/pipeline/stable_xl/image2image", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = StableXLForImage2ImageGenerationPipeline.from_core_configure(
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
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image


@register_webui("core/webui/stable_xl/inpainting")
class StableXLImageInpaintingWebUI(GenericWebUI):
    match_patterns = [
        "^stable-xl",
    ]
    block_patterns = [
        ".*controlnet",
        ".*refiner",
        ".*adapter",
    ]
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++", "DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
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

            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        raw_image = gr.Image(type="pil", label="Input Image")
                        raw_image_mask = gr.Image(type="pil", label="Input Mask Image")
                    prompt = gr.Textbox(label="Input Prompt")
                    strength = gr.Slider(0, 1, value=0.8, label="Strength", step=0.01)
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    steps = gr.Slider(
                        0, 1000, value=50, label="Diffusion Steps", step=1
                    )
                    seed = gr.Slider(
                        0, 999999999999, value=42, label="Magic Number", step=1
                    )
                    scheduler = gr.Radio(self.supported_schedulers, label="Sampler")
                    with gr.Row(variant="panel"):
                        freeu_s1 = gr.Slider(
                            0, 10, value=0.9, label="FreeU S1", step=0.1
                        )
                        freeu_s2 = gr.Slider(
                            0, 10, value=0.2, label="FreeU S2", step=0.1
                        )
                        freeu_b1 = gr.Slider(
                            0, 10, value=1.2, label="FreeU B1", step=0.1
                        )
                        freeu_b2 = gr.Slider(
                            0, 10, value=1.4, label="FreeU B2", step=0.1
                        )

                    submit = gr.Button(value="Submit")
                image = gr.Image(type="pil", label="Output Image")
                submit.click(
                    self.serve,
                    inputs=[
                        prompt,
                        raw_image,
                        raw_image_mask,
                        strength,
                        guidance_scale,
                        steps,
                        seed,
                        scheduler,
                        freeu_s1,
                        freeu_s2,
                        freeu_b1,
                        freeu_b2,
                    ],
                    outputs=[image],
                )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "StableXLImageInpainting"

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
            "core/pipeline/stable_xl/inpainting", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = StableXLForImageInpaintingPipeline.from_core_configure(self.config)
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
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            image,
            mask_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image
