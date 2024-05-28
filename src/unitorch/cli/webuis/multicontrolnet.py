# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import re
from click import Option
import torch
import gc
import gradio as gr
from PIL import Image, ImageFilter, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.models.diffusers import pretrained_diffusers_infos
from unitorch.cli.pipelines.multicontrolnet import (
    MultiControlNetForText2ImageGenerationPipeline,
    MultiControlNetForImage2ImageGenerationPipeline,
    MultiControlNetForImageInpaintingPipeline,
)
from unitorch.cli.webuis import matched_pretrained_names
from unitorch.cli.webuis import CannyWebUI, DPTWebUI


class MultiControlNetText2ImageWebUI(GenericWebUI):
    match_patterns = [
        "^stable-v1.5.*multicontrolnet",
        "^stable-v2.*multicontrolnet",
        "^stable-v2.1.*multicontrolnet",
    ]
    block_patterns = [
        ".*inpainting",
    ]
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._num_controlnets = 0
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
                    negative_prompt = gr.Textbox(label="Input Negative Prompt")
                    height = gr.Slider(512, 1024, value=512, label="Image Height")
                    width = gr.Slider(512, 1024, value=512, label="Image Width")
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    with gr.Row(variant="panel"):
                        with gr.Tab("ControlNet 1"):
                            condition_image1 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale1 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 2"):
                            condition_image2 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale2 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 3"):
                            condition_image3 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale3 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 4"):
                            condition_image4 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale4 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 5"):
                            condition_image5 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale5 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
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
                        condition_image1,
                        condition_image2,
                        condition_image3,
                        condition_image4,
                        condition_image5,
                        negative_prompt,
                        height,
                        width,
                        guidance_scale,
                        conditioning_scale1,
                        conditioning_scale2,
                        conditioning_scale3,
                        conditioning_scale4,
                        conditioning_scale5,
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
        return "Text2Image"

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
            "core/pipeline/multicontrolnet/text2image",
            "pretrained_name",
            pretrained_name,
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set(
                "core/pipeline/multicontrolnet/text2image", "pad_token", "!"
            )
        self._name = pretrained_name
        self._pipe = MultiControlNetForText2ImageGenerationPipeline.from_core_configure(
            self.config
        )
        self._status = "running"
        self._num_controlnets = len(
            nested_dict_value(
                pretrained_diffusers_infos, pretrained_name, "controlnet", "config"
            )
        )
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._num_controlnets = 0
        return self._status

    def serve(
        self,
        text: str,
        condition_image1: Optional[Image.Image] = None,
        condition_image2: Optional[Image.Image] = None,
        condition_image3: Optional[Image.Image] = None,
        condition_image4: Optional[Image.Image] = None,
        condition_image5: Optional[Image.Image] = None,
        neg_text: Optional[str] = "",
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        conditioning_scale1: Optional[float] = 1.0,
        conditioning_scale2: Optional[float] = 1.0,
        conditioning_scale3: Optional[float] = 1.0,
        conditioning_scale4: Optional[float] = 1.0,
        conditioning_scale5: Optional[float] = 1.0,
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
            condition_images=[
                condition_image1,
                condition_image2,
                condition_image3,
                condition_image4,
                condition_image5,
            ][: self._num_controlnets],
            neg_text=neg_text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[
                conditioning_scale1,
                conditioning_scale2,
                conditioning_scale3,
                conditioning_scale4,
                conditioning_scale5,
            ][: self._num_controlnets],
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image


class MultiControlNetImage2ImageWebUI(GenericWebUI):
    match_patterns = [
        "^stable-v1.5.*multicontrolnet",
        "^stable-v2.*multicontrolnet",
        "^stable-v2.1.*multicontrolnet",
    ]
    block_patterns = []
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._num_controlnets = 0
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
                    negative_prompt = gr.Textbox(label="Input Negative Prompt")
                    strength = gr.Slider(0, 1, value=0.8, label="Strength", step=0.01)
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    with gr.Row(variant="panel"):
                        with gr.Tab("ControlNet 1"):
                            condition_image1 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale1 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 2"):
                            condition_image2 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale2 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 3"):
                            condition_image3 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale3 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 4"):
                            condition_image4 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale4 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 5"):
                            condition_image5 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale5 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
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
                        condition_image1,
                        condition_image2,
                        condition_image3,
                        condition_image4,
                        condition_image5,
                        negative_prompt,
                        strength,
                        guidance_scale,
                        conditioning_scale1,
                        conditioning_scale2,
                        conditioning_scale3,
                        conditioning_scale4,
                        conditioning_scale5,
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
        return "Image2Image"

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
            "core/pipeline/multicontrolnet/image2image",
            "pretrained_name",
            pretrained_name,
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set(
                "core/pipeline/multicontrolnet/image2image", "pad_token", "!"
            )
        self._name = pretrained_name
        self._pipe = (
            MultiControlNetForImage2ImageGenerationPipeline.from_core_configure(
                self.config
            )
        )
        self._status = "running"
        self._num_controlnets = len(
            nested_dict_value(
                pretrained_diffusers_infos, pretrained_name, "controlnet", "config"
            )
        )
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._num_controlnets = 0
        return self._status

    def serve(
        self,
        text: str,
        image: Image.Image,
        condition_image1: Optional[Image.Image] = None,
        condition_image2: Optional[Image.Image] = None,
        condition_image3: Optional[Image.Image] = None,
        condition_image4: Optional[Image.Image] = None,
        condition_image5: Optional[Image.Image] = None,
        neg_text: Optional[str] = "",
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        conditioning_scale1: Optional[float] = 1.0,
        conditioning_scale2: Optional[float] = 1.0,
        conditioning_scale3: Optional[float] = 1.0,
        conditioning_scale4: Optional[float] = 1.0,
        conditioning_scale5: Optional[float] = 1.0,
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
            condition_images=[
                condition_image1,
                condition_image2,
                condition_image3,
                condition_image4,
                condition_image5,
            ][: self._num_controlnets],
            neg_text=neg_text,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[
                conditioning_scale1,
                conditioning_scale2,
                conditioning_scale3,
                conditioning_scale4,
                conditioning_scale5,
            ][: self._num_controlnets],
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image


class MultiControlNetImageInpaintingWebUI(GenericWebUI):
    match_patterns = [
        "^stable-v1.5.*multicontrolnet",
        "^stable-v2.*multicontrolnet",
        "^stable-v2.1.*multicontrolnet",
    ]
    block_patterns = []
    pretrained_names = list(pretrained_diffusers_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, match_patterns, block_patterns
    )
    supported_schedulers = ["DPM++SDE", "UniPC++"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._num_controlnets = 0
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
                    negative_prompt = gr.Textbox(label="Input Negative Prompt")
                    strength = gr.Slider(0, 1, value=0.8, label="Strength", step=0.01)
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    with gr.Row(variant="panel"):
                        with gr.Tab("ControlNet 1"):
                            condition_image1 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale1 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 2"):
                            condition_image2 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale2 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 3"):
                            condition_image3 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale3 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 4"):
                            condition_image4 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale4 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
                            )
                        with gr.Tab("ControlNet 5"):
                            condition_image5 = gr.Image(
                                type="pil", label="Input Condition Image"
                            )
                            conditioning_scale5 = gr.Slider(
                                0, 10, value=1.0, label="Conditioning Scale", step=0.1
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
                        condition_image1,
                        condition_image2,
                        condition_image3,
                        condition_image4,
                        condition_image5,
                        negative_prompt,
                        strength,
                        guidance_scale,
                        conditioning_scale1,
                        conditioning_scale2,
                        conditioning_scale3,
                        conditioning_scale4,
                        conditioning_scale5,
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
        return "ImageInpainting"

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
            "core/pipeline/multicontrolnet/inpainting",
            "pretrained_name",
            pretrained_name,
        )
        if pretrained_name.startswith("stable-v2"):
            self.config.set(
                "core/pipeline/multicontrolnet/inpainting", "pad_token", "!"
            )
        self._name = pretrained_name
        self._pipe = MultiControlNetForImageInpaintingPipeline.from_core_configure(
            self.config
        )
        self._status = "running"
        self._num_controlnets = len(
            nested_dict_value(
                pretrained_diffusers_infos, pretrained_name, "controlnet", "config"
            )
        )
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        self._num_controlnets = 0
        return self._status

    def serve(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        condition_image1: Optional[Image.Image] = None,
        condition_image2: Optional[Image.Image] = None,
        condition_image3: Optional[Image.Image] = None,
        condition_image4: Optional[Image.Image] = None,
        condition_image5: Optional[Image.Image] = None,
        neg_text: Optional[str] = "",
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        conditioning_scale1: Optional[float] = 1.0,
        conditioning_scale2: Optional[float] = 1.0,
        conditioning_scale3: Optional[float] = 1.0,
        conditioning_scale4: Optional[float] = 1.0,
        conditioning_scale5: Optional[float] = 1.0,
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
            condition_images=[
                condition_image1,
                condition_image2,
                condition_image3,
                condition_image4,
                condition_image5,
            ][: self._num_controlnets],
            neg_text=neg_text,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[
                conditioning_scale1,
                conditioning_scale2,
                conditioning_scale3,
                conditioning_scale4,
                conditioning_scale5,
            ][: self._num_controlnets],
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return image


@register_webui("core/webui/multicontrolnet")
class MultiControlNetWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            MultiControlNetText2ImageWebUI(config),
            MultiControlNetImage2ImageWebUI(config),
            MultiControlNetImageInpaintingWebUI(config),
            CannyWebUI(config),
            DPTWebUI(config),
        ]
        self._iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.name for webui in webuis],
        )

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def name(self):
        return "MultiControlNet"

    @property
    def iface(self):
        return self._iface
