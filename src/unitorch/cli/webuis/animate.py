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
from unitorch.cli.pipelines.animate import (
    AnimateForText2VideoGenerationPipeline,
)
from unitorch.cli.webuis import matched_pretrained_names

class AnimateText2VideoWebUI(GenericWebUI):
    match_patterns = [
        "^stable-v1.5.*animate",
    ]
    block_patterns = [
        ".*controlnet",
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
                    height = gr.Slider(512, 1024, value=512, label="Video Height")
                    width = gr.Slider(512, 1024, value=512, label="Video Width")
                    guidance_scale = gr.Slider(
                        0, 10, value=7.5, label="Guidance Scale", step=0.1
                    )
                    num_frames = gr.Slider(
                        1, 30, value=16, label="Video Frames", step=1
                    )
                    steps = gr.Slider(
                        0, 1000, value=25, label="Diffusion Steps", step=1
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

                video = gr.Video(label="Output Video")
                submit.click(
                    self.serve,
                    inputs=[
                        prompt,
                        negative_prompt,
                        height,
                        width,
                        num_frames,
                        guidance_scale,
                        steps,
                        seed,
                        scheduler,
                        freeu_s1,
                        freeu_s2,
                        freeu_b1,
                        freeu_b2,
                    ],
                    outputs=[video],
                )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "Text2Video"

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
        num_frames: Optional[int] = 16,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 25,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
    ):
        assert self._pipe is not None
        video = self._pipe(
            text,
            negative_text,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
        )
        return video

@register_webui("core/webui/animate")
class AnimateWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            AnimateText2VideoWebUI(config),
        ]
        self._iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.name for webui in webuis],
        )

    @property
    def name(self):
        return "Animate"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self):
        pass

    def stop(self):
        pass