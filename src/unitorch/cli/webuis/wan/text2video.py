# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import nested_dict_value
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
)
from unitorch.cli.pipelines.wan.text2video import WanForText2VideoGenerationPipeline
from unitorch.cli.webuis import (
    supported_scheduler_names,
    matched_pretrained_names,
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
    create_controlnet_layout,
    create_lora_layout,
)
from unitorch.cli.webuis import SimpleWebUI


class WanText2VideoWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_stable_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, "^wan-v2.2-"
    )
    pretrained_extension_names = list(pretrained_stable_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names,
        "^wan-v2.2-lora-",
    )
    supported_schedulers = supported_scheduler_names

    def __init__(self, config: CoreConfigureParser):
        self._pipe = None
        self._status = "Stopped" if self._pipe is None else "Running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]

        # create elements
        pretrain_layout_group = create_pretrain_layout(
            self.supported_pretrained_names, self._name
        )
        name, status, start, stop, pretrain_layout = (
            pretrain_layout_group.name,
            pretrain_layout_group.status,
            pretrain_layout_group.start,
            pretrain_layout_group.stop,
            pretrain_layout_group.layout,
        )

        prompt = create_element(
            "text", "Input Prompt", lines=3, placeholder="Prompt", show_label=False
        )
        negative_prompt = create_element(
            "text",
            "Input Negative Prompt",
            lines=3,
            placeholder="Negative Prompt",
            show_label=False,
        )
        scheduler = create_element(
            "dropdown",
            "Sampling Method",
            values=self.supported_schedulers,
            default=self.supported_schedulers[0],
        )
        steps = create_element(
            "slider", "Diffusion Steps", min_value=1, max_value=100, step=1, default=50
        )
        height = create_element(
            "slider", "Video Height", min_value=1, max_value=2048, step=1, default=480
        )
        width = create_element(
            "slider", "Video Width", min_value=1, max_value=2048, step=1, default=832
        )

        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=50, step=0.1, default=5.0
        )

        seed = create_element(
            "slider", "Seed", min_value=0, max_value=9999, step=1, default=42
        )

        num_frames = create_element(
            "slider", "Video Frames", min_value=1, max_value=100, step=1, default=32
        )
        num_fps = gr.Slider(1, 100, value=16, label="Video FPS", step=1)

        ## extensions
        self.num_loras = 5
        lora_layout_group = create_lora_layout(
            self.supported_lora_names, num_loras=self.num_loras
        )
        loras = lora_layout_group.loras
        lora_layout = lora_layout_group.layout
        lora_params = []
        for lora in loras:
            lora_params += [
                lora.checkpoint,
                lora.weight,
                lora.alpha,
                lora.url,
                lora.file,
            ]

        generate = create_element("button", "Generate", variant="primary", scale=2)
        output_video = create_element("video", "Output Video")

        # create layouts
        top1 = create_column(pretrain_layout)
        top2 = create_row(
            create_column(prompt, negative_prompt, scale=4),
            create_column(generate),
        )
        left_generation = create_tab(
            create_row(height, width),
            create_row(guidance_scale),
            create_row(num_frames, num_fps),
            create_row(scheduler, steps),
            create_row(seed),
            name="Generation",
        )
        left_extension = create_tab(
            create_row(lora_layout),
            name="Extensions",
        )
        left_settings = create_tab(
            name="Settings",
        )

        left = create_tabs(left_generation, left_extension, left_settings)
        right = create_column(output_video)
        iface = create_blocks(top1, top2, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(fn=self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(fn=self.stop, outputs=[status], trigger_mode="once")

        for lora in loras:
            lora.checkpoint.change(
                fn=lambda x: nested_dict_value(
                    pretrained_stable_extensions_infos, x, "text"
                ),
                inputs=[lora.checkpoint],
                outputs=[lora.text],
            )

        generate.click(
            fn=self.generate,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                guidance_scale,
                num_frames,
                num_fps,
                steps,
                seed,
                scheduler,
                *lora_params,
            ],
            outputs=[output_video],
            trigger_mode="once",
        )
        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Text2Video", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = WanForText2VideoGenerationPipeline.from_core_configure(
            self._config,
            pretrained_name=pretrained_name,
        )
        self._status = "Running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_frames: int,
        num_fps: int,
        num_timesteps: int,
        seed: int,
        scheduler: str,
        *params,
    ):
        assert self._pipe is not None
        lora_params = params
        lora_checkpoints = lora_params[0::5]
        lora_weights = lora_params[1::5]
        lora_alphas = lora_params[2::5]
        lora_urls = lora_params[3::5]
        lora_files = lora_params[4::5]
        video = self._pipe(
            prompt,
            negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            num_fps=num_fps,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            lora_checkpoints=lora_checkpoints,
            lora_weights=lora_weights,
            lora_alphas=lora_alphas,
            lora_urls=lora_urls,
            lora_files=lora_files,
        )
        return video
