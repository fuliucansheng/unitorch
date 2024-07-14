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
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
)
from unitorch.cli.pipelines.stable import StableForImageResolutionPipeline
from unitorch.cli.pipelines.stable import controlnet_processes
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
    create_freeu_layout,
)
from unitorch.cli.webuis import SimpleWebUI


class StableImageResolutionWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_stable_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, ["^stable-v1.5-", "^stable-v2-", "^stable-v2.1"]
    )
    pretrained_extension_names = list(pretrained_stable_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names,
        ["^stable-v1.5-lora-", "^stable-v2-lora-", "^stable-v2.1-lora-"],
    )
    supported_schedulers = supported_scheduler_names

    def __init__(self, config: CoreConfigureParser):
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
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
            "slider", "Diffusion Steps", min_value=1, max_value=100, step=1, default=25
        )
        image = create_element("image", "Input Image")

        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=10, step=0.1, default=7.5
        )
        noise_level = create_element(
            "slider", "Noise Level", min_value=0, max_value=50, step=0.1, default=20
        )

        seed = create_element(
            "slider", "Seed", min_value=0, max_value=9999, step=1, default=42
        )

        freeu_layout_group = create_freeu_layout()
        s1, s2, b1, b2, freeu_layout = (
            freeu_layout_group.s1,
            freeu_layout_group.s2,
            freeu_layout_group.b1,
            freeu_layout_group.b2,
            freeu_layout_group.layout,
        )

        ## extensions
        lora_layout_group = create_lora_layout(self.supported_lora_names, num_loras=5)
        loras = lora_layout_group.loras
        lora_layout = lora_layout_group.layout
        lora0, lora1, lora2, lora3, lora4 = loras
        (
            lora0_checkpoint,
            lora0_weight,
            lora0_alpha,
            lora0_url,
            lora0_file,
        ) = (
            lora0.checkpoint,
            lora0.weight,
            lora0.alpha,
            lora0.url,
            lora0.file,
        )
        (
            lora1_checkpoint,
            lora1_weight,
            lora1_alpha,
            lora1_url,
            lora1_file,
        ) = (
            lora1.checkpoint,
            lora1.weight,
            lora1.alpha,
            lora1.url,
            lora1.file,
        )
        (
            lora2_checkpoint,
            lora2_weight,
            lora2_alpha,
            lora2_url,
            lora2_file,
        ) = (
            lora2.checkpoint,
            lora2.weight,
            lora2.alpha,
            lora2.url,
            lora2.file,
        )
        (
            lora3_checkpoint,
            lora3_weight,
            lora3_alpha,
            lora3_url,
            lora3_file,
        ) = (
            lora3.checkpoint,
            lora3.weight,
            lora3.alpha,
            lora3.url,
            lora3.file,
        )
        (
            lora4_checkpoint,
            lora4_weight,
            lora4_alpha,
            lora4_url,
            lora4_file,
        ) = (
            lora4.checkpoint,
            lora4.weight,
            lora4.alpha,
            lora4.url,
            lora4.file,
        )
        generate = create_element("button", "Generate", variant="primary", scale=2)
        output_image = create_element("image", "Output Image")

        # create layouts
        top1 = create_column(pretrain_layout)
        top2 = create_row(
            create_column(prompt, negative_prompt, scale=4),
            create_column(generate),
        )
        left_generation = create_tab(
            create_row(image),
            create_row(scheduler, steps),
            create_row(guidance_scale),
            create_row(noise_level),
            create_row(seed),
            create_row(freeu_layout),
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
        right = create_column(output_image)
        iface = create_blocks(top1, top2, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(fn=self.start, inputs=[name], outputs=[status])
        stop.click(fn=self.stop, outputs=[status])

        generate.click(
            fn=self.serve,
            inputs=[
                prompt,
                image,
                negative_prompt,
                guidance_scale,
                noise_level,
                steps,
                seed,
                scheduler,
                s1,
                s2,
                b1,
                b2,
                lora0_checkpoint,
                lora0_weight,
                lora0_alpha,
                lora0_url,
                lora0_file,
                lora1_checkpoint,
                lora1_weight,
                lora1_alpha,
                lora1_url,
                lora1_file,
                lora2_checkpoint,
                lora2_weight,
                lora2_alpha,
                lora2_url,
                lora2_file,
                lora3_checkpoint,
                lora3_weight,
                lora3_alpha,
                lora3_url,
                lora3_file,
                lora4_checkpoint,
                lora4_weight,
                lora4_alpha,
                lora4_url,
                lora4_file,
            ],
            outputs=[output_image],
        )
        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Resolution", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = StableForImageResolutionPipeline.from_core_configure(
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
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(
        self,
        text: str,
        image: Image.Image,
        negative_text: Optional[str] = None,
        guidance_scale: Optional[float] = 7.5,
        noise_level: Optional[float] = 20,
        num_timesteps: Optional[int] = 25,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
        lora0_checkpoint: Optional[str] = None,
        lora0_weight: Optional[float] = 1.0,
        lora0_alpha: Optional[float] = 32,
        lora0_url: Optional[str] = None,
        lora0_file: Optional[str] = None,
        lora1_checkpoint: Optional[str] = None,
        lora1_weight: Optional[float] = 1.0,
        lora1_alpha: Optional[float] = 32,
        lora1_url: Optional[str] = None,
        lora1_file: Optional[str] = None,
        lora2_checkpoint: Optional[str] = None,
        lora2_weight: Optional[float] = 1.0,
        lora2_alpha: Optional[float] = 32,
        lora2_url: Optional[str] = None,
        lora2_file: Optional[str] = None,
        lora3_checkpoint: Optional[str] = None,
        lora3_weight: Optional[float] = 1.0,
        lora3_alpha: Optional[float] = 32,
        lora3_url: Optional[str] = None,
        lora3_file: Optional[str] = None,
        lora4_checkpoint: Optional[str] = None,
        lora4_weight: Optional[float] = 1.0,
        lora4_alpha: Optional[float] = 32,
        lora4_url: Optional[str] = None,
        lora4_file: Optional[str] = None,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            image,
            negative_text,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
            lora_checkpoints=(
                lora0_checkpoint,
                lora1_checkpoint,
                lora2_checkpoint,
                lora3_checkpoint,
                lora4_checkpoint,
            ),
            lora_weights=(
                lora0_weight,
                lora1_weight,
                lora2_weight,
                lora3_weight,
                lora4_weight,
            ),
            lora_alphas=(
                lora0_alpha,
                lora1_alpha,
                lora2_alpha,
                lora3_alpha,
                lora4_alpha,
            ),
            lora_urls=(
                lora0_url,
                lora1_url,
                lora2_url,
                lora3_url,
                lora4_url,
            ),
            lora_files=(
                lora0_file,
                lora1_file,
                lora2_file,
                lora3_file,
                lora4_file,
            ),
        )
        return image
