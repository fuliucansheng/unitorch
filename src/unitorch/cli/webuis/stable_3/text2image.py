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
from unitorch.cli.pipelines.stable_3 import Stable3ForText2ImageGenerationPipeline
from unitorch.cli.pipelines.stable_3 import controlnet_processes
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


class Stable3Text2ImageWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_stable_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, "^stable-3-"
    )
    pretrained_extension_names = list(pretrained_stable_extensions_infos.keys())
    supported_controlnet_names = matched_pretrained_names(
        pretrained_extension_names, "^stable-3-controlnet-"
    )
    supported_controlnet_process_names = list(controlnet_processes.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names, "^stable-3-lora-"
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
        height = create_element(
            "slider", "Image Height", min_value=1, max_value=2048, step=1, default=1024
        )
        width = create_element(
            "slider", "Image Width", min_value=1, max_value=2048, step=1, default=1024
        )

        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=10, step=0.1, default=7.5
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
        controlnet_layout_group = create_controlnet_layout(
            self.supported_controlnet_names,
            self.supported_controlnet_process_names,
            num_controlnets=5,
        )
        controlnets = controlnet_layout_group.controlnets
        controlnet_layout = controlnet_layout_group.layout
        controlnet0, controlnet1, controlnet2, controlnet3, controlnet4 = controlnets
        (
            controlnet0_input_image,
            controlnet0_output_image,
            controlnet0_checkpoint,
            controlnet0_guidance_scale,
            controlnet0_process,
        ) = (
            controlnet0.input_image,
            controlnet0.output_image,
            controlnet0.checkpoint,
            controlnet0.guidance_scale,
            controlnet0.process,
        )
        (
            controlnet1_input_image,
            controlnet1_output_image,
            controlnet1_checkpoint,
            controlnet1_guidance_scale,
            controlnet1_process,
        ) = (
            controlnet1.input_image,
            controlnet1.output_image,
            controlnet1.checkpoint,
            controlnet1.guidance_scale,
            controlnet1.process,
        )
        (
            controlnet2_input_image,
            controlnet2_output_image,
            controlnet2_checkpoint,
            controlnet2_guidance_scale,
            controlnet2_process,
        ) = (
            controlnet2.input_image,
            controlnet2.output_image,
            controlnet2.checkpoint,
            controlnet2.guidance_scale,
            controlnet2.process,
        )
        (
            controlnet3_input_image,
            controlnet3_output_image,
            controlnet3_checkpoint,
            controlnet3_guidance_scale,
            controlnet3_process,
        ) = (
            controlnet3.input_image,
            controlnet3.output_image,
            controlnet3.checkpoint,
            controlnet3.guidance_scale,
            controlnet3.process,
        )
        (
            controlnet4_input_image,
            controlnet4_output_image,
            controlnet4_checkpoint,
            controlnet4_guidance_scale,
            controlnet4_process,
        ) = (
            controlnet4.input_image,
            controlnet4.output_image,
            controlnet4.checkpoint,
            controlnet4.guidance_scale,
            controlnet4.process,
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
            create_row(scheduler, steps),
            create_row(height),
            create_row(width),
            create_row(guidance_scale),
            create_row(seed),
            create_row(freeu_layout),
            name="Generation",
        )
        left_extension = create_tab(
            create_row(controlnet_layout),
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

        controlnet0_input_image.upload(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet0_input_image, controlnet0_process],
            outputs=[controlnet0_output_image],
        )
        controlnet0_process.change(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet0_input_image, controlnet0_process],
            outputs=[controlnet0_output_image],
        )
        controlnet1_input_image.upload(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet1_input_image, controlnet1_process],
            outputs=[controlnet1_output_image],
        )
        controlnet1_process.change(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet1_input_image, controlnet1_process],
            outputs=[controlnet1_output_image],
        )
        controlnet2_input_image.upload(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet2_input_image, controlnet2_process],
            outputs=[controlnet2_output_image],
        )
        controlnet2_process.change(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet2_input_image, controlnet2_process],
            outputs=[controlnet2_output_image],
        )
        controlnet3_input_image.upload(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet3_input_image, controlnet3_process],
            outputs=[controlnet3_output_image],
        )
        controlnet3_process.change(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet3_input_image, controlnet3_process],
            outputs=[controlnet3_output_image],
        )
        controlnet4_input_image.upload(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet4_input_image, controlnet4_process],
            outputs=[controlnet4_output_image],
        )
        controlnet4_process.change(
            fn=self.processing_controlnet_inputs,
            inputs=[controlnet4_input_image, controlnet4_process],
            outputs=[controlnet4_output_image],
        )

        generate.click(
            fn=self.serve,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                guidance_scale,
                steps,
                seed,
                scheduler,
                s1,
                s2,
                b1,
                b2,
                controlnet0_checkpoint,
                controlnet0_output_image,
                controlnet0_guidance_scale,
                controlnet1_checkpoint,
                controlnet1_output_image,
                controlnet1_guidance_scale,
                controlnet2_checkpoint,
                controlnet2_output_image,
                controlnet2_guidance_scale,
                controlnet3_checkpoint,
                controlnet3_output_image,
                controlnet3_guidance_scale,
                controlnet4_checkpoint,
                controlnet4_output_image,
                controlnet4_guidance_scale,
            ],
            outputs=[output_image],
        )
        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Text2Image", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = Stable3ForText2ImageGenerationPipeline.from_core_configure(
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

    def processing_controlnet_inputs(self, image, process):
        pfunc = controlnet_processes.get(process, None)
        if pfunc is not None and image is not None:
            return pfunc(image)
        return image

    def serve(
        self,
        text: str,
        negative_text: Optional[str] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 25,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_s1: Optional[float] = 0.9,
        freeu_s2: Optional[float] = 0.2,
        freeu_b1: Optional[float] = 1.2,
        freeu_b2: Optional[float] = 1.4,
        controlnet0_checkpoint: Optional[str] = None,
        controlnet0_image: Optional[Image.Image] = None,
        controlnet0_guidance_scale: Optional[float] = 0.8,
        controlnet1_checkpoint: Optional[str] = None,
        controlnet1_image: Optional[Image.Image] = None,
        controlnet1_guidance_scale: Optional[float] = 0.8,
        controlnet2_checkpoint: Optional[str] = None,
        controlnet2_image: Optional[Image.Image] = None,
        controlnet2_guidance_scale: Optional[float] = 0.8,
        controlnet3_checkpoint: Optional[str] = None,
        controlnet3_image: Optional[Image.Image] = None,
        controlnet3_guidance_scale: Optional[float] = 0.8,
        controlnet4_checkpoint: Optional[str] = None,
        controlnet4_image: Optional[Image.Image] = None,
        controlnet4_guidance_scale: Optional[float] = 0.8,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            negative_text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
            controlnet_checkpoints=(
                controlnet0_checkpoint,
                controlnet1_checkpoint,
                controlnet2_checkpoint,
                controlnet3_checkpoint,
                controlnet4_checkpoint,
            ),
            controlnet_images=(
                controlnet0_image,
                controlnet1_image,
                controlnet2_image,
                controlnet3_image,
                controlnet4_image,
            ),
            controlnet_guidance_scales=(
                controlnet0_guidance_scale,
                controlnet1_guidance_scale,
                controlnet2_guidance_scale,
                controlnet3_guidance_scale,
                controlnet4_guidance_scale,
            ),
        )
        return image
