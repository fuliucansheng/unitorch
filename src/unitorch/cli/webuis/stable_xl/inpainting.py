# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import nested_dict_value
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
)
from unitorch.cli.pipelines.stable_xl import StableXLForImageInpaintingPipeline
from unitorch.cli.pipelines.tools import controlnet_processes
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


class StableXLImageInpaintingWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_stable_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, "stable-xl-"
    )
    pretrained_extension_names = list(pretrained_stable_extensions_infos.keys())
    supported_controlnet_names = matched_pretrained_names(
        pretrained_extension_names, "^stable-xl-controlnet-"
    )
    supported_controlnet_process_names = list(controlnet_processes.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names, "stable-xl-lora-"
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
        image = create_element("image_editor", "Input Image")
        mask_image = create_element("image", "Input Image Mask")

        guidance_scale = create_element(
            "slider", "Guidance Scale", min_value=0, max_value=50, step=0.1, default=7.5
        )
        strength = create_element(
            "slider", "Strength", min_value=0, max_value=1, step=0.01, default=0.8
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
        self.num_controlnets = 5
        controlnet_layout_group = create_controlnet_layout(
            self.supported_controlnet_names,
            self.supported_controlnet_process_names,
            num_controlnets=self.num_controlnets,
        )
        controlnets = controlnet_layout_group.controlnets
        controlnet_layout = controlnet_layout_group.layout

        controlnet_params = []
        for controlnet in controlnets:
            controlnet_params += [
                controlnet.checkpoint,
                controlnet.output_image,
                controlnet.guidance_scale,
            ]

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
        output_image = create_element("image", "Output Image")

        # create layouts
        top1 = create_column(pretrain_layout)
        top2 = create_row(
            create_column(prompt, negative_prompt, scale=4),
            create_column(generate),
        )
        left_generation = create_tab(
            create_row(image, mask_image),
            create_row(scheduler, steps),
            create_row(height, width),
            create_row(guidance_scale, strength),
            create_row(seed),
            create_row(freeu_layout),
            name="Generation",
        )
        left_extension = create_tab(
            create_row(controlnet_layout),
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

        start.click(fn=self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(fn=self.stop, outputs=[status], trigger_mode="once")

        image.change(fn=self.composite_images, inputs=[image], outputs=[mask_image])

        for controlnet in controlnets:
            controlnet.input_image.upload(
                fn=self.processing_controlnet_inputs,
                inputs=[controlnet.input_image, controlnet.process],
                outputs=[controlnet.output_image],
            )
            controlnet.process.change(
                fn=self.processing_controlnet_inputs,
                inputs=[controlnet.input_image, controlnet.process],
                outputs=[controlnet.output_image],
            )

        for lora in loras:
            lora.checkpoint.change(
                fn=lambda x: nested_dict_value(
                    pretrained_stable_extensions_infos, x, "text"
                ),
                inputs=[lora.checkpoint],
                outputs=[lora.text],
            )

        generate.click(
            fn=self.serve,
            inputs=[
                prompt,
                image,
                mask_image,
                negative_prompt,
                height,
                width,
                guidance_scale,
                strength,
                steps,
                seed,
                scheduler,
                s1,
                s2,
                b1,
                b2,
                *controlnet_params,
                *lora_params,
            ],
            outputs=[output_image],
            trigger_mode="once",
        )
        image.change(
            lambda x: x["background"].size,
            inputs=[image],
            outputs=[width, height],
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Inpainting", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = StableXLForImageInpaintingPipeline.from_core_configure(
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

    def composite_images(self, images):
        layers = images["layers"]
        if len(layers) == 0:
            return None
        image = layers[0]
        for i in range(1, len(layers)):
            image = Image.alpha_composite(image, layers[i])
        image = image.convert("L")
        image = image.point(lambda p: p < 5 and 255)
        image = ImageOps.invert(image)
        return image

    def processing_controlnet_inputs(self, image, process):
        pfunc = controlnet_processes.get(process, None)
        if pfunc is not None and image is not None:
            return pfunc(image)
        return image

    def serve(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_text: str,
        height: int,
        width: int,
        guidance_scale: float,
        strength: float,
        num_timesteps: int,
        seed: int,
        scheduler: str,
        freeu_s1: float,
        freeu_s2: float,
        freeu_b1: float,
        freeu_b2: float,
        *params,
    ):
        assert self._pipe is not None
        controlnet_params = params[: self.num_controlnets * 3]
        lora_params = params[self.num_controlnets * 3 :]
        controlnet_checkpoints = controlnet_params[::3]
        controlnet_images = controlnet_params[1::3]
        controlnet_guidance_scales = controlnet_params[2::3]
        lora_checkpoints = lora_params[::5]
        lora_weights = lora_params[1::5]
        lora_alphas = lora_params[2::5]
        lora_urls = lora_params[3::5]
        lora_files = lora_params[4::5]
        image = self._pipe(
            text,
            image["background"],
            mask_image,
            negative_text,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            strength=strength,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
            freeu_params=(freeu_s1, freeu_s2, freeu_b1, freeu_b2),
            controlnet_checkpoints=controlnet_checkpoints,
            controlnet_images=controlnet_images,
            controlnet_guidance_scales=controlnet_guidance_scales,
            lora_checkpoints=lora_checkpoints,
            lora_weights=lora_weights,
            lora_alphas=lora_alphas,
            lora_urls=lora_urls,
            lora_files=lora_files,
        )
        return image
