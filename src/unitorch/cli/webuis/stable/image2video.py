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
from unitorch.cli.pipelines.stable import StableForImage2VideoGenerationPipeline
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


class StableImage2VideoWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_stable_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names, "^stable-video-"
    )
    pretrained_extension_names = list(pretrained_stable_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names,
        "^stable-video-lora-",
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
        height = create_element(
            "slider", "Video Height", min_value=1, max_value=2048, step=1, default=576
        )
        width = create_element(
            "slider", "Video Width", min_value=1, max_value=2048, step=1, default=1024
        )

        num_frames = create_element(
            "slider", "Video Frames", min_value=1, max_value=100, step=1, default=30
        )
        num_fps = gr.Slider(1, 100, value=7, label="Video FPS", step=1)

        min_guidance_scale = create_element(
            "slider",
            "Min Guidance Scale",
            min_value=0,
            max_value=10,
            step=0.1,
            default=1.0,
        )
        max_guidance_scale = create_element(
            "slider",
            "Max Guidance Scale",
            min_value=0,
            max_value=10,
            step=0.1,
            default=2.5,
        )
        motion_bucket_id = gr.Slider(
            1, 255, value=127, label="Video Motion Bucket Id", step=1
        )
        decode_chunk_size = gr.Slider(1, 32, value=8, label="Decode Chunk Size", step=1)
        seed = create_element(
            "slider", "Seed", min_value=0, max_value=9999, step=1, default=42
        )

        ## extensions
        generate = create_element("button", "Generate", variant="primary", scale=2)
        output_video = create_element("video", "Output Video")

        # create layouts
        top = create_column(pretrain_layout)
        left_generation = create_tab(
            create_row(image),
            create_row(height),
            create_row(width),
            create_row(num_frames, num_fps),
            create_row(min_guidance_scale),
            create_row(max_guidance_scale),
            create_row(motion_bucket_id),
            create_row(decode_chunk_size),
            create_row(scheduler, steps),
            create_row(seed),
            create_row(generate),
            name="Generation",
        )
        left_extension = create_tab(
            name="Extensions",
        )
        left_settings = create_tab(
            name="Settings",
        )

        left = create_tabs(left_generation, left_extension, left_settings)
        right = create_column(output_video)
        iface = create_blocks(top, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(fn=self.start, inputs=[name], outputs=[status])
        stop.click(fn=self.stop, outputs=[status])

        generate.click(
            fn=self.serve,
            inputs=[
                image,
                height,
                width,
                num_frames,
                num_fps,
                min_guidance_scale,
                max_guidance_scale,
                motion_bucket_id,
                decode_chunk_size,
                steps,
                seed,
                scheduler,
            ],
            outputs=[output_video],
        )
        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Image2Video", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = StableForImage2VideoGenerationPipeline.from_core_configure(
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
        image: Image.Image,
        height: Optional[int] = 576,
        width: Optional[int] = 1024,
        num_frames: Optional[int] = 30,
        num_fps: Optional[int] = 6,
        min_guidance_scale: Optional[float] = 1.0,
        max_guidance_scale: Optional[float] = 2.5,
        motion_bucket_id: Optional[int] = 127,
        decode_chunk_size: Optional[int] = 8,
        num_timesteps: Optional[int] = 25,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
    ):
        assert self._pipe is not None
        video = self._pipe(
            image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_fps=num_fps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            motion_bucket_id=motion_bucket_id,
            decode_chunk_size=decode_chunk_size,
            num_timesteps=num_timesteps,
            seed=seed,
            scheduler=scheduler,
        )
        return video
