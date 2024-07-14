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
from unitorch.cli.models.llava import (
    pretrained_llava_infos,
    pretrained_llava_extensions_infos,
)
from unitorch.cli.pipelines.llava import LlavaMistralClipForGenerationPipeline
from unitorch.cli.webuis import (
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
    create_lora_layout,
    create_freeu_layout,
)
from unitorch.cli.webuis import SimpleWebUI


class LlavaMistralClipGenerationWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_llava_infos.keys())
    supported_pretrained_names = matched_pretrained_names(pretrained_names, "^llava-")
    pretrained_extension_names = list(pretrained_llava_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names, "^llava-lora-"
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]

        # Create the elements
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

        prompt = create_element("text", "Input Prompt", lines=3)
        image = create_element("image", "Input Image")
        generate = create_element("button", "Generate")
        result = create_element("text", "Output Result", lines=3)

        # Create the blocks
        left = create_column(prompt, image, lora_layout, generate)
        right = create_column(result)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # Create the events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status])
        stop.click(self.stop, outputs=[status])
        generate.click(
            self.serve,
            inputs=[
                prompt,
                image,
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
            outputs=[result],
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="MistralClip", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = LlavaMistralClipForGenerationPipeline.from_core_configure(
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
        result = self._pipe(
            text,
            image,
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
        return result
