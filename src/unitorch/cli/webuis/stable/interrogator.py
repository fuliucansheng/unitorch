# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from unitorch.utils import nested_dict_value
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.stable import ClipInterrogatorPipeline
from unitorch.cli.models.blip import pretrained_blip_infos
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


class InterrogatorWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_blip_infos.keys())
    supported_pretrained_names = ["clip-vit-large-patch14"]

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

        image = create_element("image", "Input Image")
        generate = create_element("button", "Generate")
        fast_prompt = create_element("text", "Fast Prompt", lines=2)
        classic_prompt = create_element("text", "Classic Prompt", lines=2)
        best_prompt = create_element("text", "Best Prompt", lines=2)
        negative_prompt = create_element("text", "Negative Prompt", lines=2)

        # create blocks
        left = create_column(image, generate)
        right = create_column(fast_prompt, classic_prompt, best_prompt, negative_prompt)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        generate.click(
            self.serve,
            inputs=[image],
            outputs=[fast_prompt, classic_prompt, best_prompt, negative_prompt],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Interrogator", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = ClipInterrogatorPipeline.from_core_configure(
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
    ):
        assert self._pipe is not None
        result = self._pipe(image)
        return (
            result.fast_prompt,
            result.classic_prompt,
            result.best_prompt,
            result.negative_prompt,
        )
