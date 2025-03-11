# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from typing import List, Tuple
from PIL import Image, ImageDraw
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.models.dpt import pretrained_dpt_infos
from unitorch.cli.pipelines.dpt import DPTForDepthEstimationPipeline
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
)
from unitorch.cli.webuis import SimpleWebUI


@register_webui("core/webui/dpt")
class DPTWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_dpt_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        "^dpt-",
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
        self._pipe = None
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

        input_image = create_element("image", "Input Image")
        output_image = create_element("image", "Output Image")
        generate = create_element("button", "Generate")

        # Create the blocks
        left = create_column(input_image, generate)
        right = create_column(output_image)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # Create the events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        generate.click(
            self.serve,
            inputs=[input_image],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="DPT", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = DPTForDepthEstimationPipeline.from_core_configure(
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

    def serve(
        self,
        image: Image.Image,
    ):
        assert self._pipe is not None
        result = self._pipe(
            image,
        )
        return result
