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
from unitorch.cli.models.grounding_dino import pretrained_grounding_dino_infos
from unitorch.cli.pipelines.grounding_dino import GroundingDinoForDetectionPipeline
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
    create_freeu_layout,
)
from unitorch.cli.webuis import SimpleWebUI


@register_webui("core/webui/grounding_dino")
class GroundingDinoWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_grounding_dino_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        "^grounding-dino-",
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
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

        input_text = create_element("text", "Input Text")
        input_image = create_element("image", "Input Image")
        text_threshold = create_element(
            "slider",
            "Text Threshold",
            default=0.25,
            min_value=0,
            max_value=1,
            step=0.01,
        )
        box_threshold = create_element(
            "slider", "Box Threshold", default=0.25, min_value=0, max_value=1, step=0.01
        )
        detect = create_element("button", "Detect")
        output_image = create_element("image", "Output Image")

        # create blocks
        left = create_column(
            input_text, input_image, create_row(text_threshold, box_threshold), detect
        )
        right = create_column(output_image)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        detect.click(
            self.serve,
            inputs=[input_text, input_image, text_threshold, box_threshold],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="GroundingDino", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = GroundingDinoForDetectionPipeline.from_core_configure(
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
        text_threshold: float = 0.25,
        box_threshold: float = 0.25,
    ):
        assert self._pipe is not None
        result = self._pipe(
            text,
            image,
            text_threshold=text_threshold,
            box_threshold=box_threshold,
        )
        return result
