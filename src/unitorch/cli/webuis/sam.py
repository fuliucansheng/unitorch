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
from unitorch.cli.models.sam import pretrained_sam_infos
from unitorch.cli.pipelines.sam import SamForSegmentationPipeline
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


@register_webui("core/webui/sam")
class SamWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_sam_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        "^sam-",
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

        input_image_click = create_element("image", "Input Image")
        mask_threshold_click = create_element(
            "slider", "Mask Threshold", default=0, min_value=-20, max_value=20, step=0.1
        )
        input_click_reset = create_element("button", "Reset")
        input_click_segment = create_element("button", "Segment")
        input_image_box = create_element("image", "Input Image")
        mask_threshold_box = create_element(
            "slider", "Mask Threshold", default=0, min_value=-20, max_value=20, step=0.1
        )
        input_box_reset = create_element("button", "Reset")
        input_box_segment = create_element("button", "Segment")
        output_image = create_element("image", "Output Image")

        # create blocks
        click = create_tab(
            create_column(
                input_image_click,
                mask_threshold_click,
                create_row(input_click_reset, input_click_segment),
            ),
            name="Click",
        )
        box = create_tab(
            create_column(
                input_image_box,
                mask_threshold_box,
                create_row(input_box_reset, input_box_segment),
            ),
            name="Box",
        )
        left = create_tabs(click, box)
        right = create_column(output_image)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status])
        stop.click(self.stop, outputs=[status])

        origin_input_image = gr.State(None)
        click_points = gr.State([])
        boxes_points = gr.State([])
        input_image_click.upload(
            lambda image: image.copy() if image is not None else None,
            inputs=[input_image_click],
            outputs=[origin_input_image],
        )
        input_image_click.select(
            self.add_click_points,
            inputs=[origin_input_image, click_points],
            outputs=[input_image_click, click_points],
        )
        input_click_reset.click(
            lambda x: (x, []),
            inputs=[origin_input_image],
            outputs=[input_image_click, click_points],
        )
        input_click_segment.click(
            self.serve_click,
            inputs=[origin_input_image, click_points, mask_threshold_click],
            outputs=[output_image],
        )

        input_image_box.upload(
            lambda image: image.copy() if image is not None else None,
            inputs=[input_image_box],
            outputs=[origin_input_image],
        )
        input_image_box.select(
            self.add_click_points,
            [origin_input_image, boxes_points],
            [input_image_box, boxes_points],
        )
        input_box_reset.click(
            lambda x: (x, []),
            inputs=[origin_input_image],
            outputs=[input_image_box, boxes_points],
        )
        input_box_segment.click(
            self.serve_box,
            inputs=[origin_input_image, boxes_points, mask_threshold_box],
            outputs=[output_image],
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="SAM", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = SamForSegmentationPipeline.from_core_configure(
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

    def add_click_points(self, image, click_points, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        click_points = click_points + [(x, y)]
        new_image = image.copy()
        draw = ImageDraw.Draw(new_image)
        point_color = (255, 0, 0)
        radius = 3
        for point in click_points:
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=point_color
            )

        return new_image, click_points

    def serve_click(
        self,
        image: Image.Image,
        click_points: List[Tuple[int, int]] = [(0, 0)],
        mask_threshold: float = 0.5,
    ):
        assert self._pipe is not None
        result = self._pipe(
            image,
            points=click_points,
            mask_threshold=mask_threshold,
        )
        return result

    def serve_box(
        self,
        image: Image.Image,
        boxes_points: List[Tuple[int, int]] = [(0, 0)],
        mask_threshold: float = 0.5,
    ):
        assert self._pipe is not None
        x1 = min([point[0] for point in boxes_points])
        y1 = min([point[1] for point in boxes_points])
        x2 = max([point[0] for point in boxes_points])
        y2 = max([point[1] for point in boxes_points])
        result = self._pipe(
            image,
            boxes=[[(x1, y1, x2, y2)]],
            mask_threshold=mask_threshold,
        )
        return result
