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
from unitorch.cli.pipelines.sam import SamPipeline


@register_webui("core/webui/sam")
class SamWebUI(GenericWebUI):
    supported_pretrained_names = ["sam-vit-base", "sam-vit-large", "sam-vit-huge"]

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
                pretrained_name = gr.Dropdown(
                    self.supported_pretrained_names,
                    value=self._name,
                    label="Pretrain Checkpoint Name",
                )
                status = gr.Textbox(label="Model Status", value=self._status)
                click_start = gr.Button(value="Start")
                click_stop = gr.Button(value="Stop")
                click_start.click(
                    self.start, inputs=[pretrained_name], outputs=[status]
                )
                click_stop.click(self.stop, outputs=[status])

            with gr.Row(variant="panel"):
                origin_input_image = gr.State(None)
                click_points = gr.State([])
                boxes_points = gr.State([])
                with gr.Tab("Click"):
                    input_image_click = gr.Image(
                        type="pil", label="Input Image", interactive=True
                    )
                    mask_threshold_click = gr.Slider(
                        -20, 20, value=0.0, label="Mask Threshold", step=0.1
                    )
                    with gr.Row():
                        input_click_reset = gr.Button(value="Reset Image")
                        input_click_submit = gr.Button(value="Submit")

                with gr.Tab("Box"):
                    input_image_box = gr.Image(
                        type="pil", label="Input Image", interactive=True
                    )
                    mask_threshold_box = gr.Slider(
                        -20, 20, value=0.0, label="Mask Threshold", step=0.1
                    )
                    with gr.Row():
                        input_box_reset = gr.Button(value="Reset Image")
                        input_box_submit = gr.Button(value="Submit")

                image = gr.Image(type="pil", label="Output Image")

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
                input_click_submit.click(
                    self.serve_click,
                    inputs=[origin_input_image, click_points, mask_threshold_click],
                    outputs=[image],
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
                input_box_submit.click(
                    self.serve_box,
                    inputs=[origin_input_image, boxes_points, mask_threshold_box],
                    outputs=[image],
                )

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "SAM"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self, pretrained_name, **kwargs):
        if self._status == "running":
            self.stop()
        self.config.set("core/pipeline/sam", "pretrained_name", pretrained_name)
        self._name = pretrained_name
        self._pipe = SamPipeline.from_core_configure(self.config)
        self._status = "running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
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
