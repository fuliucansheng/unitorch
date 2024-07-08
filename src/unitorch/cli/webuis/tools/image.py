# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import numpy as np
import gradio as gr
from PIL import Image, ImageFilter, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import is_opencv_available
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
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


class CannyWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        input_image = create_element("image", "Input Image")
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        # create layouts
        left = create_column(input_image, width, height, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        iface.__enter__()

        generate.click(
            fn=self.canny, inputs=[input_image, width, height], outputs=[output_image]
        )

        iface.__exit__()
        super().__init__(config, iname="Canny", iface=iface)

    def canny(
        self,
        image: Image.Image,
        width: Optional[int] = 0,
        height: Optional[int] = 0,
    ):
        if width > 0 and height > 0:
            image = image.resize((width, height))

        if is_opencv_available():
            import cv2

            image = np.array(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(image, 100, 200)
            image = Image.fromarray(image)
        else:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
        return image


class BlendWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        input_image1 = create_element("image", "Input Image1")
        input_image2 = create_element("image", "Input Image2")
        alpha = create_element(
            "slider", "Alpha", min_value=0, max_value=1, step=0.01, default=0.5
        )
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        # create layouts
        left = create_column(
            create_row(input_image1, input_image2), height, width, alpha, generate
        )
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.blend,
            inputs=[input_image1, input_image2, alpha, height, width],
            outputs=[output_image],
        )

        iface.__exit__()
        super().__init__(config, iname="Blend", iface=iface)

    def blend(
        self,
        image1,
        image2,
        alpha: Optional[float] = 0.5,
        height: Optional[int] = 0,
        width: Optional[int] = 0,
    ):
        if height > 0 and width > 0:
            image1 = image1.resize((width, height))
            image2 = image2.resize((width, height))
        image = Image.blend(image1, image2, alpha)
        return image


class InvertWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        input_image = create_element("image", "Input Image")
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        # create layouts
        left = create_column(input_image, height, width, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.invert, inputs=[input_image, height, width], outputs=[output_image]
        )

        iface.__exit__()

        super().__init__(config, iname="Invert", iface=iface)

    def invert(
        self,
        image,
        height: Optional[int] = 0,
        width: Optional[int] = 0,
    ):
        if height > 0 and width > 0:
            image = image.resize((width, height))
        image = ImageOps.invert(image)
        return image


class CompositeWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        input_image = create_element("image", "Input Image")
        input_mask = create_element("image", "Input Mask")
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        # create layouts
        left = create_column(
            create_row(input_image, input_mask), height, width, generate
        )
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.composite,
            inputs=[input_image, input_mask, height, width],
            outputs=[output_image],
        )

        iface.__exit__()
        super().__init__(config, iname="Composite", iface=iface)

    def composite(
        self,
        image,
        mask,
        height: Optional[int] = 0,
        width: Optional[int] = 0,
    ):
        if height > 0 and width > 0:
            image = image.resize((width, height))
            mask = mask.resize((width, height))
        mask = mask.convert("L")
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.paste(image, (0, 0), mask)
        return result


class ImageWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            CannyWebUI(config),
            BlendWebUI(config),
            InvertWebUI(config),
            CompositeWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Image", iface=iface)
