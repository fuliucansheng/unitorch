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
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.stable_xl.text2image import StableXLText2ImageWebUI
from unitorch.cli.webuis.stable_xl.image2image import StableXLImage2ImageWebUI
from unitorch.cli.webuis.stable_xl.inpainting import StableXLImageInpaintingWebUI


@register_webui("core/webui/stable_xl")
class StableXLWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableXLText2ImageWebUI(config),
            StableXLImage2ImageWebUI(config),
            StableXLImageInpaintingWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="StableXL", iface=iface)
