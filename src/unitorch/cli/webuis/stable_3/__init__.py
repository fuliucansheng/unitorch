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
from unitorch.cli.webuis.stable_3.text2image import Stable3Text2ImageWebUI
from unitorch.cli.webuis.stable_3.image2image import Stable3Image2ImageWebUI
from unitorch.cli.webuis.stable_3.inpainting import Stable3ImageInpaintingWebUI


@register_webui("core/webui/stable_3")
class Stable3WebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            Stable3Text2ImageWebUI(config),
            Stable3Image2ImageWebUI(config),
            Stable3ImageInpaintingWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Stable3", iface=iface)
