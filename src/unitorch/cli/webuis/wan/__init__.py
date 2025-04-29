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
from unitorch.cli.webuis.wan.text2video import WanText2VideoWebUI
from unitorch.cli.webuis.wan.image2video import WanImage2VideoWebUI


@register_webui("core/webui/wan")
class WanWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            WanText2VideoWebUI(config),
            WanImage2VideoWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Wan", iface=iface)
