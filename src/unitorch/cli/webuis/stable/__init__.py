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
from unitorch.cli.webuis.stable.text2image import StableText2ImageWebUI

from unitorch.cli.webuis.stable.image2image import StableImage2ImageWebUI
from unitorch.cli.webuis.stable.inpainting import StableImageInpaintingWebUI
from unitorch.cli.webuis.stable.resolution import StableImageResolutionWebUI
from unitorch.cli.webuis.stable.image2video import StableImage2VideoWebUI
from unitorch.cli.webuis.stable.interrogator import InterrogatorWebUI


@register_webui("core/webui/stable")
class StableWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableText2ImageWebUI(config),
            StableImage2ImageWebUI(config),
            StableImageInpaintingWebUI(config),
            StableImageResolutionWebUI(config),
            StableImage2VideoWebUI(config),
            InterrogatorWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Stable", iface=iface)
