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
from unitorch.cli.webuis.stable_flux.text2image import StableFluxText2ImageWebUI


@register_webui("core/webui/stable_flux")
class StableFluxWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableFluxText2ImageWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="StableFlux", iface=iface)
