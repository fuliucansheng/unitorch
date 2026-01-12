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
from unitorch.cli.webuis.stable_flux.image2image import StableFluxImage2ImageWebUI
from unitorch.cli.webuis.stable_flux.image_redux import StableFluxImageReduxWebUI
from unitorch.cli.webuis.stable_flux.inpainting import StableFluxImageInpaintingWebUI
from unitorch.cli.webuis.stable_flux.redux_inpainting import (
    StableFluxReduxInpaintingWebUI,
)
from unitorch.cli.webuis.stable_flux.kontext2image import StableFluxKontext2ImageWebUI


@register_webui("core/webui/stable_flux")
class StableFluxWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableFluxText2ImageWebUI(config),
            StableFluxImage2ImageWebUI(config),
            StableFluxImageReduxWebUI(config),
            StableFluxImageInpaintingWebUI(config),
            StableFluxReduxInpaintingWebUI(config),
            StableFluxKontext2ImageWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="StableFlux", iface=iface)
