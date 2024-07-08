# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli.webuis.utils.layouts import (
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
from unitorch.cli import register_webui
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.tools.image import ImageWebUI


@register_webui("core/webui/tools")
class ToolsWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ImageWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Tools", iface=iface)
