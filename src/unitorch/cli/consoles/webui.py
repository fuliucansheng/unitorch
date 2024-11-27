# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import gradio as gr
import importlib_resources
import unitorch.cli
from torch.multiprocessing import spawn
from transformers.utils import is_remote_url
from unitorch.utils import read_file, reload_module
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    registered_webui,
    init_registered_module,
)
import unitorch.cli.webuis


@fire.decorators.SetParseFn(str)
def webui(config_path: str, **kwargs):
    config_path = cached_path(config_path)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    config = CoreConfigureParser(config_path, params=params)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)

    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    reload_module(unitorch.cli.webuis)

    enabled_webuis = config.getdefault("core/cli", "enabled_webuis", None)
    single_webui = config.getdefault("core/cli", "single_webui", False)
    title = config.getdefault("core/cli", "title", "Unitorch WebUI")
    assert enabled_webuis is not None
    if isinstance(enabled_webuis, str):
        enabled_webuis = [enabled_webuis]

    for enabled_webui in enabled_webuis:
        assert enabled_webui in registered_webui, f"{enabled_webui} not found"

    webui_instance = lambda webui_name, config: registered_webui.get(webui_name)["obj"](
        config
    )

    webuis = [webui_instance(enabled_webui, config) for enabled_webui in enabled_webuis]

    if single_webui:
        assert (
            len(webuis) == 1
        ), "single_webui can only be used when there is only one webui enabled"
        webuis = webuis[0]
        demo_webui = webuis.iface
    else:
        demo_webui = gr.TabbedInterface(
            interface_list=[webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
            title=title,
        )
    demo_webui.title = title
    demo_webui.theme_css = read_file(
        os.path.join(importlib_resources.files("unitorch"), "cli/assets/style.css")
    )
    demo_webui.css = demo_webui.theme_css

    config.set_default_section("core/cli")
    host = config.getoption("host", "0.0.0.0")
    port = config.getoption("port", 7860)
    share = config.getoption("share", False)
    ssl_keyfile = config.getoption("ssl_keyfile", None)
    ssl_certfile = config.getoption("ssl_certfile", None)
    ssl_verify = config.getoption("ssl_verify", True)
    auth = config.getoption("auth", None)
    demo_webui.launch(
        server_name=host,
        server_port=port,
        share=share,
        favicon_path=os.path.join(
            importlib_resources.files("unitorch"), "cli/assets/icon.png"
        ),
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_verify=ssl_verify,
        auth=auth,
    )

    os._exit(0)


def cli_main():
    fire.Fire(webui)
