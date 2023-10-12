# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import gradio as gr
import unitorch.cli
from torch.multiprocessing import spawn
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    set_global_config,
    registered_webui,
    init_registered_module,
)
import unitorch.cli.webui


def start_webui_process(pid, webui_name: str, config: CoreConfigureParser):
    webui_instance = registered_webui.get(webui_name)["obj"](config)
    webui_instance.start()


@fire.decorators.SetParseFn(str)
def webui(config_path_or_dir: str, **kwargs):
    config_file = kwargs.pop("config_file", "config.ini")

    if os.path.isdir(config_path_or_dir):
        config_path = os.path.join(config_path_or_dir, config_file)
        sys.path.insert(0, config_path_or_dir)
        for f in os.listdir(config_path_or_dir):
            fpath = os.path.normpath(os.path.join(config_path_or_dir, f))
            if (
                not f.startswith("_")
                and not f.startswith(".")
                and (f.endswith(".py") or os.path.isdir(fpath))
            ):
                fname = f[:-3] if f.endswith(".py") else f
                module = importlib.import_module(f"{fname}")
    else:
        config_path = cached_path(config_path_or_dir)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    if config_path is not None:
        config = CoreConfigureParser(config_path, params=params)
    else:
        config = CoreConfigureParser(params=params)

    set_global_config(config)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)

    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    enabled_webuis = config.getdefault("core/cli", "enabled_webuis", None)
    assert enabled_webuis is not None
    if isinstance(enabled_webuis, str):
        enabled_webuis = [enabled_webuis]
    assert all(enabled_webui in registered_webui for enabled_webui in enabled_webuis)

    webui_processes = []
    for enabled_webui in enabled_webuis:
        webui_processes.append(
            spawn(
                start_webui_process,
                args=(enabled_webui, config),
                join=False,
                daemon=True,
            )
        )

    for webui_process in webui_processes:
        webui_process.join()

    os._exit(0)


def cli_main():
    fire.Fire(webui)
