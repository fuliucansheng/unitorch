# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import unitorch.cli
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    registered_script,
    init_registered_module,
)
import unitorch.cli.pipelines
import unitorch.cli.scripts


@fire.decorators.SetParseFn(str)
def launch(config_path: str, **kwargs):
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

    script_name = config.getdefault("core/cli", "script_name", None)

    assert script_name is not None
    assert script_name in registered_script, f"{script_name} not found"

    main_script_cls = registered_script.get(script_name)
    if main_script_cls is None:
        raise ValueError(f"script {script_name} not found")
    inst = main_script_cls["obj"](config)
    inst.launch()
    os._exit(0)


def cli_main():
    fire.Fire(launch)
