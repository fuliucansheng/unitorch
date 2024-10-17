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
    registered_task,
    registered_script,
    init_registered_module,
)


@fire.decorators.SetParseFn(str)
def eval(config_path: str, **kwargs):
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

    task_name = config.getdefault("core/cli", "task_name", None)
    assert task_name is not None and task_name in registered_task
    cli_task = init_registered_module(task_name, config, registered_task)

    cli_task.eval()

    os._exit(0)


def cli_main():
    fire.Fire(eval)
