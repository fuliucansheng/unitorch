# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import unitorch.cli
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    registered_copilot_tools,
)


@fire.decorators.SetParseFn(str)
def copilot(**kwargs):
    os._exit(0)


@fire.decorators.SetParseFn(str)
def cli(name: str, **kwargs):
    if name not in registered_copilot_tools:
        print(f"Copilot tool '{name}' is not registered.")
        os._exit(1)
    cls = registered_copilot_tools[name]
    inst = cls()
    inst.launch(**kwargs)
    os._exit(0)

def main():
    fire.Fire(copilot)

def cli_main():
    fire.Fire(cli)
