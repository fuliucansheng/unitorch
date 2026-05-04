# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import fire
import unitorch.cli
from unitorch.cli import registered_copilot_tools


@fire.decorators.SetParseFn(str)
def copilot(**kwargs):
    os._exit(0)


@fire.decorators.SetParseFn(str)
def cli(name: str, **kwargs):
    if name not in registered_copilot_tools:
        print(f"Copilot tool {name!r} is not registered.")
        os._exit(1)
    entry = registered_copilot_tools[name]
    inst = entry["obj"]()
    inst.launch(**kwargs)
    os._exit(0)


def main():
    fire.Fire(copilot)


def cli_main():
    fire.Fire(cli)
