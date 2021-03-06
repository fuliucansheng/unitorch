# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import importlib
import unitorch.cli
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser, registered_script
from unitorch_cli import load_template


@fire.decorators.SetParseFn(str)
def script(script_path_or_dir: str, **kwargs):
    config_file = kwargs.pop("config_file", "config.ini")

    if script_path_or_dir and os.path.isdir(script_path_or_dir):
        config_path = os.path.join(script_path_or_dir, config_file)
        sys.path.insert(0, script_path_or_dir)
        for f in os.listdir(script_path_or_dir):
            fpath = os.path.normpath(os.path.join(script_path_or_dir, f))
            if not f.startswith("_") and not f.startswith(".") and (f.endswith(".py") or os.path.isdir(fpath)):
                fname = f[:-3] if f.endswith(".py") else f
                module = importlib.import_module(f"{fname}")

    elif script_path_or_dir and not script_path_or_dir.endswith(".ini"):
        load_template(script_path_or_dir)
        config_path = os.path.join(script_path_or_dir, config_file)
        config_path = cached_path(config_path)
        if config_path is None:
            config_path = cached_path(config_file)
    else:
        config_path = cached_path(script_path_or_dir)

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

    depends_templates = config.getdefault("core/cli", "depends_templates", None)
    script_name = config.getdefault("core/cli", "script_name", None)

    assert script_name is not None

    if depends_templates:
        for template in depends_templates:
            load_template(template)

    main_script_cls = registered_script.get(script_name)
    if main_script_cls is None:
        raise ValueError(f"script {script_name} not found")
    inst = main_script_cls["obj"](config)
    inst.run()
    os._exit(0)


def cli_main():
    fire.Fire(script)
