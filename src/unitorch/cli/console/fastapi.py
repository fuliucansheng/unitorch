# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import uvicorn
import unitorch.cli
from fastapi import FastAPI
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    set_global_config,
    registered_fastapi,
    init_registered_module,
)
import unitorch.cli.fastapi


@fire.decorators.SetParseFn(str)
def fastapi(config_path_or_dir: str, **kwargs):
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

    enabled_services = config.getdefault("core/cli", "enabled_services", None)
    assert enabled_services is not None
    if isinstance(enabled_services, str):
        enabled_services = [enabled_services]
    assert all(
        fastapi_service in registered_fastapi for fastapi_service in enabled_services
    )
    fastapi_instances = [
        registered_fastapi.get(fastapi_service)["obj"](config)
        for fastapi_service in enabled_services
    ]
    app = FastAPI()

    for fastapi_instance in fastapi_instances:
        fastapi_instance.start()
        app.include_router(fastapi_instance.router)

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


def cli_main():
    fire.Fire(fastapi)
