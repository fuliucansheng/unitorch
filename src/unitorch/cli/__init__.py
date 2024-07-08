# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import sys
import abc
import logging
import traceback
import importlib
import importlib_resources
import importlib.metadata as importlib_metadata
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.utils import is_remote_url
from unitorch.utils import cached_path as hf_cached_path
from unitorch.utils import rpartial
from unitorch.cli.core import CoreConfigureParser


def import_library(library):
    is_load_success = False
    try:
        importlib.import_module(library)
        is_load_success = True
    except importlib_metadata.PackageNotFoundError:
        logging.debug(f"import {library} failed.")
        is_load_success = False
    return is_load_success


# extenstions
UNITORCH_EXTENSTIONS = os.environ.get("UNITORCH_EXTENSTIONS", "")
UNITORCH_EXTENSTIONS = [
    e.strip() for e in re.split(r"[,;]", UNITORCH_EXTENSTIONS) if len(e.strip()) > 0
]


def set_pkg_extensions(extensions: List[str]):
    global UNITORCH_EXTENSTIONS
    UNITORCH_EXTENSTIONS += extensions


def get_pkg_extensions():
    return UNITORCH_EXTENSTIONS


def cached_path(
    url_or_filename,
    cache_dir: Optional[str] = None,
    force_download: Optional[bool] = False,
    proxies: Optional[str] = None,
    resume_download: Optional[bool] = False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file: Optional[bool] = False,
    force_extract: Optional[bool] = False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only: Optional[bool] = False,
) -> Optional[str]:
    if not is_remote_url(url_or_filename):
        pkgs = ["unitorch"] + get_pkg_extensions()
        for pkg in pkgs:
            pkg_filename = os.path.join(importlib_resources.files(pkg), url_or_filename)
            if os.path.exists(pkg_filename):
                url_or_filename = pkg_filename
                break

    return hf_cached_path(
        url_or_filename,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        user_agent=user_agent,
        extract_compressed_file=extract_compressed_file,
        force_extract=force_extract,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
    )


# default core config object
global_config = CoreConfigureParser()


def get_global_config():
    return global_config


def set_global_config(config: Union[CoreConfigureParser, str]):
    global global_config
    if isinstance(config, CoreConfigureParser):
        global_config = config
    elif os.path.exists(config):
        global_config = CoreConfigureParser(config)
    else:
        raise ValueError(f"Can't set global config by {config}")


from unitorch.cli.decorators import (
    add_default_section_for_init,
    add_default_section_for_function,
)


# registry function
def registry_func(
    name: str,
    decorators: Union[Callable, List[Callable]] = None,
    save_dict: Optional[Dict] = dict(),
):
    def actual_func(obj):
        save_dict[name] = dict(
            {
                "obj": obj,
                "decorators": decorators,
            }
        )
        return obj

    return actual_func


# register score/dataset/loss/model/optim/writer/scheduler/task
core_modules = [
    "score",
    "dataset",
    "loss",
    "model",
    "optim",
    "writer",
    "scheduler",
    "task",
]

for module in core_modules:
    globals()[f"registered_{module}"] = dict()
    globals()[f"register_{module}"] = partial(
        registry_func,
        save_dict=globals()[f"registered_{module}"],
    )

# register process function
registered_process = dict()


def get_import_module(import_file):
    modules = sys.modules.copy()
    for k, v in modules.items():
        if hasattr(v, "__file__") and v.__file__ == import_file:
            return v
    raise "can't find the module"


def register_process(
    name: str,
    decorators: Union[Callable, List[Callable]] = None,
):
    def actual_func(obj):
        trace_stacks = traceback.extract_stack()
        import_file = trace_stacks[-2][0]
        import_cls_name = trace_stacks[-2][2]
        import_module = get_import_module(import_file)
        registered_process[name] = dict(
            {
                "cls": {
                    "module": import_module,
                    "name": import_cls_name,
                },
                "obj": obj,
                "decorators": decorators,
            }
        )
        return obj

    return actual_func


# init registered modules
def init_registered_module(
    name: str,
    config: CoreConfigureParser,
    registered_module: Dict,
    **kwargs,
):
    if name not in registered_module:
        return

    v = registered_module[name]

    if v["decorators"]:
        return v["decorators"](v["obj"]).from_core_configure(config, **kwargs)
    return v["obj"].from_core_configure(config, **kwargs)


def init_registered_process(
    name: str,
    config: CoreConfigureParser,
    **kwargs,
):
    if name not in registered_process:
        return

    v = registered_process[name]
    cls = getattr(v["cls"]["module"], v["cls"]["name"])
    inst = cls.from_core_configure(config, **kwargs)
    if v["decorators"]:
        return rpartial(v["decorators"](v["obj"]), inst)
    else:
        return rpartial(v["obj"], inst)


# script module
class GenericScript(metaclass=abc.ABCMeta):
    def __init__(self, config: CoreConfigureParser):
        pass

    @abc.abstractmethod
    def launch(self, **kwargs):
        pass


registered_script = dict()
register_script = partial(
    registry_func,
    save_dict=registered_script,
)


# service module
class GenericService(metaclass=abc.ABCMeta):
    def __init__(self, config: CoreConfigureParser):
        pass

    @abc.abstractmethod
    def start(self, **kwargs):
        pass

    @abc.abstractmethod
    def stop(self, **kwargs):
        pass

    @abc.abstractmethod
    def restart(self, **kwargs):
        pass


registered_service = dict()
register_service = partial(
    registry_func,
    save_dict=registered_service,
)


# webui module
class GenericWebUI(metaclass=abc.ABCMeta):
    ignore_elements = []

    def __init__(self, config: CoreConfigureParser):
        pass

    @property
    def iname(self):
        pass

    @property
    def iface(self):
        pass

    @abc.abstractmethod
    def start(self, **kwargs):
        pass

    @abc.abstractmethod
    def stop(self, **kwargs):
        pass


registered_webui = dict()
register_webui = partial(
    registry_func,
    save_dict=registered_webui,
)


# fastapi module
class GenericFastAPI(metaclass=abc.ABCMeta):
    def __init__(self, config: CoreConfigureParser):
        pass

    @property
    def router(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass


registered_fastapi = dict()
register_fastapi = partial(
    registry_func,
    save_dict=registered_fastapi,
)


# usful function
from unitorch.cli.writers import WriterMixin, WriterOutputs

# import cli modules
import unitorch.cli.datasets
import unitorch.cli.losses
import unitorch.cli.models
import unitorch.cli.optims
import unitorch.cli.schedulers
import unitorch.cli.scores
import unitorch.cli.tasks
import unitorch.cli.writers
