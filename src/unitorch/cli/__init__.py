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
from typing import Any, Callable, Dict, List, Optional, Union

from unitorch.utils import cached_path as hf_cached_path
from unitorch.utils import rpartial, is_remote_url
from unitorch.cli.core import Config


def import_library(library: str) -> bool:
    """Try to import *library*; return True on success, False on failure."""
    try:
        importlib.import_module(library)
        return True
    except importlib_metadata.PackageNotFoundError:
        logging.debug("import %s failed.", library)
        return False


UNITORCH_HF_ENDPOINT = os.environ.get("UNITORCH_HF_ENDPOINT", "https://huggingface.co")


def hf_endpoint_url(url: str) -> str:
    if is_remote_url(url):
        return url
    return f"{UNITORCH_HF_ENDPOINT}/{url.lstrip('/')}"


UNITORCH_EXTENSIONS: List[str] = [
    e.strip()
    for e in re.split(r"[,;]", os.environ.get("UNITORCH_EXTENSIONS", ""))
    if e.strip()
]


def set_pkg_extensions(extensions: List[str]) -> None:
    global UNITORCH_EXTENSIONS
    UNITORCH_EXTENSIONS += extensions


def get_pkg_extensions() -> List[str]:
    return UNITORCH_EXTENSIONS


def cached_path(
    url_or_filename: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[str] = None,
    resume_download: bool = False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file: bool = False,
    force_extract: bool = False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only: bool = False,
) -> Optional[str]:
    if not is_remote_url(url_or_filename):
        for pkg in ["unitorch"] + get_pkg_extensions():
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


from unitorch.cli.decorators import (
    config_defaults_init,
    config_defaults_method,
)


def registry_func(
    name: str,
    decorators: Union[Callable, List[Callable], None] = None,
    save_dict: Optional[Dict] = None,
) -> Callable:
    """Return a class decorator that registers *name* in *save_dict*."""
    if save_dict is None:
        save_dict = {}

    def actual_func(obj):
        save_dict[name] = {"obj": obj, "decorators": decorators}
        return obj

    return actual_func


_CORE_MODULES = ["score", "dataset", "loss", "model", "optim", "writer", "scheduler", "task"]

for _module in _CORE_MODULES:
    globals()[f"registered_{_module}"] = dict()
    globals()[f"register_{_module}"] = partial(
        registry_func,
        save_dict=globals()[f"registered_{_module}"],
    )

registered_process: Dict = {}


def get_import_module(import_file: str):
    for mod in sys.modules.copy().values():
        if hasattr(mod, "__file__") and mod.__file__ == import_file:
            return mod
    raise ValueError(f"Cannot find module for file: {import_file!r}")


def register_process(
    name: str,
    decorators: Union[Callable, List[Callable], None] = None,
) -> Callable:
    def actual_func(obj):
        trace_stacks = traceback.extract_stack()
        import_file = trace_stacks[-2][0]
        import_cls_name = trace_stacks[-2][2]
        import_module = get_import_module(import_file)
        registered_process[name] = {
            "cls": {"module": import_module, "name": import_cls_name},
            "obj": obj,
            "decorators": decorators,
        }
        return obj

    return actual_func


def init_registered_module(
    name: str,
    config: Config,
    registered_module: Dict,
    **kwargs,
):
    if name not in registered_module:
        return None
    v = registered_module[name]
    if v["decorators"]:
        return v["decorators"](v["obj"]).from_config(config, **kwargs)
    return v["obj"].from_config(config, **kwargs)


def init_registered_process(
    name: str,
    config: Config,
    **kwargs,
):
    if name not in registered_process:
        return None
    v = registered_process[name]
    cls = getattr(v["cls"]["module"], v["cls"]["name"])
    inst = cls.from_config(config, **kwargs)
    if v["decorators"]:
        return rpartial(v["decorators"](v["obj"]), inst)
    return rpartial(v["obj"], inst)


class GenericScript(abc.ABC):
    def __init__(self, config: Config):
        pass

    @abc.abstractmethod
    def launch(self, **kwargs):
        pass


registered_script: Dict = {}
register_script = partial(registry_func, save_dict=registered_script)


class GenericService(abc.ABC):
    def __init__(self, config: Config):
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


registered_service: Dict = {}
register_service = partial(registry_func, save_dict=registered_service)


class GenericFastAPI(abc.ABC):
    def __init__(self, config: Config):
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


registered_fastapi: Dict = {}
register_fastapi = partial(registry_func, save_dict=registered_fastapi)


class GenericCopilotTool(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def launch(self, **kwargs):
        pass

    @abc.abstractmethod
    def describe(self):
        pass


registered_copilot_tools: Dict = {}
register_copilot_tool = partial(registry_func, save_dict=registered_copilot_tools)


from unitorch.cli.writers import WriterMixin, WriterOutputs

import unitorch.cli.datasets
import unitorch.cli.losses
import unitorch.cli.models
import unitorch.cli.optims
import unitorch.cli.schedulers
import unitorch.cli.scores
import unitorch.cli.tasks
import unitorch.cli.writers
