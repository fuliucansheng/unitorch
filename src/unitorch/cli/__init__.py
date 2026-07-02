# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import sys
import abc
import base64
import dataclasses
import logging
import traceback
import inspect
import importlib
import importlib_resources
import importlib.metadata as importlib_metadata
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)

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
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    use_auth_token: Union[bool, str, None] = None,
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
        use_auth_token=use_auth_token,
    )


from unitorch.cli.decorators import (
    config_defaults_init,
    config_defaults_method,
)
from unitorch.cli.replicas import PipelineReplicaLease, PipelineReplicaPool


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


_CORE_MODULES = [
    "score",
    "dataset",
    "loss",
    "model",
    "optim",
    "writer",
    "scheduler",
    "task",
]

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


class GenericFastAPI(abc.ABC):
    def __init__(self, config: Config):
        pass

    def pipeline_pool(
        self,
        pipelines: Any = None,
        section: Optional[str] = None,
        lock: Optional[bool] = None,
    ) -> PipelineReplicaPool:
        config = getattr(self, "config", None)
        if config is None:
            raise ValueError("FastAPI service must define self.config")
        section = section or getattr(self, "_section", "core/cli")
        return PipelineReplicaPool.from_config(
            config=config,
            pipelines=pipelines,
            section=section,
            lock=lock,
        )

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


@dataclass(frozen=True)
class GenericCopilotRemoteSpec:
    route: str
    method: str = "POST"
    param_fields: Dict[str, str] = field(default_factory=dict)
    file_fields: Dict[str, str] = field(default_factory=dict)
    binary: bool = False
    image_fields: Dict[str, str] = field(default_factory=dict)
    video_fields: Dict[str, str] = field(default_factory=dict)
    req_type: Optional[str] = None
    resp_type: str = "json"

    def __post_init__(self):
        method = self.req_type or self.method
        method = method.upper()
        resp_type = self.resp_type.lower()
        if resp_type not in {"json", "image", "video"}:
            raise ValueError(f"Unsupported copilot response type: {self.resp_type}")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "req_type", method)
        object.__setattr__(self, "resp_type", resp_type)


def serialize_copilot_output(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return serialize_copilot_output(dataclasses.asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, dict):
        return {str(k): serialize_copilot_output(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [serialize_copilot_output(v) for v in value]

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()

    if hasattr(value, "tolist"):
        return value.tolist()

    if (
        hasattr(value, "mode")
        and hasattr(value, "size")
        and value.__class__.__name__ == "Image"
    ):
        return {
            "type": "image",
            "mode": value.mode,
            "size": list(value.size),
        }

    return value


@dataclass(frozen=True)
class GenericCopilotTool:
    name: str
    func: Callable
    description: str = ""
    tags: Tuple[str, ...] = tuple()
    remote: Optional[GenericCopilotRemoteSpec] = None
    python_module: str = "unitorch.cli.copilots"
    python_function: Optional[str] = None

    def __post_init__(self):
        if self.python_function is None:
            object.__setattr__(self, "python_function", self.func.__name__)

    @property
    def signature(self):
        return inspect.signature(self.func)

    @property
    def type_hints(self):
        try:
            return get_type_hints(self.func)
        except Exception:
            return {}

    def invoke(self, **kwargs):
        return serialize_copilot_output(self.func(**kwargs))


def _copilot_remote_spec(
    remote: Optional[Union[GenericCopilotRemoteSpec, Dict]],
) -> Optional[GenericCopilotRemoteSpec]:
    if remote is None or isinstance(remote, GenericCopilotRemoteSpec):
        return remote
    return GenericCopilotRemoteSpec(**remote)


registered_copilot_tool: Dict[str, GenericCopilotTool] = {}


def register_copilot_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    remote: Optional[Union[GenericCopilotRemoteSpec, Dict]] = None,
    python_function: Optional[str] = None,
    decorators: Union[Callable, List[Callable], None] = None,
) -> Callable:
    def actual_func(obj: Callable) -> Callable:
        copilot_name = name or obj.__name__
        func = obj
        obj_description = description or inspect.getdoc(obj) or ""
        obj_python_function = python_function or obj.__name__

        registered_copilot_tool[copilot_name] = GenericCopilotTool(
            name=copilot_name,
            func=func,
            description=obj_description,
            tags=tuple(tags or ()),
            remote=_copilot_remote_spec(remote),
            python_function=obj_python_function,
        )
        return obj

    return actual_func


from unitorch.cli.writers import WriterMixin, WriterOutputs

import unitorch.cli.datasets
import unitorch.cli.losses
import unitorch.cli.models
import unitorch.cli.optims
import unitorch.cli.schedulers
import unitorch.cli.scores
import unitorch.cli.tasks
import unitorch.cli.writers
