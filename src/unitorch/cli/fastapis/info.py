# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import re
import gc
import json
import logging
import psutil
import torch
import hashlib
import asyncio
import socket
import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI


@register_fastapi("core/fastapi/info")
class InfoFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/info")
        router = config.getoption("router", "/core/fastapi/info")
        self._device = config.getoption("device", "cpu")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        return "start success"

    def stop(self):
        return "stop success"

    def status(self):
        mem_info = psutil.virtual_memory()
        stats = {
            "cpu": {
                "total": mem_info[0] / 1024**3,
                "free": mem_info[1] / 1024**3,
                "used": mem_info[3] / 1024**3,
            }
        }
        if self._device != "cpu":
            free, total = torch.cuda.mem_get_info(self._device)
            total = total / 1024**3
            free = free / 1024**3
            used = total - free
            stats = {**stats, **{"cuda": {"total": total, "free": free, "used": used}}}
        return stats
