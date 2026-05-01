# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import asyncio
import psutil
import torch
from fastapi import APIRouter
from unitorch.cli import register_fastapi
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
            if isinstance(self._device, list):
                for device in self._device:
                    free, total = torch.cuda.mem_get_info(device)
                    total = total / 1024**3
                    free = free / 1024**3
                    used = total - free
                    stats = {
                        **stats,
                        **{
                            f"cuda:{device}": {
                                "total": total,
                                "free": free,
                                "used": used,
                            }
                        },
                    }
            else:
                free, total = torch.cuda.mem_get_info(self._device)
                total = total / 1024**3
                free = free / 1024**3
                used = total - free
                stats = {
                    **stats,
                    **{"cuda": {"total": total, "free": free, "used": used}},
                }
        return stats
