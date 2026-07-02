# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from __future__ import annotations

import atexit
import io
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image

from unitorch.cli.copilots import get_copilot_tool


def _clean_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if v is not None}


def _process_image(image: Any) -> io.BytesIO:
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    else:
        image = image.convert("RGB")

    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")
    byte_arr.seek(0)
    return byte_arr


def _process_video(video: Any) -> tuple:
    filename = "video.mp4"
    content_type = "video/mp4"

    if isinstance(video, tuple):
        return video

    if isinstance(video, (str, os.PathLike)):
        path = os.fspath(video)
        filename = os.path.basename(path)
        content_type = mimetypes.guess_type(filename)[0] or content_type
        with open(path, "rb") as f:
            data = f.read()
    elif isinstance(video, bytes):
        data = video
    elif hasattr(video, "read"):
        filename = os.path.basename(getattr(video, "name", filename))
        content_type = mimetypes.guess_type(filename)[0] or content_type
        data = video.read()
        if hasattr(video, "seek"):
            video.seek(0)
    else:
        raise TypeError(
            "Video inputs must be file paths, bytes, file-like objects, "
            "or requests file tuples."
        )

    byte_arr = io.BytesIO(data)
    byte_arr.seek(0)
    return filename, byte_arr, content_type


def call_fastapi(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    images: Optional[Dict[str, Any]] = None,
    videos: Optional[Dict[str, Any]] = None,
    req_type: str = "POST",
    resp_type: str = "json",
    timeout: Optional[float] = 60,
):
    req_type = req_type.upper()
    resp_type = resp_type.lower()
    assert resp_type in ["json", "image", "video"], (
        f"Unsupported response type: {resp_type}"
    )
    params = _clean_params(params)
    files = {}

    if images is not None:
        files.update(
            {
                k: (f"{k}.jpg", _process_image(v), "image/jpeg")
                for k, v in images.items()
                if v is not None
            }
        )
    if videos is not None:
        files.update(
            {k: _process_video(v) for k, v in videos.items() if v is not None}
        )

    try:
        if req_type == "POST" or files:
            resp = (
                requests.post(url, params=params, files=files, timeout=timeout)
                if files
                else requests.post(url, params=params, timeout=timeout)
            )
        else:
            resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()

        if resp_type == "json":
            return resp.json()
        if resp_type == "image":
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return resp.content
    except requests.RequestException as e:
        response = getattr(e, "response", None)
        detail = getattr(response, "text", None) if response is not None else None
        message = f"Remote copilot request failed: {e}"
        if detail:
            message = f"{message}\n{detail}"
        raise RuntimeError(message) from e
    finally:
        for file_tuple in files.values():
            file_obj = (
                file_tuple[1]
                if isinstance(file_tuple, tuple) and len(file_tuple) > 1
                else None
            )
            if isinstance(file_obj, io.BytesIO):
                file_obj.close()


def _local_host(host: str) -> str:
    if host in ("", "0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _terminate_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _find_available_port(host: str = "127.0.0.1") -> int:
    bind_host = _local_host(host).strip("[]")
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


class CopilotClient:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 60,
        config: Optional[str] = None,
        host: str = "127.0.0.1",
        ip: Optional[str] = None,
        port: Optional[int] = None,
        startup_timeout: float = 60,
    ):
        self.timeout = timeout
        self.config = config
        self.host = ip or host
        if port is not None:
            self.port = int(port)
        elif endpoint is None and self.config is not None:
            self.port = _find_available_port(self.host)
        else:
            self.port = 5000
        self.startup_timeout = startup_timeout
        self.process = None
        self.endpoint = (
            endpoint.rstrip("/")
            if endpoint is not None
            else f"http://{_local_host(self.host)}:{self.port}"
        )
        if self.config is not None:
            self.start()

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return self
        if self.config is None:
            return self

        unitorch_fastapi_cmd = shutil.which("unitorch-fastapi") or os.path.join(
            os.path.dirname(sys.executable),
            "unitorch-fastapi",
        )
        self.process = subprocess.Popen(
            [
                unitorch_fastapi_cmd,
                self.config,
                f"--host={self.host}",
                f"--port={self.port}",
            ],
            stdin=subprocess.DEVNULL,
        )
        atexit.register(self.stop)
        self._wait_until_ready()
        return self

    def stop(self):
        _terminate_process(self.process)
        self.process = None

    def _wait_until_ready(self):
        deadline = time.time() + float(self.startup_timeout)
        url = f"{self.endpoint}/health-check"
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("unitorch-fastapi exited before health-check passed")
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"unitorch-fastapi health-check timeout: {url}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def request(
        self,
        route: str,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, Any]] = None,
        videos: Optional[Dict[str, Any]] = None,
        resp_type: str = "json",
    ):
        route = route if route.startswith("/") else f"/{route}"
        return call_fastapi(
            url=f"{self.endpoint}{route}",
            params=params,
            images=images,
            videos=videos,
            req_type=method,
            resp_type=resp_type,
            timeout=self.timeout,
        )

    def invoke(self, name: str, **kwargs):
        copilot_tool = get_copilot_tool(name)
        if copilot_tool.remote is None:
            raise RuntimeError(f"Copilot tool {name!r} has no remote adapter.")

        remote = copilot_tool.remote
        params = dict(kwargs)
        images = {}
        videos = {}

        for public_name, remote_name in remote.param_fields.items():
            if public_name in params:
                params[remote_name] = params.pop(public_name)
        for public_name, remote_name in remote.file_fields.items():
            if public_name in params:
                images[remote_name] = params.pop(public_name)
        for public_name, remote_name in remote.image_fields.items():
            if public_name in params:
                images[remote_name] = params.pop(public_name)
        for public_name, remote_name in remote.video_fields.items():
            if public_name in params:
                videos[remote_name] = params.pop(public_name)

        resp_type = remote.resp_type
        if remote.binary and resp_type == "json":
            resp_type = "video"

        return self.request(
            remote.route,
            method=remote.method,
            params=params,
            images=images or None,
            videos=videos or None,
            resp_type=resp_type,
        )
