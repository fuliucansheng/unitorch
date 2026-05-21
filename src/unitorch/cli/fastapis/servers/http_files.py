# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import html
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from unitorch.cli import Config, GenericFastAPI, register_fastapi


@register_fastapi("core/fastapi/servers/http_files")
class HttpFilesFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        self.config.set_default_section("core/fastapi/servers/http_files")
        router = config.getoption("router", "/core/fastapi/servers/http_files")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/", self.serve_root, methods=["GET"])
        self._router.add_api_route(
            "/{file_path:path}", self.serve_path, methods=["GET"]
        )

        html_dir = config.getoption("html_dir", None)
        assert html_dir is not None, "html_dir must be provided"
        self.html_dir = Path(os.path.abspath(html_dir)).resolve()

    @property
    def router(self):
        return self._router

    def _resolve_path(self, file_path: str):
        requested_path = (self.html_dir / file_path.lstrip("/")).resolve()
        try:
            requested_path.relative_to(self.html_dir)
        except ValueError as e:
            raise HTTPException(status_code=404, detail="file not found") from e
        return requested_path

    def _directory_listing(self, directory: Path, file_path: str):
        entries = []
        if file_path:
            entries.append('<li><a href="../">../</a></li>')

        for child in sorted(
            directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        ):
            name = child.name + ("/" if child.is_dir() else "")
            href = quote(name)
            entries.append(f'<li><a href="{href}">{html.escape(name)}</a></li>')

        title = "/" if not file_path else f"/{file_path.strip('/')}/"
        page = "\n".join(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                '  <meta charset="utf-8">',
                f"  <title>Index of {html.escape(title)}</title>",
                "</head>",
                "<body>",
                f"  <h1>Index of {html.escape(title)}</h1>",
                "  <ul>",
                *entries,
                "  </ul>",
                "</body>",
                "</html>",
            ]
        )
        return HTMLResponse(content=page)

    def start(self):
        return "start success"

    def stop(self):
        return "stop success"

    def status(self):
        return {
            "status": "running",
            "html_dir": str(self.html_dir),
        }

    def serve_root(self):
        return self.serve_path("")

    def serve_path(self, file_path: str):
        file_path = file_path.lstrip("/")
        requested_path = self._resolve_path(file_path)
        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="file not found")

        if requested_path.is_dir():
            if file_path and not file_path.endswith("/"):
                return RedirectResponse(url=f"{file_path}/")
            index_file = requested_path / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return self._directory_listing(requested_path, file_path)

        return FileResponse(requested_path)
