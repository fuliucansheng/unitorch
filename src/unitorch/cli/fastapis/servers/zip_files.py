# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import atexit
import logging
import mimetypes
import os
import zipfile
from functools import lru_cache
from threading import Thread

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse, Response

from unitorch.cli import Config, GenericFastAPI, register_fastapi


def get_zipfile(zfs, res, idx, step):
    for i in range(idx, len(zfs), step):
        res[i] = zipfile.ZipFile(zfs[i])


def get_zipfiles(zipfiles, num_thread=48):
    num_thread = min(len(zipfiles), num_thread)
    threads = [None] * num_thread
    results = [None] * len(zipfiles)
    for i in range(len(threads)):
        threads[i] = Thread(
            target=get_zipfile,
            args=(zipfiles, results, i, num_thread),
        )
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()
    return results


@register_fastapi("core/fastapi/servers/zip_files")
class ZipFilesFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        self.config.set_default_section("core/fastapi/servers/zip_files")
        router = config.getoption("router", "/core/fastapi/servers/zip_files")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/", self.get_file, methods=["GET"])
        self._router.add_api_route("/all-files", self.all_files, methods=["GET"])
        self._router.add_api_route("/status", self.status, methods=["GET"])

        self.zip_folder = config.getoption("zip_folder", None)
        self.zip_extension = config.getoption("zip_extension", ".zip")
        self.num_thread = config.getoption("num_thread", 20)
        self._none_resp = b""
        self._zip_data = []
        self._zip_dict = {}
        self._started = False
        assert self.zip_folder is not None, "zip_folder must be provided"

        atexit.register(self.stop)

    @property
    def router(self):
        return self._router

    def _collect_zip_files(self):
        if isinstance(self.zip_folder, str):
            zip_folder = os.path.abspath(self.zip_folder)
            return [
                os.path.join(zip_folder, f)
                for f in os.listdir(zip_folder)
                if f.endswith(self.zip_extension)
            ]

        if isinstance(self.zip_folder, list):
            zip_folders = [os.path.abspath(f) for f in self.zip_folder]
            zip_files = []
            for folder in list(set(zip_folders)):
                zip_files += [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(self.zip_extension)
                ]
            return zip_files

        raise ValueError("zip_folder must be a string or a list of strings")

    def _close_zipfiles(self):
        for zf in self._zip_data:
            if zf is None:
                continue
            try:
                zf.close()
            except Exception:
                pass

    @lru_cache(maxsize=10000)
    def _get_file(self, file_name: str):
        zf_index = self._zip_dict.get(file_name)
        if zf_index is None:
            logging.warning("File %s not found.", file_name)
            return self._none_resp
        zf = self._zip_data[zf_index]
        if zf is None:
            logging.warning("File %s not found.", file_name)
            return self._none_resp
        return zf.read(file_name)

    def start(self):
        self.stop()
        zip_files = self._collect_zip_files()
        self._zip_data = get_zipfiles(zip_files, self.num_thread)
        self._zip_dict = {
            name: i
            for i, zf in enumerate(self._zip_data)
            for name in zf.namelist()
            if not name.endswith("/")
        }
        self._get_file.cache_clear()
        self._started = True
        return "start success"

    def stop(self):
        self._get_file.cache_clear()
        self._close_zipfiles()
        self._zip_data = []
        self._zip_dict = {}
        self._started = False
        return "stop success"

    def status(self):
        return {
            "status": "running" if self._started else "stopped",
            "num_archives": len(self._zip_data),
            "num_files": len(self._zip_dict),
        }

    def all_files(self):
        return PlainTextResponse("\n".join(self._zip_dict.keys()))

    def get_file(self, file: str = Query(default=None)):
        if file is None or file.endswith("/"):
            return Response(status_code=400, content=self._none_resp)
        content = self._get_file(file)
        filename = os.path.basename(file)
        media_type, _ = mimetypes.guess_type(filename)
        if media_type is None:
            media_type = "application/octet-stream"
        headers = {"Content-Disposition": f"inline; filename={filename}"}
        return Response(
            content=content,
            media_type=media_type,
            headers=headers,
        )
