# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import atexit
import asyncio
import logging
import os
import uuid
import zipfile
from threading import Thread

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from unitorch.cli import Config, GenericFastAPI, register_fastapi


def get_zipfile(zfs, res, idx, step):
    for i in range(idx, len(zfs), step):
        try:
            zf = zipfile.ZipFile(zfs[i])
            res[i] = zf.namelist()
        except Exception as e:
            logging.error("Failed to extract %s: %s", zfs[i], e)
            os.remove(zfs[i])
            res[i] = None


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


@register_fastapi("core/fastapi/servers/zip_saver")
class ZipSaverFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        self.config.set_default_section("core/fastapi/servers/zip_saver")
        router = config.getoption("router", "/core/fastapi/servers/zip_saver")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/", self.save_file, methods=["POST"])
        self._router.add_api_route("/save", self.save_file, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])

        self.zip_folder = config.getoption("zip_folder", None)
        self.random_name = config.getoption("random_name", False)
        self.zip_file_prefix = config.getoption("zip_file_prefix", "zip_files")
        self.zip_extension = config.getoption("zip_extension", ".zip")
        self.max_files_per_zip = config.getoption("max_files_per_zip", 10000000)
        self.num_thread = config.getoption("num_thread", 20)
        assert self.zip_folder is not None, "zip_folder must be provided"

        self.zip_folder = os.path.abspath(self.zip_folder)
        os.makedirs(self.zip_folder, exist_ok=True)

        self._lock = asyncio.Lock()
        self._zip_set = set()
        self._curr_file = None
        self._curr_file_path = None
        self._num_files_in_zip = 0
        self._next_zip_file_index = 0

        self._init_existing_state()

        atexit.register(self.stop)

    @property
    def router(self):
        return self._router

    def _collect_zip_files(self):
        return [
            os.path.join(self.zip_folder, f)
            for f in os.listdir(self.zip_folder)
            if f.endswith(self.zip_extension)
        ]

    def _parse_zip_index(self, filename: str):
        try:
            return int(filename.rsplit("_", 1)[-1].split(".", 1)[0])
        except Exception:
            return -1

    def _make_zip_filename(self):
        suffix = (
            f"_{uuid.uuid4().hex[:8]}_{self._next_zip_file_index}"
            if self.random_name
            else f"_{self._next_zip_file_index}"
        )
        extension = self.zip_extension or ".zip"
        if not str(extension).startswith("."):
            extension = f".{extension}"
        return os.path.join(
            self.zip_folder,
            f"{self.zip_file_prefix}{suffix}{extension}",
        )

    def _init_existing_state(self):
        zip_files = self._collect_zip_files()
        zip_data = get_zipfiles(zip_files, self.num_thread)
        self._zip_set = set()
        for data in zip_data:
            if data is None:
                continue
            self._zip_set.update(data)
        self._next_zip_file_index = (
            max(
                [
                    self._parse_zip_index(f)
                    for f in os.listdir(self.zip_folder)
                    if f.startswith(self.zip_file_prefix)
                ],
                default=-1,
            )
            + 1
        )

    def _create_new_file(self):
        if self._curr_file is not None:
            self._curr_file.close()
        self._num_files_in_zip = 0
        self._curr_file_path = self._make_zip_filename()
        self._curr_file = zipfile.ZipFile(self._curr_file_path, "w")
        self._next_zip_file_index += 1

    def start(self):
        if self._curr_file is None:
            self._create_new_file()
        return "start success"

    def stop(self):
        if self._curr_file is not None:
            self._curr_file.close()
            self._curr_file = None
        return "stop success"

    def status(self):
        return {
            "status": "running" if self._curr_file is not None else "stopped",
            "zip_folder": self.zip_folder,
            "current_zip_file": self._curr_file_path,
            "num_saved_files": len(self._zip_set),
            "num_files_in_current_zip": self._num_files_in_zip,
            "next_zip_file_index": self._next_zip_file_index,
        }

    async def save_file(
        self,
        name: str = Query(default=None),
        file: UploadFile = File(...),
    ):
        if name is None or name.endswith("/"):
            raise HTTPException(status_code=400, detail="invalid name")

        async with self._lock:
            if name in self._zip_set:
                return {"status": "exists", "name": name}

            if self._curr_file is None:
                self.start()
            if self._num_files_in_zip >= self.max_files_per_zip:
                self._create_new_file()

            content = await file.read()
            self._curr_file.writestr(name, content)
            if getattr(self._curr_file, "fp", None) is not None:
                self._curr_file.fp.flush()
            self._num_files_in_zip += 1
            self._zip_set.add(name)

        return {
            "status": "saved",
            "name": name,
            "zip_file": self._curr_file_path,
        }
