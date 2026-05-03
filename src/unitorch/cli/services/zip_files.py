# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import http.server
import zipfile
from urllib.parse import parse_qs, urlparse
from threading import Thread
from functools import lru_cache
from unitorch.cli import Config
from unitorch.cli import register_service, GenericService


def get_zipfile(zfs, res, idx, step):
    """Extracts a subset of zip files into res, stepping by step."""
    for i in range(idx, len(zfs), step):
        res[i] = zipfile.ZipFile(zfs[i])


def get_zipfiles(zipfiles, num_thread=48):
    """Opens multiple zip files concurrently using threads."""
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


def parse_params(path):
    """Parses query parameters from a URL path."""
    url_components = urlparse(path)
    query_params = parse_qs(url_components.query)
    return {k: v[0] for k, v in query_params.items()}


class ZipFilesServer(http.server.BaseHTTPRequestHandler):
    zip_data = None
    zip_dict = dict()
    none_resp = "".encode("utf-8")

    @lru_cache(maxsize=10000)
    def _get_file(self, file):
        """Retrieves file bytes from the indexed zip archives."""
        zf = self.zip_dict.get(file)
        if zf is None:
            logging.warning(f"File {file} not found.")
            return self.none_resp
        zf = self.zip_data[zf]
        if zf is None:
            logging.warning(f"File {file} not found.")
            return self.none_resp
        file = zf.read(file)
        return file

    def do_GET(self):
        if self.path == "/all-files":
            self.send_response(200)
            self.end_headers()
            names = list(self.zip_dict.keys())
            self.wfile.write("\n".join(names).encode("utf-8"))
        else:
            self.send_response(200)
            self.end_headers()
            params = parse_params(self.path)
            file = params.get("file")
            resp = self.none_resp
            if file is not None:
                resp = self._get_file(file)
            self.wfile.write(resp)

    def log_request(self, format, *args):
        return

    def log_message(self, format, *args):
        return


@register_service("core/service/zip_files")
class ZipFilesService(GenericService):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/service/zip_files")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11230)
        self.name = config.getoption("processname", "core_zip_files_service")
        self.zip_folder = config.getoption("zip_folder", None)
        self.zip_extension = config.getoption("zip_extension", ".zip")
        self.num_thread = config.getoption("num_thread", 20)
        assert self.zip_folder is not None

        if isinstance(self.zip_folder, str):
            self.zip_folder = os.path.abspath(self.zip_folder)
            zip_files = [
                os.path.join(self.zip_folder, f)
                for f in os.listdir(self.zip_folder)
                if f.endswith(self.zip_extension)
            ]
        elif isinstance(self.zip_folder, list):
            self.zip_folder = [os.path.abspath(f) for f in self.zip_folder]
            zip_files = []
            for folder in list(set(self.zip_folder)):
                zip_files += [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(self.zip_extension)
                ]

        self.zip_data = get_zipfiles(zip_files, self.num_thread)
        self.zip_dict = {
            v: i for i, k in enumerate(self.zip_data) for v in k.namelist()
        }

    def start(self, **kwargs):
        ZipFilesServer.zip_data = self.zip_data
        ZipFilesServer.zip_dict = self.zip_dict
        self.httpd = http.server.HTTPServer((self.ip, self.port), ZipFilesServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
