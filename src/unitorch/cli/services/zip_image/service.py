# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import logging
import http.server
import zipfile
from urllib.parse import parse_qs, urlparse
from PIL import Image
from threading import Thread
from functools import lru_cache
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_service, GenericService


def get_zipfile(zfs, res, idx, step):
    """
    Extracts a subset of zip files from the given list.

    Args:
        zfs (list): List of zip files.
        res (list): Result list to store extracted zip files.
        idx (int): Starting index.
        step (int): Step size.

    Returns:
        None
    """
    for i in range(idx, len(zfs), step):
        res[i] = zipfile.ZipFile(zfs[i])


def get_zipfiles(zipfiles, num_thread=48):
    """
    Extracts multiple zip files using multiple threads.

    Args:
        zipfiles (list): List of zip file paths.
        num_thread (int): Number of threads to use.

    Returns:
        list: Extracted zip files.
    """
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
    """
    Parses query parameters from the URL path.

    Args:
        path (str): URL path.

    Returns:
        dict: Parsed query parameters.
    """
    url_components = urlparse(path)
    query_params = parse_qs(url_components.query)
    return {k: v[0] for k, v in query_params.items()}


class ImageServer(http.server.BaseHTTPRequestHandler):
    """
    HTTP request handler class.
    """

    zip_data = None
    zip_dict = dict()
    none_resp = "".encode("utf-8")

    @lru_cache(maxsize=10000)
    def _get_image(self, image):
        """
        Retrieves the image data from zip files.

        Args:
            image (str): Image filename.

        Returns:
            bytes: Image data.
        """
        zf = self.zip_dict.get(image)
        if zf is None:
            return self.none_resp
        zf = self.zip_data[zf]
        if zf is None:
            return self.none_resp
        image = zf.read(image)
        return image

    def do_GET(self):
        """
        Handles GET requests.

        Returns:
            None
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        params = parse_params(self.path)
        image = params.get("image")
        resp = self.none_resp
        if image is not None:
            resp = self._get_image(image)
        self.wfile.write(resp)

    def log_request(self, format, *args):
        """
        Logs the HTTP request.

        Returns:
            None
        """
        return

    def log_message(self, format, *args):
        """
        Logs the message.

        Returns:
            None
        """
        return


@register_service("core/service/zip_image")
class ZipImageService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/zip_image")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11230)
        self.name = config.getoption("processname", "core_zip_image_service")
        self.zip_folder = config.getoption("zip_folder", None)
        self.zip_extension = config.getoption("zip_extension", ".zip")
        self.num_thread = config.getoption("num_thread", 20)
        assert self.zip_folder is not None
        if isinstance(self.zip_folder, str):
            zip_files = [
                os.path.join(self.zip_folder, f)
                for f in os.listdir(self.zip_folder)
                if f.endswith(self.zip_extension)
            ]
        elif isinstance(self.zip_folder, list):
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
        ImageServer.zip_data = self.zip_data
        ImageServer.zip_dict = self.zip_dict
        self.httpd = http.server.HTTPServer((self.ip, self.port), ImageServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
