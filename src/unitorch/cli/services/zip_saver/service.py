# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import logging
import http.server
import zipfile
import tempfile
import cgi
from threading import Thread
from urllib.parse import parse_qs, urlparse
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
        try:
            zf = zipfile.ZipFile(zfs[i])
            res[i] = zf.namelist()
        except Exception as e:
            logging.error(f"Failed to extract {zfs[i]}: {e}")
            os.remove(zfs[i])
            res[i] = None
        finally:
            pass


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


class ZipSaverServer(http.server.BaseHTTPRequestHandler):
    zip_set = {}
    zip_file_prefix = "zip_images"
    zip_folder = None
    curr_file = None
    next_zip_file_index = 0
    num_files_in_zip = 0
    max_files_per_zip = 10000

    @classmethod
    def create_new_file(cls):
        if cls.curr_file is not None:
            cls.curr_file.close()
        cls.num_files_in_zip = 0
        name = os.path.join(
            cls.zip_folder, f"{cls.zip_file_prefix}_{cls.next_zip_file_index}.zip"
        )
        cls.curr_file = zipfile.ZipFile(name, "w")
        cls.next_zip_file_index += 1

    def do_POST(self):
        content_type, pdict = cgi.parse_header(self.headers["Content-Type"])
        name = parse_qs(urlparse(self.path).query).get("name", [None])[0]
        if content_type == "multipart/form-data":
            pdict["boundary"] = bytes(pdict["boundary"], "utf-8")
            fields = cgi.parse_multipart(self.rfile, pdict)
            if "file" in fields:
                content = fields["file"][0]
                if name in ZipSaverServer.zip_set:
                    self.send_response(200)
                    self.end_headers()
                else:
                    if (
                        ZipSaverServer.num_files_in_zip
                        >= ZipSaverServer.max_files_per_zip
                    ):
                        ZipSaverServer.create_new_file()
                    ZipSaverServer.curr_file.writestr(name, content)
                    ZipSaverServer.num_files_in_zip += 1
                    ZipSaverServer.zip_set.add(name)
                    self.send_response(200)
                    self.send_header("Content-type", "application/zip")
                    self.end_headers()
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(400)
            self.end_headers()


@register_service("core/service/zip_saver")
class ZipSaverService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/zip_saver")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11231)
        self.name = config.getoption("processname", "core_zip_saver_service")
        self.zip_folder = config.getoption("zip_folder", None)
        self.zip_file_prefix = config.getoption("zip_file_prefix", "zip_images")
        self.zip_extension = config.getoption("zip_extension", ".zip")
        self.max_files_per_zip = config.getoption("max_files_per_zip", 10000000)
        assert self.zip_folder is not None
        if not os.path.exists(self.zip_folder):
            os.makedirs(self.zip_folder)

        self.num_thread = config.getoption("num_thread", 20)
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

        zip_data = get_zipfiles(zip_files, self.num_thread)
        self.zip_set = set()
        for data in zip_data:
            if data is None:
                continue
            self.zip_set.update(data)
        self.next_zip_file_index = (
            max(
                [
                    int(f.split("_")[-1].split(".")[0])
                    for f in os.listdir(self.zip_folder)
                    if f.startswith(self.zip_file_prefix)
                ],
                default=-1,
            )
            + 1
        )

    def start(self, **kwargs):
        ZipSaverServer.zip_folder = self.zip_folder
        ZipSaverServer.zip_set = self.zip_set
        ZipSaverServer.max_files_per_zip = self.max_files_per_zip
        ZipSaverServer.zip_file_prefix = self.zip_file_prefix
        ZipSaverServer.next_zip_file_index = self.next_zip_file_index + 1
        name = os.path.join(
            self.zip_folder, f"{self.zip_file_prefix}_{self.next_zip_file_index}.zip"
        )
        ZipSaverServer.curr_file = name
        ZipSaverServer.curr_file = zipfile.ZipFile(name, "w")
        self.httpd = http.server.HTTPServer((self.ip, self.port), ZipSaverServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        ZipSaverServer.curr_file.close()

    def restart(self, **kwargs):
        pass
