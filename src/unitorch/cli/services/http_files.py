# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import logging
import http.server
import zipfile
from urllib.parse import parse_qs, urlparse
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_service, GenericService


class HttpFilesServer(http.server.SimpleHTTPRequestHandler):
    html_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.html_dir, **kwargs)


@register_service("core/service/http_files")
class HttpFilesService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/http_files")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11220)
        self.name = config.getoption("processname", "core_http_files_service")
        self.html_dir = config.getoption("html_dir", None)
        assert self.html_dir is not None, "html_dir must be provided"

    def start(self, **kwargs):
        HttpFilesServer.html_dir = self.html_dir
        self.httpd = http.server.HTTPServer((self.ip, self.port), HttpFilesServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
