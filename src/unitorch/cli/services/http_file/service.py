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


class HttpFileServer(http.server.SimpleHTTPRequestHandler):
    html_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.html_dir, **kwargs)


@register_service("core/service/http_file")
class HttpFileService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/http_file")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 8000)
        self.name = config.getoption("processname", "core_http_file_service")
        self.html_dir = config.getoption("html_dir", None)
        assert self.html_dir is not None, "html_dir must be provided"

    def start(self, **kwargs):
        HttpFileServer.html_dir = self.html_dir
        self.httpd = http.server.HTTPServer((self.ip, self.port), HttpFileServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
