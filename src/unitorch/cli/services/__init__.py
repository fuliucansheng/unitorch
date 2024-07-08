# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import importlib
from functools import partial
from unitorch.cli import GenericService, registered_service, register_service


# import service modules
import unitorch.cli.services.http_file
import unitorch.cli.services.zip_image
