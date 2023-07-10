# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import logging
import importlib
import unitorch.cli
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    set_global_config,
    init_registered_module,
)
