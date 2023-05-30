# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random

from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)


class RandomProcessor:
    def __init__(
        self,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/process/random")
    def from_core_configure(cls, config, **kwargs):
        pass
