# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import random
import tempfile
import warnings

import numpy as np
import sklearn
import torch
import torch.multiprocessing
import transformers

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

VERSION = "0.0.2.0"

# ---------------------------------------------------------------------------
# Environment – tokenizers
# ---------------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------

_HOME = os.getenv("HOME", ".")

UNITORCH_CACHE: str = os.getenv(
    "UNITORCH_CACHE", os.path.join(_HOME, ".cache/unitorch")
)
os.environ["TRANSFORMERS_CACHE"] = UNITORCH_CACHE
os.environ["HF_HOME"] = UNITORCH_CACHE
os.environ["HF_DATASETS_CACHE"] = UNITORCH_CACHE

UNITORCH_TEMP: str = os.getenv(
    "UNITORCH_TEMP", os.path.join(tempfile.gettempdir(), "unitorch")
)
os.makedirs(UNITORCH_TEMP, exist_ok=True)

UNITORCH_HOME: str = os.getenv("UNITORCH_HOME", os.path.join(_HOME, ".unitorch"))
os.makedirs(UNITORCH_HOME, exist_ok=True)


def get_cache_dir() -> str:
    """Return the unitorch cache directory."""
    return UNITORCH_CACHE


def get_temp_dir() -> str:
    """Return the unitorch temporary directory."""
    return UNITORCH_TEMP


def get_dir() -> str:
    """Return the unitorch home directory."""
    return UNITORCH_HOME


def mktempdir(prefix: str = "") -> str:
    """Create and return a new temporary directory inside ``UNITORCH_TEMP``."""
    return tempfile.mkdtemp(prefix=prefix, dir=UNITORCH_TEMP)


def mktempfile(prefix: str = "", suffix: str = "") -> tuple:
    """Create and return a new temporary file inside ``UNITORCH_TEMP``."""
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=UNITORCH_TEMP)


# ---------------------------------------------------------------------------
# Logging & warnings
# ---------------------------------------------------------------------------

_DEBUG_LEVEL_MAP: dict[str, int] = {
    "OFF": 50,
    "INFO": 20,
    "DETAIL": 10,
    "CPU": 10,
    "ALL": 0,
}

UNITORCH_DEBUG: str = os.environ.get("UNITORCH_DEBUG", "INFO").upper()
if UNITORCH_DEBUG not in _DEBUG_LEVEL_MAP:
    UNITORCH_DEBUG = "INFO"

if UNITORCH_DEBUG == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "999"

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=_DEBUG_LEVEL_MAP[UNITORCH_DEBUG],
)

# ---------------------------------------------------------------------------
# PyTorch global settings
# ---------------------------------------------------------------------------

np.complex = complex  # legacy alias required by some downstream libraries

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility across ``random``, ``numpy``, and ``torch``.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

from unitorch.utils import cached_path as hf_cached_path, read_file, read_json_file
from unitorch.utils import (
    is_deepspeed_available,
    is_diffusers_available,
    is_megatron_available,
    is_opencv_available,
    is_wandb_available,
)

if is_diffusers_available():
    import diffusers

    diffusers.logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Sub-package imports
# ---------------------------------------------------------------------------

import unitorch.datasets
import unitorch.losses
import unitorch.models
import unitorch.modules
import unitorch.optims
import unitorch.schedulers
import unitorch.scores
import unitorch.tasks
