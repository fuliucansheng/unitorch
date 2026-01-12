# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import tempfile

# env setting
os.environ["TOKENIZERS_PARALLELISM"] = "false"
### cache setting
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
UNITORCH_CACHE = os.getenv(
    "UNITORCH_CACHE", os.path.join(os.getenv("HOME", "."), ".cache/unitorch")
)
os.environ["TRANSFORMERS_CACHE"] = UNITORCH_CACHE
os.environ["HF_HOME"] = UNITORCH_CACHE
os.environ["HF_DATASETS_CACHE"] = UNITORCH_CACHE


def get_cache_dir():
    return UNITORCH_CACHE


UNITORCH_TEMP = os.getenv(
    "UNITORCH_TEMP", os.path.join(tempfile.gettempdir(), "unitorch")
)
os.makedirs(UNITORCH_TEMP, exist_ok=True)


def get_temp_dir():
    return UNITORCH_TEMP


def mktempdir(prefix: str = ""):
    return tempfile.mkdtemp(prefix=prefix, dir=UNITORCH_TEMP)


def mktempfile(prefix: str = "", suffix: str = ""):
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=UNITORCH_TEMP)


UNITORCH_HOME = os.environ.get(
    "UNITORCH_HOME", os.path.join(os.getenv("HOME", "."), ".unitorch")
)
os.makedirs(UNITORCH_HOME, exist_ok=True)


def get_dir():
    """Get the path to the Unitorch home directory."""
    return UNITORCH_HOME


### version
VERSION = "0.0.1.5"

# before setup logging
import sklearn

# logging & warning setting
import logging
import warnings

UNITORCH_DEBUG_VALUES = {"OFF": 50, "INFO": 20, "DETAIL": 10, "CPU": 10, "ALL": 0}
UNITORCH_DEBUG = os.environ.get("UNITORCH_DEBUG", "INFO").upper()
UNITORCH_DEBUG = (
    UNITORCH_DEBUG if UNITORCH_DEBUG in UNITORCH_DEBUG_VALUES.keys() else "INFO"
)
UNITORCH_DEBUG_INT = UNITORCH_DEBUG_VALUES.get(UNITORCH_DEBUG)

if UNITORCH_DEBUG == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "999"

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=logging.getLevelName(UNITORCH_DEBUG_INT),
)

# settings
import torch
import random
import numpy as np
import transformers
import torch.multiprocessing

np.complex = complex

torch.multiprocessing.set_sharing_strategy("file_system")

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# useful functions
from unitorch.utils import cached_path as hf_cached_path, read_file, read_json_file
from unitorch.utils import (
    is_deepspeed_available,
    is_megatron_available,
    is_diffusers_available,
    is_opencv_available,
    is_wandb_available,
)

if is_diffusers_available():
    import diffusers

    diffusers.logging.set_verbosity_error()

# imports from other files
import unitorch.datasets
import unitorch.losses
import unitorch.scores
import unitorch.models
import unitorch.modules
import unitorch.optims
import unitorch.schedulers
import unitorch.scores
import unitorch.tasks

# more classes
