# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging
import torch
import importlib
import importlib.metadata as importlib_metadata
from unitorch import is_offline_debug_mode

# deepspeed
_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
try:
    _deepspeed_version = importlib_metadata.version("deepspeed")
    logging.debug(f"Successfully imported deepspeed version {_deepspeed_version}")
except importlib_metadata.PackageNotFoundError:
    _deepspeed_available = False


def is_deepspeed_available():
    return _deepspeed_available or is_offline_debug_mode()


# accelerate
_accelerate_available = importlib.util.find_spec("accelerate") is not None
try:
    _accelerate_version = importlib_metadata.version("accelerate")
    logging.debug(f"Successfully imported accelerate version {_accelerate_version}")
except importlib_metadata.PackageNotFoundError:
    _accelerate_available = False


def is_accelerate_available():
    return _accelerate_available or is_offline_debug_mode()


# megatron
_megatron_available = importlib.util.find_spec("megatron") is not None
try:
    _megatron_version = importlib_metadata.version("megatron")
    logging.debug(f"Successfully imported megatron version {_megatron_version}")
except importlib_metadata.PackageNotFoundError:
    _megatron_available = False


def is_megatron_available():
    return _megatron_available or is_offline_debug_mode()


# diffusers
_diffusers_available = importlib.util.find_spec("diffusers") is not None
try:
    _diffusers_version = importlib_metadata.version("diffusers")
    logging.debug(f"Successfully imported diffusers version {_diffusers_version}")
except importlib_metadata.PackageNotFoundError:
    _diffusers_available = False


def is_diffusers_available():
    return _diffusers_available or is_offline_debug_mode()


# torch
_torch_available = importlib.util.find_spec("torch") is not None
try:
    _torch_version = importlib_metadata.version("torch")
    logging.debug(f"Successfully imported torch version {_torch_version}")
except importlib_metadata.PackageNotFoundError:
    _torch_available = False


def is_torch_available():
    return _torch_available or is_offline_debug_mode()


def is_torch2_available():
    return _torch_available and _torch_version >= "2.0.0"
