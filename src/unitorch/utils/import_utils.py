# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import importlib
import importlib.metadata as importlib_metadata
import logging
import types

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_package(package_name: str) -> bool:
    """Return ``True`` if *package_name* is installed and log its version.

    Falls back to ``False`` when the spec is found but version metadata is
    unavailable (e.g. editable installs without ``METADATA``).
    """
    if importlib.util.find_spec(package_name) is None:
        return False
    try:
        version = importlib_metadata.version(package_name)
        logging.debug("Imported %s version %s.", package_name, version)
        return True
    except importlib_metadata.PackageNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Availability flags (evaluated once at import time)
# ---------------------------------------------------------------------------

_deepspeed_available = _check_package("deepspeed")
_megatron_available = importlib.util.find_spec("megatron") is not None
_fastapi_available = _check_package("fastapi")
_diffusers_available = _check_package("diffusers")
_opencv_available = importlib.util.find_spec("cv2") is not None
_onnxruntime_available = _check_package("onnxruntime")
_wandb_available = _check_package("wandb")
_gradio_available = _check_package("gradio")

if _megatron_available:
    logging.debug("Imported megatron.")
if _opencv_available:
    logging.debug("Imported opencv (cv2).")


# ---------------------------------------------------------------------------
# Public availability checks
# ---------------------------------------------------------------------------

def is_deepspeed_available() -> bool:
    return _deepspeed_available


def is_megatron_available() -> bool:
    return _megatron_available


def is_fastapi_available() -> bool:
    return _fastapi_available


def is_diffusers_available() -> bool:
    return _diffusers_available


def is_opencv_available() -> bool:
    return _opencv_available

def is_onnxruntime_available() -> bool:
    return _onnxruntime_available


def is_wandb_available() -> bool:
    return _wandb_available


def is_gradio_available() -> bool:
    return _gradio_available


def is_bfloat16_available() -> bool:
    """Return ``True`` if CUDA is available and the device supports BF16."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Module utilities
# ---------------------------------------------------------------------------

def reload_module(module: types.ModuleType) -> None:
    """Recursively reload *module* and all of its sub-modules."""
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, types.ModuleType) and attr.__name__.startswith(module.__name__):
            reload_module(attr)
    importlib.reload(module)
