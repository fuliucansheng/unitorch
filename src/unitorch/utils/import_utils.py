# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging
import torch
import importlib
import importlib.metadata as importlib_metadata


def reload_module(module):
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, type(sys)) and attr.__name__.startswith(module.__name__):
            reload_module(attr)
    importlib.reload(module)


# deepspeed
_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
try:
    _deepspeed_version = importlib_metadata.version("deepspeed")
    logging.debug(f"Successfully imported deepspeed version {_deepspeed_version}")
except importlib_metadata.PackageNotFoundError:
    _deepspeed_available = False


def is_deepspeed_available():
    return _deepspeed_available


# megatron
_megatron_available = importlib.util.find_spec("megatron") is not None
if _megatron_available:
    logging.debug(f"Successfully imported megatron")


def is_megatron_available():
    return _megatron_available


# fastapi
_fastapi_available = importlib.util.find_spec("fastapi") is not None
try:
    _fastapi_version = importlib_metadata.version("fastapi")
    logging.debug(f"Successfully imported fastapi version {_fastapi_version}")
except importlib_metadata.PackageNotFoundError:
    _fastapi_available = False


def is_fastapi_available():
    return _fastapi_available


# diffusers
_diffusers_available = importlib.util.find_spec("diffusers") is not None
try:
    _diffusers_version = importlib_metadata.version("diffusers")
    logging.debug(f"Successfully imported diffusers version {_diffusers_version}")
except importlib_metadata.PackageNotFoundError:
    _diffusers_available = False


def is_diffusers_available():
    return _diffusers_available


# opencv
_opencv_available = importlib.util.find_spec("cv2") is not None
if _opencv_available:
    logging.debug(f"Successfully imported opencv")


def is_opencv_available():
    return _opencv_available


# bitsandbytes
_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
try:
    _bitsandbytes_version = importlib_metadata.version("bitsandbytes")
    logging.debug(f"Successfully imported bitsandbytes version {_bitsandbytes_version}")
except importlib_metadata.PackageNotFoundError:
    _bitsandbytes_available = False


def is_bitsandbytes_available():
    return _bitsandbytes_available


# onnxruntime
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onnxruntime_version = importlib_metadata.version("onnxruntime")
    logging.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")
except importlib_metadata.PackageNotFoundError:
    _onnxruntime_available = False


def is_onnxruntime_available():
    return _onnxruntime_available


# wandb
_wandb_available = importlib.util.find_spec("wandb") is not None
try:
    _wandb_version = importlib_metadata.version("wandb")
    logging.debug(f"Successfully imported wandb version {_wandb_version}")
except importlib_metadata.PackageNotFoundError:
    _wandb_available = False


def is_wandb_available():
    return _wandb_available


# gradio
_gradio_available = importlib.util.find_spec("gradio") is not None
try:
    _gradio_version = importlib_metadata.version("gradio")
    logging.debug(f"Successfully imported gradio version {_gradio_version}")
except importlib_metadata.PackageNotFoundError:
    _gradio_available = False


def is_gradio_available():
    return _gradio_available


# is cuda & bfloat16 avaliable
def is_bfloat16_available():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def is_cuda_available():
    return torch.cuda.is_available()
