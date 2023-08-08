# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.utils.bitsandbytes import (
    replace_with_bnb_linear,
    set_module_quantized_tensor_to_device,
)
from unitorch.utils import is_bitsandbytes_available


def quantize_model(model, config, ignore_modules):
    assert is_bitsandbytes_available(), "Please install bitsandbytes first."
    params = {n: p.clone() for n, p in model.named_parameters()}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=ignore_modules,
        quantization_config=config,
    )
    if ignore_modules is None:
        ignore_modules = ["lm_head"]
    for name, value in {**params, **buffers}.items():
        if any(m in name for m in ignore_modules):
            continue
        set_module_quantized_tensor_to_device(model, name, value.device, value)
    return model


class QuantizationConfig(BitsAndBytesConfig):
    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)


class QuantizationMixin:
    def quantize(self, config: QuantizationConfig, ignore_modules=None):
        return quantize_model(self, config, ignore_modules)
