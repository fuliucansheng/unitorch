# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.utils.bitsandbytes import (
    replace_with_bnb_linear,
    set_module_quantized_tensor_to_device,
)
from unitorch.utils import is_bitsandbytes_available


class QuantizationConfig(BitsAndBytesConfig):
    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)


class QuantizationMixin:
    def quantize(self, config: QuantizationConfig, ignore_modules=None):
        assert is_bitsandbytes_available(), "Please install bitsandbytes first."
        params = {n: p.clone() for n, p in self.named_parameters()}
        buffers = {n: b.clone() for n, b in self.model.named_buffers()}
        self = replace_with_bnb_linear(
            self,
            modules_to_not_convert=ignore_modules,
            quantization_config=config,
        )
        for name, value in {**params, **buffers}.items():
            set_module_quantized_tensor_to_device(self.model, name, value.device, value)
        return self
