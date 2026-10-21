# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import config_defaults_init, register_model
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers.modeling_flux3 import Flux3CLIAdapter, resolve_flux3_path
from unitorch.models.peft.diffusers.modeling_flux3 import (
    Flux3LoraForText2ImageGeneration as _Flux3LoraForText2ImageGeneration,
    Flux3LoraForImageEditingGeneration as _Flux3LoraForImageEditingGeneration,
)


@register_model("core/model/diffusers/text2image/peft/lora/flux3", diffusion_model_decorator)
class Flux3LoraForText2ImageGeneration(Flux3CLIAdapter, _Flux3LoraForText2ImageGeneration):
    @classmethod
    @config_defaults_init("core/model/diffusers/text2image/peft/lora/flux3")
    def from_config(cls, config, **kwargs):
        return {"pretrained_model_path": resolve_flux3_path(config, "core/model/diffusers/text2image/peft/lora/flux3")}


@register_model("core/model/diffusers/editing/peft/lora/flux3", diffusion_model_decorator)
class Flux3LoraForImageEditingGeneration(Flux3CLIAdapter, _Flux3LoraForImageEditingGeneration):
    @classmethod
    @config_defaults_init("core/model/diffusers/editing/peft/lora/flux3")
    def from_config(cls, config, **kwargs):
        return {"pretrained_model_path": resolve_flux3_path(config, "core/model/diffusers/editing/peft/lora/flux3")}
