# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from pathlib import Path

from unitorch.cli import config_defaults_init, register_process
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.diffusers.modeling_flux3 import resolve_flux3_path
from unitorch.models.diffusers.processing_flux3 import Flux3Processor as _Flux3Processor


class Flux3Processor(_Flux3Processor):
    @classmethod
    @config_defaults_init("core/process/diffusion/flux3")
    def from_config(cls, config, **kwargs):
        path = resolve_flux3_path(config, "core/process/diffusion/flux3")
        return {"tokenizer_path": str(Path(path) / "text_encoder")}

    @register_process("core/process/diffusion/flux3/text2image")
    def _text2image(self, prompt, image):
        return TensorInputs(**super().text2image(prompt, image))

    @register_process("core/process/diffusion/flux3/text2image/inputs")
    def _text2image_inputs(self, prompt):
        return TensorInputs(**super().text2image_inputs(prompt))

    @register_process("core/process/diffusion/flux3/editing")
    def _editing(self, prompt, refer_image, image):
        return TensorInputs(**super().editing(prompt, refer_image, image))

    @register_process("core/process/diffusion/flux3/editing/inputs")
    def _editing_inputs(self, prompt, refer_image):
        return TensorInputs(**super().editing_inputs(prompt, refer_image))
