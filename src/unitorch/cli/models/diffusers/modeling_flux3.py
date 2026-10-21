# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Config adapters; all FLUX.3 model behavior lives in the package layer."""

from huggingface_hub import snapshot_download

from unitorch.cli import config_defaults_init, config_defaults_method, register_model
from unitorch.cli.models import DiffusionOutputs, LossOutputs, diffusion_model_decorator
from unitorch.models.diffusers.checkpoint_flux3 import MODEL_ID, MODEL_REVISION
from unitorch.models.diffusers.modeling_flux3 import (
    Flux3ForText2ImageGeneration as _Flux3ForText2ImageGeneration,
    Flux3ForImageEditingGeneration as _Flux3ForImageEditingGeneration,
)


def resolve_flux3_path(config, section):
    """Resolve only the exact, pinned base variant or an explicit local package."""
    name = config.getdefault(section, "pretrained_name", "flux3-image-klein-base")
    if name != "flux3-image-klein-base":
        raise ValueError("Only the exact flux3-image-klein-base variant is supported")
    path = config.getdefault(section, "pretrained_model_path", None)
    if path is None:
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                 allow_patterns=["config.json", "manifest.json", "model.safetensors", "ae.safetensors", "text_encoder/*"])
    return str(path)


class Flux3CLIAdapter:
    def forward(self, input_ids, pixel_values, attention_mask=None, refer_pixel_values=None,
                negative_input_ids=None, negative_attention_mask=None):
        return LossOutputs(loss=super().forward(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
            refer_pixel_values=refer_pixel_values, negative_input_ids=negative_input_ids,
            negative_attention_mask=negative_attention_mask,
        ))

    @config_defaults_method("core/model/diffusers/flux3/generate")
    def generate(self, input_ids, attention_mask=None, refer_pixel_values=None,
                 negative_input_ids=None, negative_attention_mask=None,
                 height=1024, width=1024, guidance_scale=None, num_infer_timesteps=None, seed=None):
        outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask,
                                   refer_pixel_values=refer_pixel_values, negative_input_ids=negative_input_ids,
                                   negative_attention_mask=negative_attention_mask, height=height, width=width,
                                   guidance_scale=guidance_scale, num_infer_timesteps=num_infer_timesteps, seed=seed)
        return DiffusionOutputs(outputs=outputs.images)


@register_model("core/model/diffusers/text2image/flux3", diffusion_model_decorator)
class Flux3ForText2ImageGeneration(Flux3CLIAdapter, _Flux3ForText2ImageGeneration):
    @classmethod
    @config_defaults_init("core/model/diffusers/text2image/flux3")
    def from_config(cls, config, **kwargs):
        return {"pretrained_model_path": resolve_flux3_path(config, "core/model/diffusers/text2image/flux3")}


@register_model("core/model/diffusers/editing/flux3", diffusion_model_decorator)
class Flux3ForImageEditingGeneration(Flux3CLIAdapter, _Flux3ForImageEditingGeneration):
    @classmethod
    @config_defaults_init("core/model/diffusers/editing/flux3")
    def from_config(cls, config, **kwargs):
        return {"pretrained_model_path": resolve_flux3_path(config, "core/model/diffusers/editing/flux3")}
