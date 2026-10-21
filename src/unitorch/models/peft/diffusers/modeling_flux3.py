# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""FP32 image-DiT LoRA for the unquantized FLUX.3 base."""

import json
from pathlib import Path

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from unitorch.models.peft import PeftCheckpointMixin
from unitorch.models.diffusers.modeling_flux3 import (
    GenericFlux3Model, Flux3ForText2ImageGeneration, Flux3ForImageEditingGeneration,
)
from unitorch.models.diffusers.checkpoint_flux3 import atomic_json, sha256_file, validate_state


class GenericFlux3LoraMixin(PeftCheckpointMixin):
    """Adapters cover all DiT attention streams, including source-image blocks."""

    def __init__(self, pretrained_model_path: str, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.0, target_modules=None, torch_dtype="bfloat16",
                 vae_dtype="float32", num_infer_timesteps: int = 50, guidance_scale: float = 3.5,
                 flow_shift: float = 6.93, num_train_timesteps: int = 1000,
                 text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
                 gradient_checkpointing: bool = True, caption_dropout: float = 0.1,
                 offload_conditioners: bool = False, seed: int = 1123):
        super().__init__(pretrained_model_path=pretrained_model_path, torch_dtype=torch_dtype,
                         vae_dtype=vae_dtype, num_infer_timesteps=num_infer_timesteps,
                         guidance_scale=guidance_scale, flow_shift=flow_shift,
                         num_train_timesteps=num_train_timesteps,
                         text_encoder_out_layers=text_encoder_out_layers,
                         gradient_checkpointing=gradient_checkpointing,
                         freeze_text_encoder=True, freeze_vae_encoder=True,
                         freeze_transformer_encoder=True, caption_dropout=caption_dropout,
                         offload_conditioners=offload_conditioners, seed=seed)
        targets = target_modules or ["to_q", "to_k", "to_v", "to_out.0", "to_add_out", "add_q_proj", "add_k_proj", "add_v_proj"]
        self.lora_config = dict(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                target_modules=targets)
        self.transformer.add_adapter(LoraConfig(**self.lora_config, init_lora_weights="gaussian"))
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.train(self.training)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return result

    save_checkpoint = GenericFlux3Model.save_checkpoint
    from_checkpoint = GenericFlux3Model.from_checkpoint
    checkpoint_state_dict = GenericFlux3Model.checkpoint_state_dict

    def export_adapter(self, directory: str):
        """Export complete PEFT safetensors plus immutable base/config identity."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        state = {k: v.detach().float().cpu().contiguous()
                 for k, v in get_peft_model_state_dict(self.transformer).items()}
        validate_state(state, state, "adapter")
        destination = path / "adapter_model.safetensors"
        save_file(state, str(destination))
        self.transformer.peft_config["default"].save_pretrained(path)
        atomic_json(path / "flux3_adapter.json", {**self._checkpoint_metadata(), "sha256": sha256_file(destination)})
        return str(path)

    def load_adapter(self, directory: str):
        """Check the full adapter and its base before mutating the transformer."""
        path = Path(directory)
        metadata = json.loads((path / "flux3_adapter.json").read_text())
        if any(metadata.get(k) != v for k, v in self._checkpoint_metadata().items()):
            raise ValueError("FLUX.3 adapter base/config mismatch; FLUX.2 adapters are incompatible")
        destination = path / "adapter_model.safetensors"
        if sha256_file(destination) != metadata["sha256"]:
            raise ValueError("FLUX.3 adapter SHA256 mismatch")
        state = load_file(str(destination))
        validate_state(state, get_peft_model_state_dict(self.transformer), "adapter")
        set_peft_model_state_dict(self.transformer, state)


class Flux3LoraForText2ImageGeneration(GenericFlux3LoraMixin, Flux3ForText2ImageGeneration):
    """Train and sample FLUX.3 text-to-image LoRA."""


class Flux3LoraForImageEditingGeneration(GenericFlux3LoraMixin, Flux3ForImageEditingGeneration):
    """Train and sample FLUX.3 editing LoRA with frozen Qwen3-VL and VAE."""
