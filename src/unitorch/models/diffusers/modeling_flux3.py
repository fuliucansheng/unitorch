# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""FLUX.3 base rectified flow with independent text/reference/target streams."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.diffusers.checkpoint_flux3 import (
    atomic_json, load_flux3_components, sha256_file, validate_state,
)


def image_ids(height, width, device, reference_index=None):
    """Image coordinates (t, h/2, w/2, l); reference l = -(index + 1)."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device),
                                torch.arange(width, device=device), indexing="ij")
    zeros = torch.zeros_like(rows)
    layer = zeros if reference_index is None else zeros - reference_index - 1
    return torch.stack([zeros, rows / 2, cols / 2, layer], -1).reshape(-1, 4).float()


def text_ids(length, device):
    result = torch.zeros(length, 4, device=device)
    result[:, 3] = torch.arange(length, device=device)
    return result


def flow_schedule(steps, shift=6.93, device="cpu"):
    if steps < 1 or shift <= 0:
        raise ValueError("steps and shift must be positive")
    t = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return shift * t / (1 + (shift - 1) * t)


def patchify(latents):
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def pack(latents):
    return latents.flatten(2).transpose(1, 2)


class GenericFlux3Model(GenericModel):
    """Load the exact base family; train the image DiT with frozen conditioners.

    Qwen3-VL reads scene captions; source images enter the separate clean VAE
    reference stream. Target images are used only for the flow loss. Caption
    expansion is a separate, source-only stage (see caption_flux3).
    """

    def __init__(
        self, pretrained_model_path: str, torch_dtype: Union[str, torch.dtype] = "bfloat16",
        vae_dtype: Union[str, torch.dtype] = "float32", num_infer_timesteps: int = 50,
        guidance_scale: float = 3.5, flow_shift: float = 6.93, num_train_timesteps: int = 1000,
        text_encoder_out_layers: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32),
        gradient_checkpointing: bool = True, freeze_text_encoder: bool = True,
        freeze_vae_encoder: bool = True, freeze_transformer_encoder: bool = False,
        caption_dropout: float = 0.1, offload_conditioners: bool = False, seed: int = 1123,
    ):
        super().__init__()
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        vae_dtype = getattr(torch, vae_dtype) if isinstance(vae_dtype, str) else vae_dtype
        if dtype not in (torch.float32, torch.bfloat16) or vae_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("FLUX.3 supports float32 or bfloat16; FP16 is not supported")
        if not 0 <= caption_dropout <= 1 or num_train_timesteps < 1:
            raise ValueError("Invalid caption dropout or training timesteps")
        flow_schedule(num_infer_timesteps, flow_shift)
        if offload_conditioners and not (freeze_text_encoder and freeze_vae_encoder):
            raise ValueError("Conditioner offload requires frozen text and VAE encoders")
        self.transformer, self.text, self.vae, self.base_fingerprint, self.native_config = load_flux3_components(
            pretrained_model_path, dtype, vae_dtype,
        )
        self.pretrained_model_path = pretrained_model_path
        self.text_encoder_out_layers = tuple(text_encoder_out_layers)
        expected = len(self.text_encoder_out_layers) * self.text.config.text_config.hidden_size
        if expected != self.transformer.config.joint_attention_dim:
            raise ValueError("Text layer stack width does not match FLUX.3 transformer")
        if self.vae.config.latent_channels * 4 != self.transformer.config.in_channels:
            raise ValueError("VAE latent width does not match FLUX.3 transformer")
        self.image_size_multiple = 2 ** len(self.vae.config.block_out_channels)
        self.num_infer_timesteps, self.guidance_scale, self.flow_shift = num_infer_timesteps, guidance_scale, flow_shift
        self.num_train_timesteps, self.caption_dropout, self.seed = num_train_timesteps, caption_dropout, seed
        self.offload_conditioners = offload_conditioners
        self.text.requires_grad_(not freeze_text_encoder)
        self.vae.requires_grad_(False)
        if not freeze_vae_encoder:
            self.vae.encoder.requires_grad_(True)
            self.vae.quant_conv.requires_grad_(True)
        self.transformer.requires_grad_(not freeze_transformer_encoder)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.text, self.vae):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        self.vae.decoder.eval()
        self.vae.bn.eval()
        return self

    def _conditioner_context(self, module):
        return torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()))

    def _stage(self, module):
        device = self.transformer.device
        if self.offload_conditioners:
            for other in (self.text, self.vae):
                if other is not module:
                    other.cpu()
            module.to(device)
        elif module.device != device:
            raise ValueError("All components must share a device, or enable offload_conditioners")
        return device

    def encode_text(self, input_ids, attention_mask=None, padded_length=None):
        """Preserve right-padding embeddings and the Qwen 4.57 padding position 1."""
        device = self._stage(self.text)
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if input_ids.ndim != 2 or mask.shape != input_ids.shape:
            raise ValueError("Expected batched input_ids and matching attention_mask")
        length = min(input_ids.shape[1], ((int(mask.sum(-1).max()) + 79) // 80) * 80)
        length = max(length, padded_length or 0)
        if length > input_ids.shape[1]:
            input_ids = F.pad(input_ids, (0, length - input_ids.shape[1]))
            mask = F.pad(mask, (0, length - mask.shape[1]))
        ids, mask = input_ids[:, :length].to(device), mask[:, :length].to(device)
        positions = torch.arange(length, device=device)[None].expand_as(ids).masked_fill(mask == 0, 1)
        with self._conditioner_context(self.text):
            output = self.text(input_ids=ids, attention_mask=mask, position_ids=positions,
                               output_hidden_states=True, use_cache=False)
            result = torch.stack([output.hidden_states[k] for k in self.text_encoder_out_layers], dim=2).flatten(2)
        if self.offload_conditioners:
            self.text.cpu()
        return result.to(self.transformer.dtype)

    def encode_image(self, pixels):
        device = self._stage(self.vae)
        if pixels.ndim != 4 or pixels.shape[1] != 3 or any(d % self.image_size_multiple for d in pixels.shape[-2:]):
            raise ValueError(f"Image tensors must be BCHW RGB with sides divisible by {self.image_size_multiple}")
        with self._conditioner_context(self.vae):
            # Posterior mode, never sampled: the reference implementation's mean.
            latents = patchify(self.vae.encode(pixels.to(device=device, dtype=self.vae.dtype)).latent_dist.mode())
            mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1)
            std = (self.vae.bn.running_var + self.vae.config.batch_norm_eps).sqrt().reshape(1, -1, 1, 1)
            latents = (latents - mean) / std
        if self.offload_conditioners:
            self.vae.cpu()
        return latents.to(self.transformer.dtype)

    def encode_references(self, refer_pixel_values):
        if refer_pixel_values is None:
            return None, None
        if refer_pixel_values.ndim == 4:
            refer_pixel_values = refer_pixel_values[:, None]
        if refer_pixel_values.ndim != 5 or not 1 <= refer_pixel_values.shape[1] <= 10:
            raise ValueError("References must be BCHW or BRCHW with 1 to 10 images")
        tokens, ids = [], []
        for index in range(refer_pixel_values.shape[1]):
            latent = self.encode_image(refer_pixel_values[:, index])
            tokens.append(pack(latent))
            ids.append(image_ids(*latent.shape[-2:], latent.device, reference_index=index))
        return torch.cat(tokens, dim=1), torch.cat(ids)

    def _predict(self, noisy, embeds, timestep, ids, references, reference_ids):
        dtype, device = self.transformer.dtype, self.transformer.device
        context = torch.autocast(device.type, dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
        with context:
            return self.transformer(hidden_states=noisy.to(dtype), encoder_hidden_states=embeds.to(dtype),
                                    timestep=timestep, img_ids=ids, txt_ids=text_ids(embeds.shape[1], device),
                                    image_cond_hidden_states=references, image_cond_ids=reference_ids,
                                    return_dict=False)[0]

    def forward(self, input_ids, pixel_values, attention_mask=None, refer_pixel_values=None,
                negative_input_ids=None, negative_attention_mask=None):
        embeds = self.encode_text(input_ids, attention_mask)
        if self.training and self.caption_dropout:
            if negative_input_ids is None:
                raise ValueError("Caption dropout requires tokenized empty/negative captions")
            negative = self.encode_text(negative_input_ids, negative_attention_mask, padded_length=embeds.shape[1])
            drop = torch.rand(embeds.shape[0], 1, 1, device=embeds.device) < self.caption_dropout
            embeds = torch.where(drop, negative, embeds)
        reference, reference_ids = self.encode_references(refer_pixel_values)
        clean = self.encode_image(pixel_values)
        ids = image_ids(*clean.shape[-2:], clean.device)
        clean = pack(clean)
        noise = torch.randn_like(clean)
        u = (self.num_train_timesteps - torch.randint(self.num_train_timesteps, (clean.shape[0],), device=clean.device)).float() / self.num_train_timesteps
        sigma = self.flow_shift * u / (1 + (self.flow_shift - 1) * u)
        noisy = (1 - sigma[:, None, None]) * clean + sigma[:, None, None] * noise
        prediction = self._predict(noisy, embeds, sigma, ids, reference, reference_ids)
        loss = F.mse_loss(prediction.float(), (noise - clean).float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite FLUX.3 flow loss")
        return loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, refer_pixel_values=None,
                 negative_input_ids=None, negative_attention_mask=None, height=1024, width=1024,
                 guidance_scale=None, num_infer_timesteps=None, seed=None):
        if any(d <= 0 or d % self.image_size_multiple for d in (height, width)):
            raise ValueError(f"Output sides must be positive multiples of {self.image_size_multiple}")
        guidance = self.guidance_scale if guidance_scale is None else guidance_scale
        if guidance < 0:
            raise ValueError("guidance_scale must be nonnegative")
        was_training = self.training
        self.eval()
        try:
            embeds = self.encode_text(input_ids, attention_mask)
            negative = None
            if guidance != 1:
                if negative_input_ids is None:
                    raise ValueError("Base CFG requires negative_input_ids; use the FLUX.3 processor")
                negative = self.encode_text(negative_input_ids, negative_attention_mask)
            reference, reference_ids = self.encode_references(refer_pixel_values)
            h, w = height // self.image_size_multiple, width // self.image_size_multiple
            generator = torch.Generator(device="cpu").manual_seed(self.seed if seed is None else seed)
            latents = torch.randn((input_ids.shape[0], h * w, self.transformer.config.in_channels), generator=generator)
            # Native BFL draws CPU noise, rounds to BF16, then integrates in FP32.
            latents = latents.to(self.transformer.dtype).float().to(self.transformer.device)
            ids = image_ids(h, w, latents.device)
            steps = self.num_infer_timesteps if num_infer_timesteps is None else num_infer_timesteps
            schedule = flow_schedule(steps, self.flow_shift, latents.device)
            for current, following in zip(schedule[:-1], schedule[1:]):
                t = current.expand(latents.shape[0])
                conditional = self._predict(latents, embeds, t, ids, reference, reference_ids).float()
                if negative is not None:
                    unconditional = self._predict(latents, negative, t, ids, reference, reference_ids).float()
                    conditional = unconditional + guidance * (conditional - unconditional)
                latents = latents + (following - current) * conditional
                if not torch.isfinite(latents).all():
                    raise FloatingPointError("Non-finite FLUX.3 inference latents")
            self._stage(self.vae)
            z = latents.transpose(1, 2).reshape(latents.shape[0], -1, h, w)
            z = z * (self.vae.bn.running_var.to(z) + self.vae.config.batch_norm_eps).sqrt().reshape(1, -1, 1, 1) + self.vae.bn.running_mean.to(z).reshape(1, -1, 1, 1)
            z = z.reshape(z.shape[0], -1, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(z.shape[0], -1, 2 * h, 2 * w)
            images = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0].float()
            if self.offload_conditioners:
                self.vae.cpu()
            return GenericOutputs(images=(images / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1))
        finally:
            self.train(was_training)

    def checkpoint_state_dict(self, state_dict=None):
        state = self.state_dict() if state_dict is None else state_dict
        selected = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in state.items() if k in selected}

    def from_pretrained(self, weight_path=None, state_dict=None, **kwargs):
        """Refuse the generic permissive loader for this strictly identified base."""
        raise ValueError("FLUX.3 requires a native pretrained_model_path or from_checkpoint with base metadata; partial weights are not accepted")

    def _checkpoint_metadata(self):
        return dict(format_version=1, model="flux3-image-klein-base", base_fingerprint=self.base_fingerprint,
                    text_encoder_out_layers=list(self.text_encoder_out_layers), flow_shift=self.flow_shift,
                    mode="lora" if hasattr(self, "lora_config") else "e2e",
                    lora_config=getattr(self, "lora_config", None))

    def save_checkpoint(self, ckpt_dir, weight_name=None, **kwargs):
        path = Path(ckpt_dir) / (weight_name or self.checkpoint_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.checkpoint_state_dict().items()}
        validate_state(state, state)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".flux3-", suffix=".bin")
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
            atomic_json(str(path) + ".json", {**self._checkpoint_metadata(), "sha256": sha256_file(path)})
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def from_checkpoint(self, ckpt_dir, weight_name=None, **kwargs):
        path = Path(ckpt_dir) / (weight_name or self.checkpoint_name)
        if weight_name == "pytorch_model_latest.bin" and not path.exists():
            if any((path.parent / name).exists() for name in ("info.json", "pytorch_optim_latest.bin", "pytorch_scheduler_latest.bin", "pytorch_model_latest.bin.json")):
                raise ValueError("Incomplete FLUX.3 latest checkpoint")
            return  # The supervised runner creates its fresh output directory first.
        metadata = json.loads(Path(str(path) + ".json").read_text())
        expected = self._checkpoint_metadata()
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("FLUX.3 checkpoint base/config mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_state(state, self.checkpoint_state_dict())
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError("FLUX.3 checkpoint SHA256 mismatch")
        self.load_state_dict(state, strict=False)


class Flux3ForText2ImageGeneration(GenericFlux3Model):
    """FLUX.3 base text-to-image generation and flow training."""


class Flux3ForImageEditingGeneration(GenericFlux3Model):
    """FLUX.3 base editing; the reference is required at train and infer time."""

    def forward(self, input_ids, pixel_values, refer_pixel_values, **kwargs):
        if refer_pixel_values is None:
            raise ValueError("Editing requires source image tensors")
        return super().forward(input_ids, pixel_values, refer_pixel_values=refer_pixel_values, **kwargs)

    def generate(self, input_ids, refer_pixel_values, **kwargs):
        if refer_pixel_values is None:
            raise ValueError("Editing requires source image tensors")
        return super().generate(input_ids, refer_pixel_values=refer_pixel_values, **kwargs)
