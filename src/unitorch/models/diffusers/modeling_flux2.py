# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from typing import Optional, Tuple

import numpy as np
import torch
import diffusers.schedulers as schedulers
from diffusers import Flux2Pipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import (
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.schedulers import SchedulerMixin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin


class GenericFlux2Model(GenericModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
        "^bn.*": "vae.",
    }

    @staticmethod
    def _build_text_encoder(text_config_path: str):
        with open(text_config_path) as f:
            text_config_dict = json.load(f)
        model_type = text_config_dict.get("model_type")

        if model_type == "mistral3":
            text_config = Mistral3Config.from_dict(text_config_dict)
            return Mistral3ForConditionalGeneration(text_config)

        if model_type == "qwen3":
            text_config = Qwen3Config.from_dict(text_config_dict)
            return Qwen3ForCausalLM(text_config)

        raise ValueError(f"Unsupported FLUX.2 text encoder model_type: {model_type}")

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        freeze_transformer_encoder: bool = False,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
    ) -> None:
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma
        self.text_encoder_out_layers = tuple(text_encoder_out_layers)

        with open(config_path) as f:
            self.transformer = Flux2Transformer2DModel.from_config(json.load(f)).to(
                torch.bfloat16
            )

        self.text = self._build_text_encoder(text_config_path).to(torch.bfloat16)

        with open(vae_config_path) as f:
            self.vae = AutoencoderKLFlux2.from_config(json.load(f)).to(torch.bfloat16)

        with open(scheduler_config_path) as f:
            scheduler_config_dict = json.load(f)
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        self.pipeline = Flux2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text,
            tokenizer=None,
            transformer=self.transformer,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.vae_scale_factor = self.pipeline.vae_scale_factor

        if freeze_vae_encoder:
            for p in self.vae.parameters():
                p.requires_grad_(False)
        if freeze_text_encoder:
            for p in self.text.parameters():
                p.requires_grad_(False)
        if freeze_transformer_encoder:
            for p in self.transformer.parameters():
                p.requires_grad_(False)

    def get_sigmas(
        self, timesteps: torch.Tensor, n_dim: int = 4, dtype=torch.float32
    ) -> torch.Tensor:
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        text_encoder_out_layers = tuple(
            text_encoder_out_layers or self.text_encoder_out_layers
        )
        outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[k] for k in text_encoder_out_layers], dim=1
        )
        hidden_states = hidden_states.to(dtype=self.transformer.dtype, device=self.device)
        batch_size, num_channels, seq_len, hidden_dim = hidden_states.shape
        prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )
        text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(self.device)
        return prompt_embeds, text_ids

    def _normalize_patched_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def _denormalize_patched_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * latents_bn_std + latents_bn_mean

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        sample_mode: str = "sample",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        encoded = self.vae.encode(image)
        if sample_mode == "sample":
            latents = encoded.latent_dist.sample(generator)
        else:
            latents = encoded.latent_dist.mode()
        latents = Flux2Pipeline._patchify_latents(latents)
        return self._normalize_patched_latents(latents)

    def _sample_timesteps_and_sigmas(self, latents: torch.Tensor):
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        return timesteps, sigmas

    def _compute_flow_loss(
        self,
        pred: torch.Tensor,
        noise: torch.Tensor,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        while sigmas.dim() > pred.dim():
            sigmas = sigmas.squeeze(-1)
        while sigmas.dim() < pred.dim():
            sigmas = sigmas.unsqueeze(-1)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            dim=1,
        )
        return loss.mean()

    def _prepare_condition_latents(
        self,
        refer_pixel_values: torch.Tensor,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ):
        image_latents = self._encode_vae_image(
            refer_pixel_values,
            sample_mode="argmax",
            generator=generator,
        )
        image_latent_ids = Flux2Pipeline._prepare_image_ids([image_latents[:1]]).to(
            self.device
        )
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latents = Flux2Pipeline._pack_latents(image_latents)
        return image_latents, image_latent_ids

    def _prepare_noise_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        num_latents_channels = self.transformer.config.in_channels // 4
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if latents is None:
            latents = randn_tensor(
                shape,
                generator=generator,
                device=self.device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device=self.device, dtype=dtype)

        latent_ids = Flux2Pipeline._prepare_latent_ids(latents).to(self.device)
        latents = Flux2Pipeline._pack_latents(latents)
        return latents, latent_ids

    def _decode_latents(self, latents: torch.Tensor, latent_ids: torch.Tensor):
        latents = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)
        latents = self._denormalize_patched_latents(latents)
        latents = Flux2Pipeline._unpatchify_latents(latents)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="np")
        return torch.from_numpy(image)

    @torch.no_grad()
    def _generate_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        refer_pixel_values: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
        num_infer_timesteps: Optional[int] = None,
    ) -> GenericOutputs:
        batch_size = prompt_embeds.shape[0]
        num_infer_timesteps = num_infer_timesteps or self.num_infer_timesteps
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        latents, latent_ids = self._prepare_noise_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )

        image_latents = None
        image_latent_ids = None
        if refer_pixel_values is not None:
            image_latents, image_latent_ids = self._prepare_condition_latents(
                refer_pixel_values=refer_pixel_values,
                batch_size=batch_size,
                generator=generator,
            )

        sigmas = np.linspace(1.0, 1 / num_infer_timesteps, num_infer_timesteps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(
            image_seq_len=latents.shape[1],
            num_steps=num_infer_timesteps,
        )
        timesteps, num_infer_timesteps = retrieve_timesteps(
            self.scheduler,
            num_infer_timesteps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        guidance = torch.full(
            [1],
            guidance_scale,
            device=self.device,
            dtype=torch.float32,
        ).expand(batch_size)

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(self.transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(
                    self.transformer.dtype
                )
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        images = self._decode_latents(latents, latent_ids)
        return GenericOutputs(images=images)


class Flux2ForText2ImageGeneration(GenericFlux2Model):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: float = 4.0,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.guidance_scale = guidance_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        latents = self._encode_vae_image(pixel_values, sample_mode="sample")
        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        latent_ids = Flux2Pipeline._prepare_latent_ids(noise_latents).to(self.device)
        noise_latents = Flux2Pipeline._pack_latents(noise_latents)
        noise = Flux2Pipeline._pack_latents(noise)
        latents = Flux2Pipeline._pack_latents(latents)

        batch_size = latents.shape[0]
        guidance = torch.full(
            [1],
            self.guidance_scale,
            device=self.device,
            dtype=torch.float32,
        ).expand(batch_size)
        pred = self.transformer(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
    ) -> GenericOutputs:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        return self._generate_from_embeds(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )


class Flux2ForImageEditingGeneration(GenericFlux2Model):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: float = 4.0,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.guidance_scale = guidance_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        latents = self._encode_vae_image(pixel_values, sample_mode="sample")
        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        latent_ids = Flux2Pipeline._prepare_latent_ids(noise_latents).to(self.device)
        noise_latents = Flux2Pipeline._pack_latents(noise_latents)
        noise = Flux2Pipeline._pack_latents(noise)
        latents = Flux2Pipeline._pack_latents(latents)

        image_latents, image_latent_ids = self._prepare_condition_latents(
            refer_pixel_values=refer_pixel_values,
            batch_size=latents.shape[0],
        )
        latent_model_input = torch.cat([noise_latents, image_latents], dim=1)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        batch_size = latents.shape[0]
        guidance = torch.full(
            [1],
            self.guidance_scale,
            device=self.device,
            dtype=torch.float32,
        ).expand(batch_size)
        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0][:, : noise_latents.shape[1]]
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 4.0,
    ) -> GenericOutputs:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        return self._generate_from_embeds(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            refer_pixel_values=refer_pixel_values,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
