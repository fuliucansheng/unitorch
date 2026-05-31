# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional

try:
    from diffusers import LucyEditPipeline
except ImportError:  # pragma: no cover - depends on installed diffusers version
    LucyEditPipeline = None

from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)

from unitorch.models import GenericOutputs
from unitorch.models.diffusers.modeling_wan import GenericWanModel


class LucyForVideoEditingGeneration(GenericWanModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        expand_timesteps: Optional[bool] = True,
    ):
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
        )
        if LucyEditPipeline is None:
            raise ImportError("LucyEditPipeline requires a newer diffusers version.")

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.expand_timesteps = expand_timesteps
        self.pipeline = LucyEditPipeline(
            vae=self.vae,
            text_encoder=self.text,
            tokenizer=None,
            transformer=self.transformer,
            transformer_2=None,
            scheduler=self.scheduler,
            boundary_ratio=None,
            expand_timesteps=expand_timesteps,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        return (latents - latents_mean) * latents_std

    def _expand_timesteps(self, timesteps: torch.Tensor, latents: torch.Tensor):
        if not self.expand_timesteps:
            return timesteps
        latent_tokens = latents[:, 0, :, ::2, ::2]
        return (
            timesteps.view(-1, 1, 1, 1)
            .expand(latent_tokens.shape)
            .reshape(latents.shape[0], -1)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = self._normalize_latents(latents)
        condition_latents = self.vae.encode(refer_pixel_values).latent_dist.mode()
        condition_latents = self._normalize_latents(condition_latents).to(latents.dtype)
        assert latents.shape == condition_latents.shape

        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
        batch = latents.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise
        latent_model_input = torch.cat([noise_latents, condition_latents], dim=1)

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.transformer(
            latent_model_input,
            self._expand_timesteps(timesteps, latents),
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none",
            sigmas=sigmas,
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        return loss.mean()

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        guidance_scale: Optional[float] = 5.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )

        if refer_pixel_values.ndim == 4:
            refer_pixel_values = refer_pixel_values.unsqueeze(0)
        video = refer_pixel_values.permute(0, 2, 1, 3, 4)
        frames = self.pipeline(
            video=video,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return GenericOutputs(frames=frames)
