# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from typing import Optional

import diffusers.schedulers as schedulers
import torch
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.pipelines import QwenImageEditPipeline, QwenImagePipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, SchedulerMixin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin


def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def _unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
) -> torch.Tensor:
    batch_size, _num_patches, channels = latents.shape
    h = height // vae_scale_factor
    w = width // vae_scale_factor
    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, h, w)


def _build_padded_embeds(hidden_states, start_index: int, device):
    """Trim, right-pad, and stack a list of hidden-state tensors."""
    trimmed = [e[start_index:] for e in hidden_states]
    masks = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in trimmed]
    max_len = max(e.size(0) for e in trimmed)
    embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))]) for u in trimmed
    ])
    mask = torch.stack([
        torch.cat([m, m.new_zeros(max_len - m.size(0))]) for m in masks
    ])
    return embeds, mask


class GenericQWenImageModel(GenericModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }
    replace_keys_in_state_dict = {
        "\\.query\\.": ".to_q.",
        "\\.key\\.": ".to_k.",
        "\\.value\\.": ".to_v.",
        "\\.proj_attn\\.": ".to_out.0.",
    }

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
    ) -> None:
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma

        with open(config_path) as f:
            self.transformer = QwenImageTransformer2DModel.from_config(json.load(f)).to(torch.bfloat16)

        self.text = Qwen2_5_VLForConditionalGeneration(
            Qwen2_5_VLConfig.from_json_file(text_config_path)
        ).to(torch.bfloat16)

        with open(vae_config_path) as f:
            self.vae = AutoencoderKLQwenImage.from_config(json.load(f)).to(torch.bfloat16)

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

        if freeze_vae_encoder:
            for p in self.vae.parameters():
                p.requires_grad_(False)
        if freeze_text_encoder:
            for p in self.text.parameters():
                p.requires_grad_(False)
        if freeze_transformer_encoder:
            for p in self.transformer.parameters():
                p.requires_grad_(False)

    def get_sigmas(self, timesteps: torch.Tensor, n_dim: int = 4, dtype=torch.float32) -> torch.Tensor:
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_start_index: int,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """Run the text encoder and return (prompt_embeds, prompt_embeds_mask)."""
        if pixel_values is not None and image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
            pixel_values = pixel_values.view(-1, pixel_values.size(-1))

        outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        split = self._extract_masked_hidden(outputs.hidden_states[-1], attention_mask)
        return _build_padded_embeds(split, prompt_start_index, input_ids.device)

    def _normalize_latents(self, latents: torch.Tensor, device) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device)
        return (latents - latents_mean) * latents_std

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
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        )
        return loss.mean()

    def _make_guidance(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.transformer.config.guidance_embeds and self.guidance_scale is not None:
            return torch.full([1], self.guidance_scale, device=self.device, dtype=torch.float32).expand(batch_size)
        return None

    def get_prompt_outputs(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        prompt_start_index: int = 0,
        enable_cpu_offload: bool = False,
        cpu_offload_device: str = "cpu",
    ) -> GenericOutputs:
        if enable_cpu_offload:
            self.text = self.text.to(cpu_offload_device)
            input_ids = input_ids.to(cpu_offload_device)
            negative_input_ids = negative_input_ids.to(cpu_offload_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(cpu_offload_device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(cpu_offload_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(cpu_offload_device)
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(cpu_offload_device)

        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            input_ids, attention_mask, prompt_start_index,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt(
            negative_input_ids, negative_attention_mask, prompt_start_index,
        )

        if enable_cpu_offload:
            self.text = self.text.to("cpu")

        return GenericOutputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )


class QWenImageText2ImageGeneration(GenericQWenImageModel):
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
        guidance_scale: float = 1.0,
        prompt_start_index: int = 34,
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
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = QwenImagePipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale
        self.prompt_start_index = prompt_start_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            input_ids, attention_mask, self.prompt_start_index
        )

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)

        latents = self._normalize_latents(
            self.vae.encode(pixel_values).latent_dist.sample(), pixel_values.device
        )
        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        B, C, H, W = latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]
        noise_latents = _pack_latents(noise_latents, B, C, H, W)

        vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        height, width = pixel_values.size(-2), pixel_values.size(-1)
        img_shapes = [[(1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)]] * B

        pred = self.transformer(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=self._make_guidance(B),
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0]
        pred = _unpack_latents(pred, H * vae_scale_factor, W * vae_scale_factor, vae_scale_factor)
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
    ) -> GenericOutputs:
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            prompt_start_index=self.prompt_start_index,
        )
        images = self.pipeline(
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            prompt_embeds_mask=outputs.prompt_embeds_mask,
            negative_prompt_embeds_mask=outputs.negative_prompt_embeds_mask,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(self.seed),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            output_type="np.array",
        ).images
        return GenericOutputs(images=torch.from_numpy(images))


class QWenImageEditingGeneration(GenericQWenImageModel):
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
        guidance_scale: float = 1.0,
        prompt_start_index: int = 64,
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
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = QwenImageEditPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            processor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale
        self.prompt_start_index = prompt_start_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            input_ids, attention_mask, self.prompt_start_index,
            pixel_values=refer_pixel_values,
            image_grid_thw=refer_image_grid_thw,
        )

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)
        if refer_vae_pixel_values.ndim == 4:
            refer_vae_pixel_values = refer_vae_pixel_values.unsqueeze(2)

        latents = self._normalize_latents(
            self.vae.encode(pixel_values).latent_dist.sample(), pixel_values.device
        )
        refer_latents = self._normalize_latents(
            self.vae.encode(refer_vae_pixel_values).latent_dist.mode(), pixel_values.device
        )

        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        B, C, H, W = latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]
        noise_latents_packed = _pack_latents(noise_latents, B, C, H, W)
        rB, rC, rH, rW = refer_latents.shape
        refer_latents_packed = _pack_latents(refer_latents, rB, rC, rH, rW)
        latent_model_input = torch.cat([noise_latents_packed, refer_latents_packed], dim=1)

        vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        height, width = pixel_values.size(-2), pixel_values.size(-1)
        refer_height, refer_width = refer_vae_pixel_values.size(-2), refer_vae_pixel_values.size(-1)
        img_shapes = [[
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            (1, refer_height // vae_scale_factor // 2, refer_width // vae_scale_factor // 2),
        ]] * B

        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=self._make_guidance(B),
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0][:, : noise_latents_packed.shape[1]]

        pred = _unpack_latents(pred, H * vae_scale_factor, W * vae_scale_factor, vae_scale_factor)
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
    ) -> GenericOutputs:
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            pixel_values=refer_pixel_values,
            image_grid_thw=refer_image_grid_thw,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            prompt_start_index=self.prompt_start_index,
        )
        images = self.pipeline(
            image=refer_vae_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            prompt_embeds_mask=outputs.prompt_embeds_mask,
            negative_prompt_embeds_mask=outputs.negative_prompt_embeds_mask,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(self.seed),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            output_type="np.array",
        ).images
        return GenericOutputs(images=torch.from_numpy(images))
