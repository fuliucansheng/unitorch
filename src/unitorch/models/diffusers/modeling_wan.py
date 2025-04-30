# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import (
    PretrainedConfig,
    UMT5Config,
    UMT5EncoderModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.models import (
    AutoencoderKLWan,
    WanTransformer3DModel,
)
from diffusers.pipelines import (
    WanPipeline,
    WanImageToVideoPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.diffusers import compute_snr


class GenericWanModel(GenericModel, QuantizationMixin, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        # vae weights
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    replace_keys_in_state_dict = {
        "\.query\.": ".to_q.",
        "\.key\.": ".to_k.",
        "\.value\.": ".to_v.",
        "\.proj_attn\.": ".to_out.0.",
    }

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        image_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_transformer_encoder: Optional[bool] = False,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma

        config_dict = json.load(open(config_path))
        if in_channels is not None:
            config_dict.update({"in_channels": in_channels})
        if out_channels is not None:
            config_dict.update({"out_channels": out_channels})
        self.transformer = WanTransformer3DModel.from_config(config_dict).to(
            torch.bfloat16
        )

        text_config = UMT5Config.from_json_file(text_config_path)
        self.text = UMT5EncoderModel(text_config).to(torch.bfloat16)

        if image_config_path is not None:
            image_config = CLIPVisionConfig.from_json_file(image_config_path)
            self.image = CLIPVisionModel(image_config).to(torch.bfloat16)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKLWan.from_config(vae_config_dict).to(torch.bfloat16)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if freeze_vae_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text.parameters():
                param.requires_grad = False

        if freeze_transformer_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config, ignore_modules=["lm_head", "transformer", "vae"]
            )

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_prompt_outputs(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = "cpu",
    ):
        if enable_cpu_offload:
            self.text.to(cpu_offload_device)
            input_ids = input_ids.to(cpu_offload_device)
            attention_mask = attention_mask.to(cpu_offload_device)
            negative_input_ids = negative_input_ids.to(cpu_offload_device)
            negative_attention_mask = negative_attention_mask.to(cpu_offload_device)
        prompt_embeds = self.text(
            input_ids,
            attention_mask,
        )[0]
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1)
        negative_prompt_embeds = self.text(
            negative_input_ids,
            negative_attention_mask,
        )[0]
        negative_prompt_embeds = (
            negative_prompt_embeds * negative_attention_mask.unsqueeze(-1)
        )
        if enable_cpu_offload:
            self.text.to("cpu")
        return GenericOutputs(
            prompt_embeds=prompt_embeds.to("cpu")
            if enable_cpu_offload
            else prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds.to("cpu")
            if enable_cpu_offload
            else negative_prompt_embeds,
        )


class WanForText2VideoGeneration(GenericWanModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.shape[0]
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme="none",
        #     batch_size=batch,
        #     logit_mean=0.0,
        #     logit_std=1.0,
        #     mode_scale=1.29,
        # )
        # indices = (u * self.scheduler.config.num_train_timesteps).long()
        # timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        # sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        # noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.transformer(
            noise_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        # weighting = compute_loss_weighting_for_sd3(
        #     weighting_scheme="none", sigmas=sigmas
        # )
        # target = noise - latents
        # loss = torch.mean(
        #     (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
        #         target.shape[0], -1
        #     ),
        #     1,
        # )
        # loss = loss.mean()
        loss = F.mse_loss(outputs, noise, reduction="mean")
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
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

        frames = self.pipeline(
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


class WanForImage2VideoGeneration(GenericWanModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        image_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            image_config_path=image_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = WanImageToVideoPipeline(
            vae=self.vae,
            text_encoder=self.text,
            image_encoder=self.image,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            image_processor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        vae_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.shape[0]
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme="none",
        #     batch_size=batch,
        #     logit_mean=0.0,
        #     logit_std=1.0,
        #     mode_scale=1.29,
        # )
        # indices = (u * self.scheduler.config.num_train_timesteps).long()
        # timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        # sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        # noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        num_frames = pixel_values.shape[-3]

        video_condition = torch.cat(
            [
                vae_pixel_values.unsqueeze(2),
                vae_pixel_values.new_zeros(
                    vae_pixel_values.shape[0],
                    vae_pixel_values.shape[1],
                    num_frames - 1,
                    vae_pixel_values.shape[-2],
                    vae_pixel_values.shape[-1],
                    device=vae_pixel_values.device,
                ),
            ],
            dim=2,
        )
        latent_condition = self.vae.encode(video_condition).latent_dist.mode()
        latent_condition = latent_condition.repeat(latents.shape[0], 1, 1, 1, 1).to(
            latents.dtype
        )
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(
            latents.shape[0], 1, num_frames, latents.shape[-2], latents.shape[-1]
        )
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.pipeline.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
        )
        mask_lat_size = mask_lat_size.view(
            latents.shape[0],
            -1,
            self.pipeline.vae_scale_factor_temporal,
            latents.shape[-2],
            latents.shape[-1],
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        condition_latents = torch.concat([mask_lat_size, latent_condition], dim=1)
        latent_model_input = torch.cat([noise_latents, condition_latents], dim=1)

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        condition_hidden_states = self.image(
            condition_pixel_values,
            output_hidden_states=True,
        ).hidden_states[-2]
        outputs = self.transformer(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=condition_hidden_states,
        ).sample
        # weighting = compute_loss_weighting_for_sd3(
        #     weighting_scheme="none", sigmas=sigmas
        # )
        # target = noise - latents
        # loss = torch.mean(
        #     (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
        #         target.shape[0], -1
        #     ),
        #     1,
        # )
        # loss = loss.mean()
        loss = F.mse_loss(outputs, noise, reduction="mean")
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = 81,
        guidance_scale: Optional[float] = 5.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )

        condition_hidden_states = self.image(
            condition_pixel_values,
            output_hidden_states=True,
        ).hidden_states[-2]

        frames = self.pipeline(
            image=vae_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            image_embeds=condition_hidden_states,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=vae_pixel_values.size(-2),
            width=vae_pixel_values.size(-1),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return GenericOutputs(frames=frames)
