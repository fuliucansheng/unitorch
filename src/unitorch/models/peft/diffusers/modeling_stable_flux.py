# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from peft import LoraConfig
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.models import (
    FluxControlNetModel,
    FluxMultiControlNetModel,
    FluxTransformer2DModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxFillPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import GenericPeftModel
from unitorch.models.diffusers import compute_snr
from unitorch.models.diffusers.modeling_stable_flux import (
    _prepare_latent_image_ids,
    _pack_latents,
    _unpack_latents,
)


class GenericStableFluxLoraModel(GenericPeftModel, QuantizationMixin):
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
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        controlnet_configs_path: Union[str, List[str]] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma

        config_dict = json.load(open(config_path))
        if image_size is not None:
            config_dict.update({"sample_size": image_size})
        if in_channels is not None:
            config_dict.update({"in_channels": in_channels})
        if out_channels is not None:
            config_dict.update({"out_channels": out_channels})
        self.transformer = FluxTransformer2DModel.from_config(config_dict).to(
            torch.bfloat16
        )

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config).to(torch.bfloat16)

        text_config2 = CLIPTextConfig.from_json_file(text2_config_path)
        self.text2 = T5EncoderModel(text_config2).to(torch.bfloat16)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict).to(torch.bfloat16)

        if isinstance(controlnet_configs_path, list):
            if len(controlnet_configs_path) == 0:
                controlnet_configs_path = None
            elif len(controlnet_configs_path) == 1:
                controlnet_configs_path = controlnet_configs_path[0]

        if isinstance(controlnet_configs_path, list):
            controlnets = []
            for controlnet_config_path in controlnet_configs_path:
                controlnet_config_dict = json.load(open(controlnet_config_path))
                controlnets.append(
                    FluxControlNetModel.from_config(controlnet_config_dict).to(
                        torch.bfloat16
                    )
                )
            self.num_controlnets = len(controlnets)
            self.controlnet = FluxMultiControlNetModel(
                controlnets=controlnets,
            )
        elif isinstance(controlnet_configs_path, str):
            controlnet_config_dict = json.load(open(controlnet_configs_path))
            self.controlnet = FluxControlNetModel.from_config(
                controlnet_config_dict
            ).to(torch.bfloat16)
            self.num_controlnets = 1
        else:
            self.controlnet = None
            self.num_controlnets = 0

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        for param in self.vae.parameters():
            param.requires_grad = False

        for param in self.text.parameters():
            param.requires_grad = False
        for param in self.text2.parameters():
            param.requires_grad = False

        for param in self.transformer.parameters():
            param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config, ignore_modules=["lm_head", "transformer", "vae"]
            )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if enable_text_adapter:
            self.text.add_adapter(lora_config)
            self.text2.add_adapter(lora_config)
        if enable_transformer_adapter:
            self.transformer.add_adapter(lora_config)

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
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=False,
        )
        pooled_prompt_embeds = prompt_outputs.pooler_output

        prompt_embeds = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=False,
        )[0]

        return GenericOutputs(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )


class StableFluxLoraForText2ImageGeneration(GenericStableFluxLoraModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        controlnet_configs_path: Union[str, List[str]] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 3.5,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            controlnet_configs_path=controlnet_configs_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
        )

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = FluxPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (
            latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        latent_image_ids = _prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            self.device,
            self.dtype,
        )

        noise = torch.randn(latents.shape).to(latents.device)
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

        noise_latents = _pack_latents(
            noise_latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        text_ids = torch.zeros(outputs.prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], self.guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        outputs = self.transformer(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=outputs.prompt_embeds,
            pooled_projections=outputs.pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        outputs = _unpack_latents(
            outputs,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 3.5,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )

        images = self.pipeline(
            prompt_embeds=outputs.prompt_embeds.to(torch.bfloat16),
            pooled_prompt_embeds=outputs.pooled_prompt_embeds.to(torch.bfloat16),
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class StableFluxLoraForImageInpainting(GenericStableFluxLoraModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        controlnet_configs_path: Union[str, List[str]] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 3.5,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            controlnet_configs_path=controlnet_configs_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = FluxFillPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale
        self.num_channels_transformer = self.transformer.config.in_channels

    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (
            latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        latent_image_ids = _prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            self.device,
            self.dtype,
        )

        noise = torch.randn(latents.shape).to(latents.device)
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
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if self.num_channels_transformer == 384:
            masked_pixel_values = pixel_values.clone()
            masked_pixel_masks = pixel_masks.clone()
            masked_pixel_masks = masked_pixel_masks.expand_as(masked_pixel_values)
            masked_pixel_values[masked_pixel_masks > 0.5] = 0.0
            masked_latents = self.vae.encode(masked_pixel_values).latent_dist.sample()
            masked_latents = (
                masked_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

            batch_size, _, height, width = pixel_masks.shape
            height = 2 * (int(height) // (vae_scale_factor * 2))
            width = 2 * (int(width) // (vae_scale_factor * 2))
            pixel_masks = pixel_masks[:, 0, :, :].view(
                batch_size, height, vae_scale_factor, width, vae_scale_factor
            )
            pixel_masks = pixel_masks.permute(
                0, 2, 4, 1, 3
            )  # batch_size, 8, 8, height, width
            pixel_masks = pixel_masks.reshape(
                batch_size, vae_scale_factor * vae_scale_factor, height, width
            )

            latent_model_input = torch.cat(
                [noise_latents, masked_latents, pixel_masks], dim=1
            )
        else:
            latent_model_input = noise_latents

        latent_model_input = _pack_latents(
            latent_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latent_model_input.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        text_ids = torch.zeros(outputs.prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], self.guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        outputs = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=outputs.prompt_embeds,
            pooled_projections=outputs.pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        outputs = _unpack_latents(
            outputs,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss

    def generate(
        self,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 3.5,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )

        images = self.pipeline(
            image=pixel_values,
            mask_image=pixel_masks,
            prompt_embeds=outputs.prompt_embeds.to(torch.bfloat16),
            pooled_prompt_embeds=outputs.pooled_prompt_embeds.to(torch.bfloat16),
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            width=pixel_values.size(-1),
            height=pixel_values.size(-2),
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
