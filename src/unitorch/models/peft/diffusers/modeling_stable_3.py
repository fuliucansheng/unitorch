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
    SD3ControlNetModel,
    SD3MultiControlNetModel,
    SD3Transformer2DModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    DDPMPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import GenericPeftModel
from unitorch.models.diffusers import compute_snr


class GenericStable3LoraModel(GenericPeftModel, QuantizationMixin):
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
        text3_config_path: str,
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
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "q_proj",
            "v_proj",
            "out_proj",
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
        self.transformer = SD3Transformer2DModel.from_config(config_dict)

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModelWithProjection(text_config)

        text_config2 = CLIPTextConfig.from_json_file(text2_config_path)
        self.text2 = CLIPTextModelWithProjection(text_config2)

        text_config3 = T5Config.from_json_file(text3_config_path)
        self.text3 = T5EncoderModel(text_config3)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)

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
                    SD3ControlNetModel.from_config(controlnet_config_dict)
                )
            self.num_controlnets = len(controlnets)
            self.controlnet = SD3MultiControlNetModel(
                controlnets=controlnets,
            )
        elif isinstance(controlnet_configs_path, str):
            controlnet_config_dict = json.load(open(controlnet_configs_path))
            self.controlnet = SD3ControlNetModel.from_config(controlnet_config_dict)
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
        for param in self.text3.parameters():
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
            # self.text3.add_adapter(lora_config)
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
        input3_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        negative_input3_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        negative_attention3_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_outputs[0]
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            # negative_attention_mask,
            output_hidden_states=True,
        )
        negative_pooled_prompt_embeds = negative_prompt_outputs[0]
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]

        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        pooled_prompt2_embeds = prompt2_outputs[0]
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            # negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_pooled_prompt2_embeds = negative_prompt2_outputs[0]
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]

        prompt3_outputs = self.text3(
            input3_ids,
            # attention3_mask,
            output_hidden_states=True,
        )
        prompt3_embeds = prompt3_outputs[0]
        negative_prompt3_outputs = self.text3(
            negative_input3_ids,
            # negative_attention3_mask,
            output_hidden_states=True,
        )
        negative_prompt3_embeds = negative_prompt3_outputs[0]

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = torch.concat(
            [pooled_prompt_embeds, pooled_prompt2_embeds], dim=-1
        )
        negative_pooled_prompt_embeds = torch.concat(
            [negative_pooled_prompt_embeds, negative_pooled_prompt2_embeds], dim=-1
        )

        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, prompt3_embeds.shape[-1] - prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([prompt_embeds, prompt3_embeds], dim=-2)
        negative_prompt_embeds = torch.nn.functional.pad(
            negative_prompt_embeds,
            (0, negative_prompt3_embeds.shape[-1] - negative_prompt_embeds.shape[-1]),
        )
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds, negative_prompt3_embeds], dim=-2
        )

        return GenericOutputs(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )


class Stable3LoraForText2ImageGeneration(GenericStable3LoraModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        text3_config_path: str,
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
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "q_proj",
            "v_proj",
            "out_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
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

        self.pipeline = StableDiffusion3Pipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            text_encoder_3=self.text3,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        input3_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_outputs[0]
        prompt_embeds = prompt_outputs.hidden_states[-2]

        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        pooled_prompt2_embeds = prompt2_outputs[0]
        prompt2_embeds = prompt2_outputs.hidden_states[-2]

        prompt3_outputs = self.text3(
            input3_ids,
            # attention3_mask,
            output_hidden_states=True,
        )
        prompt3_embeds = prompt3_outputs[0]

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        pooled_prompt_embeds = torch.concat(
            [pooled_prompt_embeds, pooled_prompt2_embeds], dim=-1
        )

        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, prompt3_embeds.shape[-1] - prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([prompt_embeds, prompt3_embeds], dim=-2)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

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

        outputs = self.transformer(
            noise_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        ).sample

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
        input3_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        negative_input3_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        negative_attention3_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 5.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            input3_ids=input3_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            negative_input3_ids=negative_input3_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            attention3_mask=attention3_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
            negative_attention3_mask=negative_attention3_mask,
        )

        images = self.pipeline(
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
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


class Stable3LoraForImageInpainting(GenericStable3LoraModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        text3_config_path: str,
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
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "q_proj",
            "v_proj",
            "out_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
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

        self.pipeline = StableDiffusion3InpaintPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            text_encoder_3=self.text3,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.num_channels_transformer = self.transformer.config.in_channels

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        input3_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_outputs[0]
        prompt_embeds = prompt_outputs.hidden_states[-2]

        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        pooled_prompt2_embeds = prompt2_outputs[0]
        prompt2_embeds = prompt2_outputs.hidden_states[-2]

        prompt3_outputs = self.text3(
            input3_ids,
            # attention3_mask,
            output_hidden_states=True,
        )
        prompt3_embeds = prompt3_outputs[0]

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        pooled_prompt_embeds = torch.concat(
            [pooled_prompt_embeds, pooled_prompt2_embeds], dim=-1
        )

        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, prompt3_embeds.shape[-1] - prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([prompt_embeds, prompt3_embeds], dim=-2)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

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

        if self.num_channels_transformer == 33:
            masked_pixel_values = pixel_values.clone()
            masked_pixel_masks = pixel_masks.clone()
            masked_pixel_masks = masked_pixel_masks.expand_as(masked_pixel_values)
            masked_pixel_values[masked_pixel_masks > 0.5] = -1.0
            masked_latents = self.vae.encode(masked_pixel_values).latent_dist.sample()
            masked_latents = masked_latents * self.vae.config.scaling_factor

            pixel_masks = torch.nn.functional.interpolate(
                pixel_masks, size=latents.shape[-2:], mode="nearest"
            )

            latent_model_input = torch.cat(
                [noise_latents, pixel_masks, masked_latents], dim=1
            )
        else:
            latent_model_input = noise_latents

        outputs = self.transformer(
            latent_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        ).sample

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
        input3_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        negative_input3_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        negative_attention3_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 7.5,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            input3_ids=input3_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            negative_input3_ids=negative_input3_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            attention3_mask=attention3_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
            negative_attention3_mask=negative_attention3_mask,
        )

        images = self.pipeline(
            image=pixel_values,
            mask_image=pixel_masks,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            width=pixel_values.size(-1),
            height=pixel_values.size(-2),
            strength=strength,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
