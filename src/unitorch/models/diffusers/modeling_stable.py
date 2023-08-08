# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTextConfig, CLIPTextModel
from diffusers import EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.schedulers import (
    DDPMScheduler,
    PNDMScheduler,
)
from diffusers.pipelines import (
    DDPMPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


class StableForImageGeneration(GenericModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        # unet weights
        "^conv_in.*": "unet.",
        "^conv_norm_out.*": "unet.",
        "^conv_out.*": "unet.",
        "^time_embedding.*": "unet.",
        "^up_blocks.*": "unet.",
        "^mid_block.*": "unet.",
        "^down_blocks.*": "unet.",
        # vae weights
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    def __init__(
        self,
        config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = 224,
        in_channels: Optional[int] = 3,
        out_channels: Optional[int] = 3,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps

        config_dict = json.load(open(config_path))
        config_dict.update(
            {
                "sample_size": image_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
            }
        )
        self.unet = UNet2DModel.from_config(config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        self.scheduler = DDPMScheduler.from_config(scheduler_config_dict)

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

    def forward(self, pixel_values: torch.Tensor):
        noise = torch.randn(pixel_values.shape).to(pixel_values.device)
        batch = pixel_values.size(0)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_pixel_values = self.scheduler.add_noise(
            pixel_values,
            noise,
            timesteps,
        )
        outputs = self.unet(
            noise_pixel_values,
            timesteps,
            return_dict=False,
        )[0]
        loss = F.mse_loss(outputs, noise)
        return loss

    def generate(self, batch_size: int):
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = DDPMPipeline(self.unet, self.scheduler, batch_size)
        images = pipeline(
            batch_size=batch_size,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
        ).images

        return GenericOutputs(images=images)


class StableForText2ImageGeneration(GenericModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        # unet weights
        "^conv_in.*": "unet.",
        "^conv_norm_out.*": "unet.",
        "^conv_out.*": "unet.",
        "^time_embedding.*": "unet.",
        "^up_blocks.*": "unet.",
        "^mid_block.*": "unet.",
        "^down_blocks.*": "unet.",
        # text weights
        "^text_model.*": "text.",
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
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = 224,
        in_channels: Optional[int] = 4,
        out_channels: Optional[int] = 4,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.image_size = image_size

        config_dict = json.load(open(config_path))
        config_dict.update(
            {
                "sample_size": image_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
            }
        )
        self.unet = UNet2DConditionModel.from_config(config_dict)

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        self.scheduler = DDPMScheduler.from_config(scheduler_config_dict)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if freeze_vae_encoder:
            self.vae.requires_grad = False

        if freeze_text_encoder:
            self.text.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.unet(
            noise_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(outputs, noise)
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 7.5,
    ):
        prompt_embeds = self.text(
            input_ids,
            attention_mask,
        )[0]
        negative_prompt_embeds = self.text(
            negative_input_ids,
            negative_attention_mask,
        )[0]
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text,
            unet=self.unet,
            scheduler=self.scheduler,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class StableForImageInpainting(GenericModel):
    pass


class StableForImageResolution(GenericModel):
    pass
