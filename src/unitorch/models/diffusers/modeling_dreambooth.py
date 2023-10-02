# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel
from diffusers.schedulers import SchedulerMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionDepth2ImgPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.diffusers.modeling_stable import compute_snr


class DreamboothForText2ImageGeneration(GenericModel, QuantizationMixin):
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
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        prior_loss_weight: Optional[float] = 1.0,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.image_size = image_size
        self.snr_gamma = snr_gamma
        self.prior_loss_weight = prior_loss_weight

        config_dict = json.load(open(config_path))
        if image_size is not None:
            config_dict.update({"sample_size": image_size})
        if in_channels is not None:
            config_dict.update({"in_channels": in_channels})
        if out_channels is not None:
            config_dict.update({"out_channels": out_channels})
        self.unet = UNet2DConditionModel.from_config(config_dict)

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if freeze_vae_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text.parameters():
                param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config, ignore_modules=["lm_head", "unet"])

        if lora_r is not None:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.enable_lora(lora_r=lora_r)

    def enable_lora(self, lora_r: Optional[int] = 4):
        lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if isinstance(
                attn_processor,
                (
                    AttnAddedKVProcessor,
                    SlicedAttnAddedKVProcessor,
                    AttnAddedKVProcessor2_0,
                ),
            ):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else LoRAAttnProcessor
                )

            module = lora_attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_r,
            )

            lora_attn_procs[name] = module

        self.unet.set_attn_processor(lora_attn_procs)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        class_pixel_values: Optional[torch.Tensor] = None,
        class_input_ids: Optional[torch.Tensor] = None,
        class_attention_mask: Optional[torch.Tensor] = None,
    ):
        assert all(t is None for t in [class_pixel_values, class_input_ids]) or all(
            t is not None for t in [class_pixel_values, class_input_ids]
        )

        if class_pixel_values is None:
            mix_pixel_values = pixel_values
            mix_input_ids = input_ids
            mix_attention_mask = attention_mask
        else:
            mix_pixel_values = torch.cat([pixel_values, class_pixel_values], dim=0)
            mix_input_ids = torch.cat([input_ids, class_input_ids], dim=0)
            if attention_mask is not None and class_attention_mask is not None:
                mix_attention_mask = torch.cat(
                    [attention_mask, class_attention_mask], dim=0
                )
            else:
                mix_attention_mask = None

        latents = self.vae.encode(mix_pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch,),
            device=mix_pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = self.text(mix_input_ids, mix_attention_mask)[0]
        outputs = self.unet(
            noise_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)
        if class_pixel_values is not None:
            outputs, outputs_class = torch.chunk(outputs, 2, dim=0)
            noise, noise_class = torch.chunk(noise, 2, dim=0)
            loss_class = F.mse_loss(outputs_class, noise_class, reduction="mean")
        else:
            loss_class = 0
        if self.snr_gamma > 0:
            snr = compute_snr(timesteps, self.scheduler)
            base_weight = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            if self.scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = base_weight + 1
            else:
                mse_loss_weights = base_weight
            loss = F.mse_loss(outputs, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(outputs, noise, reduction="mean")

        return loss + self.prior_loss_weight * loss_class

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
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
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
