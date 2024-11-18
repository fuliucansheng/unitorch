# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers import SchedulerMixin
from diffusers.models import (
    T2IAdapter,
    MultiAdapter,
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.pipelines import StableDiffusionAdapterPipeline
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.diffusers import GenericStableModel


class StableAdapterForText2ImageGeneration(GenericStableModel):
    """
    StableAdapter model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        adapter_configs_path (str): Path to the adapter model configuration file.
        scheduler_config_path (str): Path to the scheduler configuration file.
        quant_config_path (Optional[str]): Path to the quantization configuration file (default: None).
        image_size (Optional[int]): Size of the input image (default: None).
        in_channels (Optional[int]): Number of input channels (default: None).
        out_channels (Optional[int]): Number of output channels (default: None).
        num_train_timesteps (Optional[int]): Number of training timesteps (default: 1000).
        num_infer_timesteps (Optional[int]): Number of inference timesteps (default: 50).
        freeze_vae_encoder (Optional[bool]): Whether to freeze the VAE encoder (default: True).
        freeze_text_encoder (Optional[bool]): Whether to freeze the text encoder (default: True).
        freeze_unet_encoder (Optional[bool]): Whether to freeze the UNet encoder (default: True).
        seed (Optional[int]): Random seed (default: 1123).
    """

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        adapter_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = False,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            adapter_configs_path=adapter_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )
        self.pipeline = StableDiffusionAdapterPipeline(
            vae=self.vae,
            text_encoder=self.text,
            unet=self.unet,
            adapter=self.adapter,
            scheduler=self.scheduler,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        adapter_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            input2_ids (torch.Tensor): Second input IDs.
            add_time_ids (torch.Tensor): Additional time IDs.
            pixel_values (torch.Tensor): Pixel values.
            adapter_pixel_values (torch.Tensor): Condition pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).

        Returns:
            torch.Tensor: Loss value.
        """
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

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

        encoder_hidden_states = self.text(input_ids)[0]
        # encoder_hidden_states = self.text(input_ids, attention_mask)[0]

        down_block_additional_residuals = self.adapter(adapter_pixel_values)
        down_block_additional_residuals = [
            sample.to(dtype=noise_latents.dtype)
            for sample in down_block_additional_residuals
        ]
        outputs = self.unet(
            noise_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample

        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(outputs, noise, reduction="mean")
        return loss

    def generate(
        self,
        adapter_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 5.0,
        adapter_conditioning_scale: Optional[Union[float, List[float]]] = None,
    ):
        """
        Generate images using the model.

        Args:
            adapter_pixel_values (torch.Tensor): Condition pixel values.
            input_ids (torch.Tensor): Input IDs.
            negative_input_ids (torch.Tensor): Negative input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            negative_attention_mask (Optional[torch.Tensor]): Negative attention mask (default: None).
            height (Optional[int]): Height of the generated images (default: 1024).
            width (Optional[int]): Width of the generated images (default: 1024).
            guidance_scale (Optional[float]): Scale for guidance (default: 5.0).

        Returns:
            GenericOutputs: Generated images.
        """
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )
        if adapter_conditioning_scale is None:
            if self.num_adapters == 1:
                adapter_conditioning_scale = 1.0
            else:
                adapter_conditioning_scale = [1.0] * self.num_adapters
        elif not isinstance(adapter_conditioning_scale, list) and self.num_adapters > 1:
            adapter_conditioning_scale = [
                adapter_conditioning_scale
            ] * self.num_adapters

        images = self.pipeline(
            image=adapter_pixel_values
            if self.num_adapters == 1
            else list(adapter_pixel_values.transpose(0, 1)),
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
