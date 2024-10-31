# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
from torch import autocast
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin
from diffusers.pipelines import (
    FluxControlNetPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxControlNetInpaintPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.diffusers import GenericStableFluxModel
from unitorch.models.diffusers import compute_snr


class ControlNetFluxForText2ImageGeneration(GenericStableFluxModel):
    """
    ControlNetXL model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        text2_config_path (str): Path to the second text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        controlnet_configs_path (str): Path to the ControlNet model configuration file.
        scheduler_config_path (str): Path to the scheduler configuration file.
        quant_config_path (Optional[str]): Path to the quantization configuration file (default: None).
        image_size (Optional[int]): Size of the input image (default: None).
        in_channels (Optional[int]): Number of input channels (default: None).
        out_channels (Optional[int]): Number of output channels (default: None).
        num_train_timesteps (Optional[int]): Number of training timesteps (default: 1000).
        num_infer_timesteps (Optional[int]): Number of inference timesteps (default: 50).
        freeze_vae_encoder (Optional[bool]): Whether to freeze the VAE encoder (default: True).
        freeze_text_encoder (Optional[bool]): Whether to freeze the text encoder (default: True).
        freeze_transformer_encoder (Optional[bool]): Whether to freeze the transformer encoder (default: True).
        seed (Optional[int]): Random seed (default: 1123).
    """

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_transformer_encoder: Optional[bool] = False,
        seed: Optional[int] = 1123,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_transformer_encoder=freeze_transformer_encoder,
            seed=seed,
        )
        self.pipeline = FluxControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.controlnet_conditioning_mode = controlnet_conditioning_mode

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def forward(
        self,
        condition_pixel_values: torch.Tensor,
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

        height, width = pixel_values.shape[-2:]

        guidance = torch.full(
            [1], self.guidance_scale, device=self.device, dtype=torch.float32
        )
        guidance = guidance.expand(latents.shape[0])
        text_ids = torch.zeros(outputs.prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=self.dtype
        )
        latent_image_ids = self._prepare_latent_image_ids(
            latents.shape[0], height // 2, width // 2, self.device, self.dtype
        )
        condition_latents = self.vae.encode(condition_pixel_values).latent_dist.sample()
        condition_latents = condition_latents * self.vae.config.scaling_factor
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            conditioning_scale=guidance,
            encoder_hidden_states=outputs.prompt_embeds,
            pooled_projections=outputs.pooled_prompt_embeds,
            controlnet_cond=condition_latents,
            controlnet_mode=self.controlnet_conditioning_mode,
            return_dict=False,
        )

        outputs = self.transformer(
            noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=outputs.prompt_embeds,
            pooled_projections=outputs.pooled_prompt_embeds,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
        ).sample

        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)
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
            mse_loss_weights[snr == 0] = 1.0
            loss = F.mse_loss(outputs, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(outputs, noise, reduction="mean")
        return loss

    def generate(
        self,
        condition_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 5.0,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        """
        Generate images using the model.
        Args:
            condition_pixel_values (torch.Tensor): Condition pixel values.
            input_ids (torch.Tensor): Input IDs.
            input2_ids (torch.Tensor): Second input IDs.
            negative_input_ids (torch.Tensor): Negative input IDs.
            negative_input2_ids (torch.Tensor): Negative second input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).
            negative_attention_mask (Optional[torch.Tensor]): Negative attention mask (default: None).
            negative_attention2_mask (Optional[torch.Tensor]): Negative second attention mask (default: None).
            height (Optional[int]): Height of the generated images (default: 1024).
            width (Optional[int]): Width of the generated images (default: 1024).
            guidance_scale (Optional[float]): Scale for guidance (default: 5.0).

        Returns:
            GenericOutputs: Generated images.
        """
        controlnet_conditioning_mode = (
            controlnet_conditioning_mode or self.controlnet_conditioning_mode
        )
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )
        if controlnet_conditioning_scale is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_scale = 1.0
            else:
                controlnet_conditioning_scale = [1.0] * self.num_controlnets
        elif (
            not isinstance(controlnet_conditioning_scale, list)
            and self.num_controlnets > 1
        ):
            controlnet_conditioning_scale = [
                controlnet_conditioning_scale
            ] * self.num_controlnets

        if controlnet_conditioning_mode is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_mode = None
            else:
                controlnet_conditioning_mode = [None] * self.num_controlnets

        images = self.pipeline(
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
            control_mode=controlnet_conditioning_mode,
            prompt_embeds=outputs.prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class ControlNetFluxForImage2ImageGeneration(GenericStableFluxModel):
    """
    ControlNetXL model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        text2_config_path (str): Path to the second text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        controlnet_configs_path (str): Path to the ControlNet model configuration file.
        scheduler_config_path (str): Path to the scheduler configuration file.
        quant_config_path (Optional[str]): Path to the quantization configuration file (default: None).
        image_size (Optional[int]): Size of the input image (default: None).
        in_channels (Optional[int]): Number of input channels (default: None).
        out_channels (Optional[int]): Number of output channels (default: None).
        num_train_timesteps (Optional[int]): Number of training timesteps (default: 1000).
        num_infer_timesteps (Optional[int]): Number of inference timesteps (default: 50).
        freeze_vae_encoder (Optional[bool]): Whether to freeze the VAE encoder (default: True).
        freeze_text_encoder (Optional[bool]): Whether to freeze the text encoder (default: True).
        freeze_transformer_encoder (Optional[bool]): Whether to freeze the transformer encoder (default: True).
        seed (Optional[int]): Random seed (default: 1123).
    """

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_transformer_encoder: Optional[bool] = False,
        seed: Optional[int] = 1123,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_transformer_encoder=freeze_transformer_encoder,
            seed=seed,
        )
        self.pipeline = FluxControlNetImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.controlnet_conditioning_mode = controlnet_conditioning_mode

    def forward(
        self,
    ):
        raise NotImplementedError

    def generate(
        self,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.99,
        guidance_scale: Optional[float] = 5.0,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        """
        Generate images using the model.

        Args:
            condition_pixel_values (torch.Tensor): Condition pixel values.
            input_ids (torch.Tensor): Input IDs.
            input2_ids (torch.Tensor): Second input IDs.
            negative_input_ids (torch.Tensor): Negative input IDs.
            negative_input2_ids (torch.Tensor): Negative second input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).
            negative_attention_mask (Optional[torch.Tensor]): Negative attention mask (default: None).
            negative_attention2_mask (Optional[torch.Tensor]): Negative second attention mask (default: None).
            height (Optional[int]): Height of the generated images (default: 1024).
            width (Optional[int]): Width of the generated images (default: 1024).
            guidance_scale (Optional[float]): Scale for guidance (default: 5.0).

        Returns:
            GenericOutputs: Generated images.
        """
        controlnet_conditioning_mode = (
            controlnet_conditioning_mode or self.controlnet_conditioning_mode
        )
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )
        if controlnet_conditioning_scale is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_scale = 1.0
            else:
                controlnet_conditioning_scale = [1.0] * self.num_controlnets
        elif (
            not isinstance(controlnet_conditioning_scale, list)
            and self.num_controlnets > 1
        ):
            controlnet_conditioning_scale = [
                controlnet_conditioning_scale
            ] * self.num_controlnets

        if controlnet_conditioning_mode is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_mode = None
            else:
                controlnet_conditioning_mode = [None] * self.num_controlnets

        images = self.pipeline(
            image=pixel_values,
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
            control_mode=controlnet_conditioning_mode,
            prompt_embeds=outputs.prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class ControlNetFluxForImageInpainting(GenericStableFluxModel):
    """
    ControlNetXL model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        text2_config_path (str): Path to the second text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        controlnet_configs_path (str): Path to the ControlNet model configuration file.
        scheduler_config_path (str): Path to the scheduler configuration file.
        quant_config_path (Optional[str]): Path to the quantization configuration file (default: None).
        image_size (Optional[int]): Size of the input image (default: None).
        in_channels (Optional[int]): Number of input channels (default: None).
        out_channels (Optional[int]): Number of output channels (default: None).
        num_train_timesteps (Optional[int]): Number of training timesteps (default: 1000).
        num_infer_timesteps (Optional[int]): Number of inference timesteps (default: 50).
        freeze_vae_encoder (Optional[bool]): Whether to freeze the VAE encoder (default: True).
        freeze_text_encoder (Optional[bool]): Whether to freeze the text encoder (default: True).
        freeze_transformer_encoder (Optional[bool]): Whether to freeze the transformer encoder (default: True).
        seed (Optional[int]): Random seed (default: 1123).
    """

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        inpainting_controlnet_config_path: Union[str] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_transformer_encoder: Optional[bool] = False,
        seed: Optional[int] = 1123,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
        inpainting_controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            inpainting_controlnet_config_path=inpainting_controlnet_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_transformer_encoder=freeze_transformer_encoder,
            seed=seed,
        )
        self.pipeline = FluxControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.controlnet_conditioning_mode = controlnet_conditioning_mode
        self.inpainting_controlnet_conditioning_mode = (
            inpainting_controlnet_conditioning_mode
        )

    def forward(
        self,
    ):
        raise NotImplementedError

    def generate(
        self,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor = None,
        inpainting_condition_pixel_values: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 5.0,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
        inpainting_controlnet_conditioning_scale: Optional[float] = None,
        inpainting_controlnet_conditioning_mode: Optional[int] = None,
    ):
        """
        Generate images using the model.

        Args:
            condition_pixel_values (torch.Tensor): Condition pixel values.
            input_ids (torch.Tensor): Input IDs.
            input2_ids (torch.Tensor): Second input IDs.
            negative_input_ids (torch.Tensor): Negative input IDs.
            negative_input2_ids (torch.Tensor): Negative second input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).
            negative_attention_mask (Optional[torch.Tensor]): Negative attention mask (default: None).
            negative_attention2_mask (Optional[torch.Tensor]): Negative second attention mask (default: None).
            height (Optional[int]): Height of the generated images (default: 1024).
            width (Optional[int]): Width of the generated images (default: 1024).
            guidance_scale (Optional[float]): Scale for guidance (default: 5.0).

        Returns:
            GenericOutputs: Generated images.
        """
        assert (
            condition_pixel_values is not None
            or inpainting_condition_pixel_values is not None
        )
        controlnet_conditioning_mode = (
            controlnet_conditioning_mode or self.controlnet_conditioning_mode
        )
        inpainting_controlnet_conditioning_mode = (
            inpainting_controlnet_conditioning_mode
            or self.inpainting_controlnet_conditioning_mode
        )
        if inpainting_condition_pixel_values is not None:
            if self.num_controlnets == 1:
                condition_pixel_values = inpainting_condition_pixel_values
                controlnet_conditioning_scale = inpainting_controlnet_conditioning_scale
                controlnet_conditioning_mode = inpainting_controlnet_conditioning_mode
            else:
                condition_pixel_values = torch.cat(
                    [
                        condition_pixel_values,
                        inpainting_condition_pixel_values.unsqueeze(1),
                    ],
                    dim=1,
                )
                if controlnet_conditioning_scale is None:
                    controlnet_conditioning_scale = [1.0] * (self.num_controlnets - 1)
                controlnet_conditioning_scale += [
                    inpainting_controlnet_conditioning_scale
                ]
                if controlnet_conditioning_mode is None:
                    controlnet_conditioning_mode = [None] * (self.num_controlnets - 1)
                controlnet_conditioning_mode += [
                    inpainting_controlnet_conditioning_mode
                ]
        else:
            if controlnet_conditioning_scale is None:
                if self.num_controlnets == 1:
                    controlnet_conditioning_scale = 1.0
                else:
                    controlnet_conditioning_scale = [1.0] * self.num_controlnets
            elif (
                not isinstance(controlnet_conditioning_scale, list)
                and self.num_controlnets > 1
            ):
                controlnet_conditioning_scale = [
                    controlnet_conditioning_scale
                ] * self.num_controlnets

            if controlnet_conditioning_mode is None:
                if self.num_controlnets == 1:
                    controlnet_conditioning_mode = None
                else:
                    controlnet_conditioning_mode = [None] * self.num_controlnets

        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )

        images = self.pipeline(
            image=pixel_values,
            mask_image=pixel_masks,
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
            control_mode=controlnet_conditioning_mode,
            prompt_embeds=outputs.prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            width=pixel_values.size(-1),
            height=pixel_values.size(-2),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
