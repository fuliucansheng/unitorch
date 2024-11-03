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
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.pipelines import (
    FluxPipeline,
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
from unitorch.models.diffusers.modeling_stable_flux import (
    _prepare_latent_image_ids,
    _pack_latents,
    _unpack_latents,
)
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
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        guidance_scale: Optional[float] = 3.5,
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
            snr_gamma=snr_gamma,
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
        self.guidance_scale = guidance_scale
        self.pipeline.set_progress_bar_config(disable=True)
        self.controlnet_conditioning_mode = controlnet_conditioning_mode

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

        if self.num_controlnets == 1:
            condition_latents = self.vae.encode(
                condition_pixel_values
            ).latent_dist.sample()
            condition_latents = (
                condition_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            condition_latents = _pack_latents(
                condition_latents,
                batch_size=condition_latents.shape[0],
                num_channels_latents=condition_latents.shape[1],
                height=condition_latents.shape[2],
                width=condition_latents.shape[3],
            )
        else:
            condition_pixel_values = condition_pixel_values.transpose(0, 1)
            condition_latents = [
                self.vae.encode(_condition_pixel_values).latent_dist.sample()
                for _condition_pixel_values in list(condition_pixel_values)
            ]
            condition_latents = [
                (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                for latent in condition_latents
            ]
            condition_latents = [
                _pack_latents(
                    latent,
                    batch_size=latent.shape[0],
                    num_channels_latents=latent.shape[1],
                    height=latent.shape[2],
                    width=latent.shape[3],
                )
                for latent in condition_latents
            ]

        if self.num_controlnets == 1:
            use_guidance = self.controlnet.config.guidance_embeds
        else:
            use_guidance = [net.config.guidance_embeds for net in self.controlnet.nets]
            assert all(g == use_guidance[0] for g in use_guidance)
            use_guidance = use_guidance[0]

        if use_guidance:
            guidance = torch.full(
                [1], self.guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.controlnet_conditioning_mode is None:
            controlnet_mode = (
                None if self.num_controlnets == 1 else [None] * self.num_controlnets
            )
        else:
            if self.num_controlnets == 1:
                controlnet_mode = (
                    torch.tensor(self.controlnet_conditioning_mode)
                    .to(self.device)
                    .long()
                )
                controlnet_mode = controlnet_mode.expand(latents.shape[0]).reshape(
                    -1, 1
                )
            else:
                assert (
                    isinstance(self.controlnet_conditioning_mode, list)
                    and len(self.controlnet_conditioning_mode) == self.num_controlnets
                )

                def _get_mode(mode):
                    if mode is None:
                        return None
                    _mode = torch.tensor(mode).to(self.device).long()
                    return _mode.expand(latents.shape[0]).reshape(-1)

                controlnet_mode = [
                    _get_mode(mode) for mode in self.controlnet_conditioning_mode
                ]
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=outputs.prompt_embeds,
            pooled_projections=outputs.pooled_prompt_embeds,
            controlnet_cond=condition_latents,
            controlnet_mode=controlnet_mode,
            conditioning_scale=1.0
            if self.num_controlnets == 1
            else [1.0] * self.num_controlnets,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
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
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
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
            num_inference_steps=self.num_infer_timesteps,
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
        snr_gamma: Optional[float] = 5.0,
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
            snr_gamma=snr_gamma,
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
            num_inference_steps=self.num_infer_timesteps,
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
        snr_gamma: Optional[float] = 5.0,
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
            snr_gamma=snr_gamma,
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
            num_inference_steps=self.num_infer_timesteps,
            width=pixel_values.size(-1),
            height=pixel_values.size(-2),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
