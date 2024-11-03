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
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3ControlNetInpaintingPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.diffusers import GenericStable3Model


class ControlNet3ForText2ImageGeneration(GenericStable3Model):
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
        text3_config_path: str,
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
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
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
        self.pipeline = StableDiffusion3ControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            text_encoder_3=self.text3,
            transformer=self.transformer,
            controlnet=self.controlnet,
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
        condition_pixel_values: torch.Tensor,
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

        condition_latents = self.vae.encode(condition_pixel_values).latent_dist.sample()
        condition_latents = condition_latents * self.vae.config.scaling_factor
        control_block_samples = self.controlnet(
            hidden_states=noise_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=condition_latents,
            return_dict=False,
        )[0]

        outputs = self.transformer(
            noise_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
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
        condition_pixel_values: torch.Tensor,
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
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
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

        images = self.pipeline(
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
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
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class ControlNet3ForImageInpainting(GenericStable3Model):
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
        text3_config_path: str,
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
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
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
        self.pipeline = StableDiffusion3ControlNetInpaintingPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            text_encoder_3=self.text3,
            transformer=self.transformer,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
    ):
        raise NotImplementedError

    def generate(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        input3_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        negative_input3_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor = None,
        inpainting_condition_pixel_values: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        attention3_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        negative_attention3_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 5.0,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        inpainting_controlnet_conditioning_scale: Optional[float] = None,
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
        if inpainting_condition_pixel_values is not None:
            if self.num_controlnets == 1:
                condition_pixel_values = inpainting_condition_pixel_values
                controlnet_conditioning_scale = inpainting_controlnet_conditioning_scale
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
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
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
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
