# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
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
    ControlNetModel,
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


class GenericControlNetXLModel(GenericModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        # unet weights
        "^add_embedding.*": "unet.",
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
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps

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
        text_config2 = CLIPTextConfig.from_json_file(text2_config_path)
        self.text2 = CLIPTextModelWithProjection(text_config2)
        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)
        controlnet_config_dict = json.load(open(controlnet_config_path))
        self.controlnet = ControlNetModel.from_config(controlnet_config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
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

            for param in self.text2.parameters():
                param.requires_grad = False

        if freeze_unet_encoder:
            for param in self.unet.parameters():
                param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config,
                ignore_modules=["lm_head", "unet", "vae", "controlnet"],
            )

        if lora_r is not None:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.enable_lora(lora_r=lora_r)

        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)

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

    def get_prompt_embeds(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            # negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            # negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = prompt2_outputs[0]
        negative_pooled_prompt_embeds = negative_prompt2_outputs[0]
        return GenericOutputs(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )


class ControlNetXLForText2ImageGeneration(GenericControlNetXLModel):
    """
    ControlNetXL model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        text2_config_path (str): Path to the second text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        controlnet_config_path (str): Path to the ControlNet model configuration file.
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
        lora_r (Optional[int]): Lora parameter (default: None).
        seed (Optional[int]): Random seed (default: 1123).
    """

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_config_path=controlnet_config_path,
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
            lora_r=lora_r,
            seed=seed,
        )
        self.pipeline = StableDiffusionXLControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        add_time_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
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
            condition_pixel_values (torch.Tensor): Condition pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).

        Returns:
            torch.Tensor: Loss value.
        """
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        pooled_prompt_embeds = prompt2_outputs[0]

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

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noise_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_pixel_values,
            added_cond_kwargs={
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds,
            },
            return_dict=False,
        )
        outputs = self.unet(
            noise_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs={
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds,
            },
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(outputs, noise, reduction="mean")
        return loss

    def generate(
        self,
        condition_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 5.0,
        controlnet_conditioning_scale: Optional[float] = 0.5,
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
        outputs = self.get_prompt_embeds(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
        )
        images = self.pipeline(
            image=condition_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class ControlNetXLForImage2ImageGeneration(GenericControlNetXLModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_config_path=controlnet_config_path,
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
            lora_r=lora_r,
            seed=seed,
        )
        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
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
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.99,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = 0.5,
    ):
        outputs = self.get_prompt_embeds(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
        )
        images = self.pipeline(
            image=pixel_values,
            control_image=condition_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class ControlNetXLForImageInpainting(GenericControlNetXLModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_config_path=controlnet_config_path,
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
            lora_r=lora_r,
            seed=seed,
        )
        self.pipeline = StableDiffusionXLControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
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
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = 0.5,
    ):
        outputs = self.get_prompt_embeds(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
        )
        images = self.pipeline(
            image=pixel_values,
            mask_image=pixel_masks,
            control_image=condition_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
