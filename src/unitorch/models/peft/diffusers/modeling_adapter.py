# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from peft import LoraConfig
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
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import GenericPeftModel


class GenericStableAdapterLoraModel(GenericPeftModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        # unet weights
        "^conv_in.*": "unet.",
        "^conv_norm_out.*": "unet.",
        "^conv_out.*": "unet.",
        "^time_embedding.*": "unet.",
        "^class_embedding.*": "unet.",
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
        adapter_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
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
        enable_unet_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.snr_gamma = snr_gamma
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
        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)
        if isinstance(adapter_configs_path, list):
            adapters = []
            for adapter_config_path in adapter_configs_path:
                adapter_config_dict = json.load(open(adapter_config_path))
                adapters.append(T2IAdapter.from_config(adapter_config_dict))
            self.num_adapters = len(adapters)
            self.adapter = MultiAdapter(
                adapters=adapters,
            )
        elif isinstance(adapter_configs_path, str):
            adapter_config_dict = json.load(open(adapter_configs_path))
            self.adapter = T2IAdapter.from_config(adapter_config_dict)
            self.num_adapters = 1
        else:
            self.adapter = None
            self.num_adapters = 0

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config,
                ignore_modules=["lm_head", "unet", "vae", "adapter"],
            )

        for param in self.parameters():
            param.requires_grad = False

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
        if enable_unet_adapter:
            self.unet.add_adapter(lora_config)


class StableAdapterLoraForText2ImageGeneration(GenericStableAdapterLoraModel):
    """
    ControlNetXL model for text-to-image generation.

    Args:
        config_path (str): Path to the model configuration file.
        text_config_path (str): Path to the text model configuration file.
        text2_config_path (str): Path to the second text model configuration file.
        vae_config_path (str): Path to the VAE model configuration file.
        adapter_configs_path (str): Path to the ControlNet model configuration file.
        scheduler_config_path (str): Path to the scheduler configuration file.
        quant_config_path (Optional[str]): Path to the quantization configuration file (default: None).
        image_size (Optional[int]): Size of the input image (default: None).
        in_channels (Optional[int]): Number of input channels (default: None).
        out_channels (Optional[int]): Number of output channels (default: None).
        num_train_timesteps (Optional[int]): Number of training timesteps (default: 1000).
        num_infer_timesteps (Optional[int]): Number of inference timesteps (default: 50).
        lora_r (Optional[int]): Lora parameter (default: None).
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
        enable_unet_adapter: Optional[bool] = True,
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
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_unet_adapter=enable_unet_adapter,
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
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            pixel_values (torch.Tensor): Pixel values.
            adapter_pixel_values (torch.Tensor): Condition pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).

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
        adapter_conditioning_scale: Optional[float] = 0.5,
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
        prompt_embeds = self.text(
            input_ids,
            # attention_mask,
        )[0]
        negative_prompt_embeds = self.text(
            negative_input_ids,
            # negative_attention_mask,
        )[0]
        images = self.pipeline(
            image=adapter_pixel_values,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=float(adapter_conditioning_scale),
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
