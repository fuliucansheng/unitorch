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
from diffusers.models import (
    FluxControlNetModel,
    FluxMultiControlNetModel,
    FluxTransformer2DModel,
    AutoencoderKL,
)
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
from unitorch.models.peft import GenericPeftModel
from unitorch.models.diffusers import compute_snr


class GenericControlNetFluxLoraModel(GenericPeftModel, QuantizationMixin):
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
        self.transformer = FluxTransformer2DModel.from_config(config_dict)

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config)

        text_config2 = CLIPTextConfig.from_json_file(text2_config_path)
        self.text2 = T5EncoderModel(text_config2)

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
                    FluxControlNetModel.from_config(controlnet_config_dict)
                )
            self.num_controlnets = len(controlnets)
            self.controlnet = FluxMultiControlNetModel(
                controlnets=controlnets,
            )
        elif isinstance(controlnet_configs_path, str):
            controlnet_config_dict = json.load(open(controlnet_configs_path))
            self.controlnet = FluxControlNetModel.from_config(controlnet_config_dict)
            self.num_controlnets = 1
        else:
            self.controlnet = None
            self.num_controlnets = 0

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
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
            target_modules=[
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
        )
        if enable_text_adapter:
            self.text.add_adapter(lora_config)
            self.text2.add_adapter(lora_config)
        if enable_transformer_adapter:
            self.transformer.add_adapter(lora_config)

        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)

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


class ControlNetFluxLoraForText2ImageGeneration(GenericControlNetFluxLoraModel):
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
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
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
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
        )

        new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.scheduler.config
        )

        self.pipeline = FluxControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            controlnet=self.controlnet,
            scheduler=new_scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

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
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = 1.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
        )

        images = self.pipeline(
            control_image=condition_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
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
