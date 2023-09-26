# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import safetensors
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers import SchedulerMixin
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    DDPMPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


class StableXLForText2ImageGeneration(GenericModel, QuantizationMixin):
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
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
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

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if freeze_vae_encoder:
            self.vae.requires_grad = False

        if freeze_text_encoder:
            self.text.requires_grad = False
            self.text2.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

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
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 5.0,
    ):
        prompt_outputs = self.text(
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = prompt2_outputs[0]
        negative_pooled_prompt_embeds = negative_prompt2_outputs[0]

        images = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class StableXLForImage2ImageGeneration(GenericModel, QuantizationMixin):
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
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
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

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if freeze_vae_encoder:
            self.vae.requires_grad = False

        if freeze_text_encoder:
            self.text.requires_grad = False
            self.text2.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

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
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
    ):
        prompt_outputs = self.text(
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = StableDiffusionXLImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = prompt2_outputs[0]
        negative_pooled_prompt_embeds = negative_prompt2_outputs[0]

        images = pipeline(
            image=pixel_values,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            strength=strength,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class StableXLForImageInpainting(GenericModel, QuantizationMixin):
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
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
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

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if freeze_vae_encoder:
            self.vae.requires_grad = False

        if freeze_text_encoder:
            self.text.requires_grad = False
            self.text2.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

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
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 7.5,
    ):
        prompt_outputs = self.text(
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = StableDiffusionXLInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = prompt2_outputs[0]
        negative_pooled_prompt_embeds = negative_prompt2_outputs[0]

        images = pipeline(
            image=pixel_values,
            mask_image=pixel_masks,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            strength=strength,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


# base + refiner
class StableXLRefinerForText2ImageGeneration(GenericModel, QuantizationMixin):
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
        scheduler_config_path: str,
        refiner_config_path: Optional[str] = None,
        refiner_text_config_path: Optional[str] = None,
        refiner_text2_config_path: Optional[str] = None,
        refiner_vae_config_path: Optional[str] = None,
        refiner_scheduler_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
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

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if refiner_config_path is not None:
            refiner_config_dict = json.load(open(refiner_config_path))
            if image_size is not None:
                refiner_config_dict.update({"sample_size": image_size})
            if in_channels is not None:
                refiner_config_dict.update({"in_channels": in_channels})
            if out_channels is not None:
                refiner_config_dict.update({"out_channels": out_channels})
            self.refiner_unet = UNet2DConditionModel.from_config(refiner_config_dict)
        else:
            self.refiner_unet = None

        if refiner_text_config_path is not None:
            refiner_text_config = CLIPTextConfig.from_json_file(
                refiner_text_config_path
            )
            self.refiner_text = CLIPTextModel(refiner_text_config)
        else:
            self.refiner_text = None

        if refiner_text2_config_path is not None:
            refiner_text_config2 = CLIPTextConfig.from_json_file(
                refiner_text2_config_path
            )
            self.refiner_text2 = CLIPTextModelWithProjection(refiner_text_config2)
        else:
            self.refiner_text2 = None

        if refiner_vae_config_path is not None:
            refiner_vae_config_dict = json.load(open(refiner_vae_config_path))
            self.refiner_vae = AutoencoderKL.from_config(refiner_vae_config_dict)
        else:
            self.refiner_vae = None

        if refiner_scheduler_config_path is not None:
            scheduler_config_dict = json.load(open(refiner_scheduler_config_path))
            scheduler_class_name = scheduler_config_dict.get(
                "_class_name", "DDPMScheduler"
            )
            assert hasattr(schedulers, scheduler_class_name)
            scheduler_class = getattr(schedulers, scheduler_class_name)
            assert issubclass(scheduler_class, SchedulerMixin)
            self.refiner_scheduler = scheduler_class.from_config(scheduler_config_dict)
        else:
            self.refiner_scheduler = None

        if freeze_vae_encoder:
            self.vae.requires_grad = False

        if freeze_text_encoder:
            self.text.requires_grad = False
            self.text2.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

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
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        high_noise_frac: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 5.0,
    ):
        prompt_outputs = self.text(
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        negative_prompt_outputs = self.text(
            negative_input_ids,
            negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        negative_prompt2_outputs = self.text2(
            negative_input2_ids,
            negative_attention2_mask,
            output_hidden_states=True,
        )
        negative_prompt2_embeds = negative_prompt2_outputs.hidden_states[-2]
        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        negative_prompt_embeds = torch.concat(
            [negative_prompt_embeds, negative_prompt2_embeds], dim=-1
        )
        pooled_prompt_embeds = prompt2_outputs[0]
        negative_pooled_prompt_embeds = negative_prompt2_outputs[0]

        images = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            height=height,
            width=width,
            denoising_end=high_noise_frac,
            guidance_scale=guidance_scale,
            output_type="latent",
        ).images

        refiner_pipeline = StableDiffusionXLImg2ImgPipeline(
            vae=self.refiner_vae,
            text_encoder=self.refiner_text,
            text_encoder_2=self.refiner_text2,
            unet=self.refiner_unet,
            scheduler=self.refiner_scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )

        images = refiner_pipeline(
            image=images,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=torch.Generator(device=pipeline.device).manual_seed(self.seed),
            denoising_start=high_noise_frac,
            guidance_scale=guidance_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
