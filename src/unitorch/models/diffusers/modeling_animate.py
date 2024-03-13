# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel
from diffusers.schedulers import SchedulerMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNetMotionModel
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.animatediff.pipeline_animatediff import (
    AnimateDiffPipeline as AnimateText2VideoPipeline,
    AnimateDiffPipelineOutput,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


class GenericAnimateModel(GenericModel, QuantizationMixin):
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
        motion_config_path: str,
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
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.image_size = image_size

        config_dict = json.load(open(config_path))
        if image_size is not None:
            config_dict.update({"sample_size": image_size})
        if in_channels is not None:
            config_dict.update({"in_channels": in_channels})
        if out_channels is not None:
            config_dict.update({"out_channels": out_channels})
        self.unet = UNet2DConditionModel.from_config(config_dict)

        motion_config_dict = json.load(open(motion_config_path))
        self.motion = MotionAdapter.from_config(motion_config_dict)

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
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
            self.quantize(self.quant_config, ignore_modules=["lm_head", "unet", "vae"])

        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)


class AnimateForText2VideoGeneration(GenericAnimateModel):
    """
    AnimateForText2VideoGeneration is a class that represents a model for generating animated videos from text prompts.

    Args:
        config_path (str): The path to the configuration file.
        motion_config_path (str): The path to the motion configuration file.
        text_config_path (str): The path to the text configuration file.
        vae_config_path (str): The path to the VAE (Variational Autoencoder) configuration file.
        scheduler_config_path (str): The path to the scheduler configuration file.
        quant_config_path (str, optional): The path to the quantization configuration file. Defaults to None.
        image_size (int, optional): The size of the input images. Defaults to None.
        in_channels (int, optional): The number of input channels. Defaults to None.
        out_channels (int, optional): The number of output channels. Defaults to None.
        num_train_timesteps (int, optional): The number of training timesteps. Defaults to 1000.
        num_infer_timesteps (int, optional): The number of inference timesteps. Defaults to 50.
        freeze_vae_encoder (bool, optional): Whether to freeze the VAE encoder. Defaults to True.
        freeze_text_encoder (bool, optional): Whether to freeze the text encoder. Defaults to True.
        seed (int, optional): The random seed. Defaults to 1123.
    """

    def __init__(
        self,
        config_path: str,
        motion_config_path: str,
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
        seed: Optional[int] = 1123,
    ):
        """
        Initializes the AnimateModel object.

        Args:
            config_path (str): The path to the main configuration file.
            motion_config_path (str): The path to the motion configuration file.
            text_config_path (str): The path to the text configuration file.
            vae_config_path (str): The path to the VAE configuration file.
            scheduler_config_path (str): The path to the scheduler configuration file.
            quant_config_path (str, optional): The path to the quantization configuration file. Defaults to None.
            image_size (int, optional): The size of the input images. Defaults to None.
            in_channels (int, optional): The number of input channels. Defaults to None.
            out_channels (int, optional): The number of output channels. Defaults to None.
            num_train_timesteps (int, optional): The number of training timesteps. Defaults to 1000.
            num_infer_timesteps (int, optional): The number of inference timesteps. Defaults to 50.
            freeze_vae_encoder (bool, optional): Whether to freeze the VAE encoder. Defaults to True.
            freeze_text_encoder (bool, optional): Whether to freeze the text encoder. Defaults to True.
            seed (int, optional): The random seed. Defaults to 1123.
        """
        super().__init__(
            config_path=config_path,
            motion_config_path=motion_config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            seed=seed,
        )

        self.pipeline = AnimateText2VideoPipeline(
            vae=self.vae,
            text_encoder=self.text,
            unet=self.unet,
            motion_adapter=self.motion,
            scheduler=self.scheduler,
            tokenizer=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def forward(
        self,
    ):
        """
        Performs the forward pass of the model.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError

    def train(self, mode=True):
        """
        Trains the model.

        Args:
            mode (bool): If True, sets the model to training mode. If False, sets the model to evaluation mode.

        Returns:
            None
        """
        if not mode:
            self.pipeline.unet = UNetMotionModel.from_unet2d(self.unet, self.motion)
            self.pipeline.unet.to(device=self.pipeline.device)
            self.unet.eval()
            self.motion.eval()
            self.vae.eval()
            self.text.eval()
        else:
            self.unet.train()
            self.motion.train()
            self.vae.train()
            self.text.train()

    def to(self, *args, **kwargs):
        """
        Moves the model parameters and buffers to the specified device.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The model with the parameters and buffers moved to the specified device.
        """
        self.pipeline.unet.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_frames: Optional[int] = 16,
        guidance_scale: Optional[float] = 7.5,
    ):
        """
        Generates animated frames based on the given input_ids and negative_input_ids.

        Args:
            input_ids (torch.Tensor): The input tensor representing the prompt.
            negative_input_ids (torch.Tensor): The input tensor representing the negative prompt.
            attention_mask (Optional[torch.Tensor], optional): The attention mask tensor. Defaults to None.
            negative_attention_mask (Optional[torch.Tensor], optional): The attention mask tensor for the negative prompt. Defaults to None.
            height (Optional[int], optional): The height of the generated frames. Defaults to 512.
            width (Optional[int], optional): The width of the generated frames. Defaults to 512.
            num_frames (Optional[int], optional): The number of frames to generate. Defaults to 16.
            guidance_scale (Optional[float], optional): The scale factor for guidance. Defaults to 7.5.

        Returns:
            GenericOutputs: The generated frames.
        """
        prompt_embeds = self.text(
            input_ids,
            # attention_mask,
        )[0]
        negative_prompt_embeds = self.text(
            negative_input_ids,
            # negative_attention_mask,
        )[0]

        frames = self.pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=self.num_infer_timesteps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return GenericOutputs(frames=frames)


class AnimateForImage2VideoGeneration(GenericAnimateModel):
    pass
