# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import json
import logging
import torch
import hashlib
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.pipelines import (
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3ControlNetInpaintingPipeline,
)
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.models.diffusers import GenericStable3Model
from unitorch.models.diffusers import Stable3Processor

from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch.cli.pipelines import Schedulers


class Stable3ForImageInpaintingPipeline(GenericStable3Model):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        text3_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        merge2_path: str,
        vocab3_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        max_seq_length2: Optional[int] = 256,
        pad_token: Optional[str] = "<|endoftext|>",
        pad_token2: Optional[str] = "!",
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = Stable3Processor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            vocab3_path=vocab3_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            pad_token=pad_token,
            pad_token2=pad_token2,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

    @classmethod
    @add_default_section_for_init("core/pipeline/stable_3/inpainting")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "stable-v3-medium",
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        text2_config_path: Optional[str] = None,
        text3_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vocab2_path: Optional[str] = None,
        merge2_path: Optional[str] = None,
        vocab3_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/stable_3/inpainting")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", text_config_path)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        text2_config_path = config.getoption("text2_config_path", text2_config_path)
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        text3_config_path = config.getoption("text3_config_path", text3_config_path)
        text3_config_path = pop_value(
            text3_config_path,
            nested_dict_value(pretrained_infos, "text3", "config"),
        )
        text3_config_path = cached_path(text3_config_path)

        vae_config_path = config.getoption("vae_config_path", vae_config_path)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption(
            "scheduler_config_path", scheduler_config_path
        )
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", merge_path)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        vocab2_path = config.getoption("vocab2_path", vocab2_path)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrained_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        merge2_path = config.getoption("merge2_path", merge2_path)
        merge2_path = pop_value(
            merge2_path,
            nested_dict_value(pretrained_infos, "text2", "merge"),
        )
        merge2_path = cached_path(merge2_path)

        vocab3_path = config.getoption("vocab3_path", vocab3_path)
        vocab3_path = pop_value(
            vocab3_path,
            nested_dict_value(pretrained_infos, "text3", "vocab"),
        )
        vocab3_path = cached_path(vocab3_path)

        quant_config_path = config.getoption("quant_config_path", quant_config_path)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        max_seq_length2 = config.getoption("max_seq_length2", 256)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        pad_token2 = config.getoption("pad_token2", "!")
        weight_path = config.getoption("pretrained_weight_path", pretrained_weight_path)
        device = config.getoption("device", device)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text3", "weight"),
                    prefix_keys={"": "text3."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            text3_config_path=text3_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            vocab3_path=vocab3_path,
            quant_config_path=quant_config_path,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            pad_token=pad_token,
            pad_token2=pad_token2,
            weight_path=weight_path,
            state_dict=state_dict,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/stable_3/inpainting")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        neg_text: Optional[str] = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = 7.5,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        controlnet_checkpoints: Optional[List[str]] = [],
        controlnet_images: Optional[List[Image.Image]] = [],
        controlnet_guidance_scales: Optional[List[float]] = [],
        lora_checkpoints: Optional[Union[str, List[str]]] = [],
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        lora_urls: Optional[Union[str, List[str]]] = [],
        lora_files: Optional[Union[str, List[str]]] = [],
    ):
        if width is None or height is None:
            width, height = image.size
            width = width // 8 * 8
            height = height // 8 * 8
        image = image.resize((width, height))
        mask_image = mask_image.resize((width, height))

        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.inpainting_inputs(image, mask_image)

        assert scheduler is None or scheduler in Schedulers
        if scheduler is not None:
            self.scheduler = Schedulers.get(scheduler).from_config(
                self.scheduler.config
            )

        if any(ckpt is not None for ckpt in controlnet_checkpoints) and any(
            img is not None for img in controlnet_images
        ):
            controlnets, conditioning_scales, conditioning_images = [], [], []
            (
                inpaint_controlnet,
                inpaint_conditioning_scale,
                inpaint_conditioning_image,
            ) = (None, None, None)
            for checkpoint, conditioning_scale, conditioning_image in zip(
                controlnet_checkpoints, controlnet_guidance_scales, controlnet_images
            ):
                if checkpoint is None or conditioning_image is None:
                    continue

                if "inpainting" in checkpoint:
                    inpaint_controlnet_config_path = cached_path(
                        nested_dict_value(
                            pretrained_stable_extensions_infos,
                            checkpoint,
                            "controlnet",
                            "config",
                        )
                    )
                    inpaint_controlnet_config_dict = json.load(
                        open(inpaint_controlnet_config_path)
                    )
                    inpaint_controlnet = SD3ControlNetModel.from_config(
                        inpaint_controlnet_config_dict
                    )
                    inpaint_controlnet.load_state_dict(
                        load_weight(
                            nested_dict_value(
                                pretrained_stable_extensions_infos,
                                checkpoint,
                                "controlnet",
                                "weight",
                            )
                        )
                    )
                    inpaint_controlnet.to(device=self._device)
                    logging.info(f"Loading inpaint controlnet from {checkpoint}")
                    inpaint_conditioning_scale = conditioning_scale
                    inpaint_conditioning_image = conditioning_image.resize(image.size)
                    continue

                controlnet_config_path = cached_path(
                    nested_dict_value(
                        pretrained_stable_extensions_infos,
                        checkpoint,
                        "controlnet",
                        "config",
                    )
                )
                controlnet_config_dict = json.load(open(controlnet_config_path))
                controlnet = SD3ControlNetModel.from_config(controlnet_config_dict)
                controlnet.load_state_dict(
                    load_weight(
                        nested_dict_value(
                            pretrained_stable_extensions_infos,
                            checkpoint,
                            "controlnet",
                            "weight",
                        )
                    )
                )
                controlnet.to(device=self._device)
                logging.info(f"Loading controlnet from {checkpoint}")
                controlnets.append(controlnet)
                conditioning_scales.append(conditioning_scale)
                conditioning_images.append(conditioning_image.resize(image.size))

            if inpaint_controlnet is not None:
                controlnets.append(inpaint_controlnet)
                conditioning_scales.append(inpaint_conditioning_scale)

            if len(controlnets) > 1:
                controlnets = SD3MultiControlNetModel(controlnets)
            else:
                controlnets = controlnets[0]

            self.pipeline = StableDiffusion3ControlNetInpaintingPipeline(
                vae=self.vae,
                text_encoder=self.text,
                text_encoder_2=self.text2,
                text_encoder_3=self.text3,
                transformer=self.transformer,
                controlnet=controlnets,
                scheduler=self.scheduler,
                tokenizer=None,
                tokenizer_2=None,
                tokenizer_3=None,
            )
            if len(conditioning_images) > 0:
                controlnets_inputs = self.processor.controlnets_inputs(
                    conditioning_images
                )
            else:
                controlnets_inputs = None

            if inpaint_conditioning_image is not None:
                inpaint_controlnet_inputs = self.processor.inpainting_control_inputs(
                    inpaint_conditioning_image, mask_image
                )
            else:
                inpaint_controlnet_inputs = None

            if controlnets_inputs is not None and inpaint_controlnet_inputs is not None:
                condition_pixel_values = torch.cat(
                    [
                        controlnets_inputs.pixel_values,
                        inpaint_controlnet_inputs.pixel_values.unsqueeze(0),
                    ],
                    dim=0,
                )
            elif controlnets_inputs is not None:
                condition_pixel_values = controlnets_inputs.pixel_values
            elif inpaint_controlnet_inputs is not None:
                condition_pixel_values = (
                    inpaint_controlnet_inputs.pixel_values.unsqueeze(0)
                )

            enable_controlnet = True
            inputs = {
                **text_inputs,
                **image_inputs,
                **{"condition_pixel_values": condition_pixel_values},
            }
        else:
            self.pipeline = StableDiffusion3InpaintPipeline(
                vae=self.vae,
                text_encoder=self.text,
                text_encoder_2=self.text2,
                text_encoder_3=self.text3,
                transformer=self.transformer,
                scheduler=self.scheduler,
                tokenizer=None,
                tokenizer_2=None,
                tokenizer_3=None,
            )
            enable_controlnet = False
            inputs = {**text_inputs, **image_inputs}
        self.pipeline.set_progress_bar_config(disable=True)
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }
        if isinstance(lora_checkpoints, str):
            lora_checkpoints = [lora_checkpoints]
        if isinstance(lora_weights, float):
            lora_weights = [lora_weights]
        if isinstance(lora_alphas, float):
            lora_alphas = [lora_alphas]
        if isinstance(lora_urls, str):
            lora_urls = [lora_urls]
        if isinstance(lora_files, str):
            lora_files = [lora_files]

        assert (
            len(lora_checkpoints) == len(lora_weights)
            and len(lora_checkpoints) == len(lora_alphas)
            and len(lora_checkpoints) == len(lora_urls)
            and len(lora_checkpoints) == len(lora_files)
        )
        processed_lora_files, processed_lora_weights, processed_lora_alphas = [], [], []
        for ckpt, url, file, weight, alpha in zip(
            lora_checkpoints, lora_urls, lora_files, lora_weights, lora_alphas
        ):
            if ckpt is not None:
                processed_lora_files.append(
                    nested_dict_value(
                        pretrained_stable_extensions_infos, ckpt, "weight"
                    )
                )
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif url is not None and is_remote_url(url):
                processed_lora_files.append(url)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif file is not None:
                processed_lora_files.append(file)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)

        if len(processed_lora_files) > 0:
            self.load_lora_weights(
                processed_lora_files,
                lora_weights=processed_lora_weights,
                lora_alphas=processed_lora_alphas,
            )

        prompt_embeds_results = self.get_prompt_outputs(
            inputs["input_ids"],
            input2_ids=inputs["input2_ids"],
            input3_ids=inputs["input3_ids"],
            negative_input_ids=inputs["negative_input_ids"],
            negative_input2_ids=inputs["negative_input2_ids"],
            negative_input3_ids=inputs["negative_input3_ids"],
        )
        prompt_embeds = prompt_embeds_results.prompt_embeds
        negative_prompt_embeds = prompt_embeds_results.negative_prompt_embeds
        pooled_prompt_embeds = prompt_embeds_results.pooled_prompt_embeds
        negative_pooled_prompt_embeds = (
            prompt_embeds_results.negative_pooled_prompt_embeds
        )

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

        if enable_controlnet:
            outputs = self.pipeline(
                image=inputs["pixel_values"],
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=inputs["pixel_values"].size(-1),
                height=inputs["pixel_values"].size(-2),
                generator=torch.Generator(device=self.pipeline.device).manual_seed(
                    self.seed
                ),
                control_image=list(inputs["condition_pixel_values"].transpose(0, 1)),
                num_inference_steps=num_timesteps,
                guidance_scale=guidance_scale,
                strength=strength,
                controlnet_conditioning_scale=conditioning_scales,
                output_type="np.array",
            )
        else:
            outputs = self.pipeline(
                image=inputs["pixel_values"],
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=inputs["pixel_values"].size(-1),
                height=inputs["pixel_values"].size(-2),
                generator=torch.Generator(device=self.pipeline.device).manual_seed(
                    self.seed
                ),
                num_inference_steps=num_timesteps,
                guidance_scale=guidance_scale,
                strength=strength,
                output_type="np.array",
            )
        self.unload_lora_weights()

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]


@register_script("core/script/stable_3/inpainting")
class Stable3ForImage2ImageGenerationScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        pipe = Stable3ForImage2ImageGenerationPipeline.from_core_configure(config)

        config.set_default_section("core/script/stable_3/inpainting")

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        prompt_col = config.getoption("prompt_col", None)
        neg_prompt_col = config.getoption("neg_prompt_col", None)
        image_col = config.getoption("image_col", None)

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert prompt_col in data.columns, f"Column {prompt_col} not found in data."
        assert image_col in data.columns, f"Column {image_col} not found in data."

        output_folder = config.getoption("output_folder", None)
        output_file = config.getoption("output_file", None)

        def save(image):
            name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
            image.save(f"{output_folder}/{name}")
            return name

        results = []
        if neg_prompt_col in data.columns:
            for prompt, neg_prompt, image in zip(
                data[prompt_col], data[neg_prompt_col], data[image_col]
            ):
                image = Image.open(image)
                result = pipe(prompt, image, neg_text=neg_prompt)
                results.append(save(result))
        else:
            for prompt, image in zip(data[prompt_col], data[image_col]):
                image = Image.open(image)
                result = pipe(prompt, image)
                results.append(save(result))

        data["result"] = results

        data.to_csv(output_file, sep="\t", header=None, index=None, quoting=3)
