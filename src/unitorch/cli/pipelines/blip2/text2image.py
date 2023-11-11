# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch import is_xformers_available
from unitorch.models.diffusers import (
    Blip2ForText2ImageGeneration as _Blip2ForText2ImageGeneration,
)
from unitorch.models.diffusers import Blip2Processor

from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import pretrained_diffusers_infos, load_weight


class Blip2ForText2ImageGenerationPipeline(_Blip2ForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        qformer_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        pad_token: Optional[str] = "<|endoftext|>",
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            qformer_config_path=qformer_config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = Blip2Processor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            pad_token=pad_token,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        if enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
            self.to(torch.half)
        else:
            self.to(device=self._device)

        if enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

    @classmethod
    @add_default_section_for_init("core/pipeline/blip2/text2image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/pipeline/blip2/text2image")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-v1.5-blipdiffuion"
        )
        pretrain_infos = nested_dict_value(pretrained_diffusers_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrain_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        qformer_config_path = config.getoption("qformer_config_path", None)
        qformer_config_path = pop_value(
            qformer_config_path,
            nested_dict_value(pretrain_infos, "qformer", "config"),
        )
        qformer_config_path = cached_path(qformer_config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrain_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrain_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrain_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrain_infos, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        weight_path = config.getoption("pretrained_weight_path", None)
        device = config.getoption("device", "cpu")
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                    replace_keys={
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text", "weight"),
                    replace_keys={
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "vae", "weight"),
                    replace_keys={
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "qformer", "weight"),
                    prefix_keys={"": "qformer."},
                ),
            ]

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            qformer_config_path=qformer_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            state_dict=state_dict,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/blip2/text2image")
    def __call__(
        self,
        text: str,
        refer_text: str,
        refer_image: Union[Image.Image, str],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        inputs = self.processor.text2image_inputs(text, refer_text, refer_image)
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)
        self.seed = seed
        outputs = self.generate(
            **inputs,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        images = numpy_to_pil(outputs.images.cpu().numpy())
        return images[0]
