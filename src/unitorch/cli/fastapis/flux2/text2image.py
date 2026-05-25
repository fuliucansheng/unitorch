# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import gc
import io
import asyncio
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image
from torch import autocast

from diffusers import Flux2Pipeline

from unitorch.models.diffusers import Flux2Processor, GenericFlux2Model
from unitorch.utils import is_bfloat16_available, nested_dict_value, pop_value
from unitorch.cli import (
    Config,
    GenericFastAPI,
    config_defaults_init,
    config_defaults_method,
    register_fastapi,
)
from unitorch.cli.models.diffusers import (
    load_weight,
    pretrained_stable_extensions_infos,
    pretrained_stable_infos,
)
from unitorch.cli.models.diffusers.modeling_flux2 import (
    _flux2_model_kwargs,
    _flux2_state_dict,
)


class Flux2FastAPIPipeline(GenericFlux2Model):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        processor_name_or_path: str,
        processor_subfolder: Optional[str] = "tokenizer",
        max_sequence_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        text_encoder_out_layers: Optional[tuple] = (10, 20, 30),
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        self.processor = Flux2Processor(
            processor_name_or_path=processor_name_or_path,
            processor_subfolder=processor_subfolder,
            vae_config_path=vae_config_path,
            max_seq_length=max_sequence_length,
            use_auth_token=use_auth_token,
        )
        self.max_sequence_length = max_sequence_length
        self._device = "cpu" if device == "cpu" else int(device)

        if state_dict is not None:
            self.from_pretrained(state_dict=state_dict)
        else:
            self.from_pretrained(weight_path)
        self.eval()

        self.pipeline = Flux2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text,
            tokenizer=self.processor.processor,
            transformer=self.transformer,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload
        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

    @classmethod
    def _from_config_section(
        cls,
        config,
        section: str,
        pretrained_name: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        processor_name_or_path: Optional[str] = None,
        processor_subfolder: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        model_kwargs = _flux2_model_kwargs(
            config,
            section,
            pretrained_name=pretrained_name,
        )
        pretrained_infos = model_kwargs.pop("pretrained_infos")
        use_auth_token = model_kwargs.pop("use_auth_token")

        processor_name_or_path = processor_name_or_path or config.getoption(
            "processor_name_or_path", None
        )
        processor_name_or_path = pop_value(
            processor_name_or_path,
            nested_dict_value(pretrained_infos, "processor", "name"),
        )

        processor_subfolder = processor_subfolder or config.getoption(
            "processor_subfolder", None
        )
        processor_subfolder = pop_value(
            processor_subfolder,
            nested_dict_value(pretrained_infos, "processor", "subfolder"),
            check_none=False,
        )

        max_sequence_length = config.getoption("max_sequence_length", 512)
        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)

        state_dict = None
        if weight_path is None:
            state_dict = _flux2_state_dict(pretrained_infos, use_auth_token)
        else:
            state_dict = load_weight(weight_path, use_auth_token=use_auth_token)

        pretrained_lora_names = pretrained_lora_names or config.getoption(
            "pretrained_lora_names", None
        )
        pretrained_lora_weights = pretrained_lora_weights or config.getoption(
            "pretrained_lora_weights", 1.0
        )
        pretrained_lora_alphas = pretrained_lora_alphas or config.getoption(
            "pretrained_lora_alphas", 32.0
        )

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos,
                    name,
                    "lora",
                    "weight",
                )
                for name in pretrained_lora_names
            ]

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
        )

        return cls(
            config_path=model_kwargs["config_path"],
            text_config_path=model_kwargs["text_config_path"],
            vae_config_path=model_kwargs["vae_config_path"],
            scheduler_config_path=model_kwargs["scheduler_config_path"],
            processor_name_or_path=processor_name_or_path,
            processor_subfolder=processor_subfolder,
            max_sequence_length=max_sequence_length,
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            use_auth_token=use_auth_token,
            text_encoder_out_layers=model_kwargs["text_encoder_out_layers"],
        )

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/flux2/text2image")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        processor_name_or_path: Optional[str] = None,
        processor_subfolder: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        return cls._from_config_section(
            config,
            "core/fastapi/pipeline/flux2/text2image",
            pretrained_name=pretrained_name,
            pretrained_weight_path=pretrained_weight_path,
            processor_name_or_path=processor_name_or_path,
            processor_subfolder=processor_subfolder,
            device=device,
            pretrained_lora_names=pretrained_lora_names,
            pretrained_lora_weights_path=pretrained_lora_weights_path,
            pretrained_lora_weights=pretrained_lora_weights,
            pretrained_lora_alphas=pretrained_lora_alphas,
            **kwargs,
        )

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    @config_defaults_method("core/fastapi/pipeline/flux2/text2image")
    def __call__(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        caption_upsample_temperature: Optional[float] = None,
    ):
        generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
        outputs = self.pipeline(
            image=image,
            prompt=text,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            output_type="pil",
            max_sequence_length=self.max_sequence_length,
            caption_upsample_temperature=caption_upsample_temperature,
        )
        return outputs.images[0]


@register_fastapi("core/fastapi/flux2/text2image")
class Flux2Text2ImageFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/flux2/text2image")
        router = config.getoption("router", "/core/fastapi/flux2/text2image")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["GET"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["POST"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(
        self,
        pretrained_name: Optional[str] = "flux2-dev",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
        self._pipe = Flux2FastAPIPipeline.from_config(
            self.config,
            pretrained_name=pretrained_name,
            pretrained_lora_names=pretrained_lora_names,
            pretrained_lora_weights=pretrained_lora_weights,
            pretrained_lora_alphas=pretrained_lora_alphas,
        )
        return "start success"

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        return "stop success"

    def status(self):
        return "running" if self._pipe is not None else "stopped"

    async def generate(
        self,
        text: str,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        caption_upsample_temperature: Optional[float] = None,
    ):
        assert self._pipe is not None
        async with self._lock:
            image = self._pipe(
                text=text,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
                caption_upsample_temperature=caption_upsample_temperature,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
