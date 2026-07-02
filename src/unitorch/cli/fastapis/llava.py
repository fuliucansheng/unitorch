# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile
from unitorch.utils import is_remote_url
from unitorch.models.llava import (
    LlavaMistralClipForGeneration as _LlavaMistralClipForGeneration,
    LlavaLlamaSiglipForGeneration as _LlavaLlamaSiglipForGeneration,
)
from unitorch.models.llava import LlavaMistralClipProcessor, LlavaLlamaSiglipProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import cached_path, config_defaults_init, config_defaults_method
from unitorch.cli.models.llava import (
    pretrained_llava_infos,
    pretrained_llava_extensions_infos,
)
from unitorch.cli import register_fastapi
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli import PipelineReplicaPool


class LlavaMistralClipForGenerationPipeline(_LlavaMistralClipForGeneration):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):

        super().__init__(
            config_path=config_path,
        )
        self.processor = LlavaMistralClipProcessor(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/llava/mistral_clip")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/llava/mistral_clip")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "llava-v1.6-mistral-7b-hf"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        tokenizer_file = tokenizer_file or config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(
                pretrained_llava_infos, pretrained_name, "tokenizer_config"
            ),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(
                pretrained_llava_infos, pretrained_name, "special_tokens_map"
            ),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template) if chat_template is not None else None
        )

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        max_seq_length = config.getoption("max_seq_length", 512)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/llava/mistral_clip")
    def __call__(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        max_seq_length: Optional[int] = 512,
        num_beams: Optional[int] = 2,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = [2],
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 512,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.generation_inputs(
            text=prompt,
            image=image,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }

        outputs = super().generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        decoded = self.processor.detokenize(outputs.sequences)
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return decoded[0]


class LlavaLlamaSiglipForGenerationPipeline(_LlavaLlamaSiglipForGeneration):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):

        super().__init__(
            config_path=config_path,
        )
        self.processor = LlavaLlamaSiglipProcessor(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/llava/llama_siglip")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/llava/llama_siglip")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "llava-v1.6-joycaption-2"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        tokenizer_file = tokenizer_file or config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(
                pretrained_llava_infos, pretrained_name, "tokenizer_config"
            ),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(
                pretrained_llava_infos, pretrained_name, "special_tokens_map"
            ),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template) if chat_template is not None else None
        )

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        max_seq_length = config.getoption("max_seq_length", 128)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path=vision_config_path,
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/llava/llama_siglip")
    def __call__(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        max_seq_length: Optional[int] = 128,
        num_beams: Optional[int] = 2,
        decoder_start_token_id: Optional[int] = 128000,
        decoder_end_token_id: Optional[Union[int, List[int]]] = [
            128001,
            128008,
            128009,
        ],
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 512,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.generation_inputs(
            text=prompt,
            image=image,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        outputs = super().generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        decoded = self.processor.detokenize(outputs.sequences)
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()

        return decoded[0]


@register_fastapi("core/fastapi/llava/mistral_clip")
class LlavaMistralClipFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section(f"core/fastapi/llava/mistral_clip")
        self._section = f"core/fastapi/llava/mistral_clip"
        router = config.getoption("router", "/core/fastapi/llava/mistral_clip")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self):
        num_replicas = int(
            self.config.getdefault(self._section, "pipeline_num_replicas", 1)
        )
        devices = self.config.getdefault(
            self._section, "pipeline_replica_devices", "cpu"
        )
        lock = self.config.getdefault(
            self._section, "pipeline_replica_lock", True
        )
        if devices is None:
            devices = []
        if isinstance(devices, str):
            devices = [devices] * num_replicas
        pipelines = [
            LlavaMistralClipForGenerationPipeline.from_config(
                self.config,
                pretrained_name="llava-v1.6-mistral-7b-hf",
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        text: str,
        image: UploadFile,
    ):
        assert self._pipes is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = f"[INST] <image>\n {text} [/INST]"
        pipe = self._pipes.acquire()
        try:
            caption = pipe(
                text,
                image,
                
            )
        finally:
            pipe.release()
        return caption


@register_fastapi("core/fastapi/llava/joycaption2")
class LlavaLlamaSiglipFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section(f"core/fastapi/llava/joycaption2")
        self._section = f"core/fastapi/llava/joycaption2"
        router = config.getoption("router", "/core/fastapi/llava/joycaption2")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self):
        num_replicas = int(
            self.config.getdefault(self._section, "pipeline_num_replicas", 1)
        )
        devices = self.config.getdefault(
            self._section, "pipeline_replica_devices", "cpu"
        )
        lock = self.config.getdefault(
            self._section, "pipeline_replica_lock", True
        )
        if devices is None:
            devices = []
        if isinstance(devices, str):
            devices = [devices] * num_replicas
        pipelines = [
            LlavaLlamaSiglipForGenerationPipeline.from_config(
                self.config,
                pretrained_name="llava-v1.6-joycaption-2",
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        text: str,
        image: UploadFile,
    ):
        assert self._pipes is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = f"<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 26 July 2024\\n\\nYou are a helpful image captioner.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>{text}|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        pipe = self._pipes.acquire()
        try:
            caption = pipe(
                text,
                image,
                
            )
        finally:
            pipe.release()
        return caption
