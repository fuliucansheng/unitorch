# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter

from unitorch.utils import is_remote_url
from unitorch.models.gemma import GemmaForGeneration as _GemmaForGeneration
from unitorch.models.gemma import GemmaProcessor
from unitorch.utils import nested_dict_value
from unitorch.cli import config_defaults_init, config_defaults_method
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli import PipelineReplicaPool
from unitorch.cli import cached_path, register_fastapi
from unitorch.cli.models.gemma import (
    pretrained_gemma_extensions_infos,
    resolve_pretrained_gemma_path,
)


class GemmaForGenerationPipeline(_GemmaForGeneration):
    def __init__(
        self,
        config_path: str,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(config_path=config_path)
        self.processor = GemmaProcessor(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            chat_template=chat_template,
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
    @config_defaults_init("core/fastapi/pipeline/gemma")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/gemma")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "gemma-4-12b"
        )

        config_path = config_path or config.getoption("config_path", None)
        if config_path is None:
            config_path = resolve_pretrained_gemma_path(pretrained_name, "config")
        else:
            config_path = cached_path(config_path)

        tokenizer_file = tokenizer_file or config.getoption("tokenizer_file", None)
        if tokenizer_file is None:
            tokenizer_file = resolve_pretrained_gemma_path(pretrained_name, "tokenizer")
        else:
            tokenizer_file = cached_path(tokenizer_file)

        tokenizer_config = config.getoption("tokenizer_config", None)
        if tokenizer_config is None:
            tokenizer_config = resolve_pretrained_gemma_path(
                pretrained_name,
                "tokenizer_config",
            )
        else:
            tokenizer_config = cached_path(tokenizer_config)

        chat_template = config.getoption("chat_template", None)
        chat_template = cached_path(chat_template) if chat_template is not None else None

        max_seq_length = config.getoption("max_seq_length", 12800)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = (
            pretrained_weight_path
            if pretrained_weight_path is not None
            else resolve_pretrained_gemma_path(pretrained_name, "weight")
        )
        if pretrained_weight_path is not None:
            weight_path = cached_path(weight_path)

        return cls(
            config_path=config_path,
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/gemma")
    def __call__(
        self,
        prompt: str,
        use_chat_template: Optional[bool] = False,
        max_seq_length: Optional[int] = 12800,
        num_beams: Optional[int] = 2,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
        decoder_pad_token_id: Optional[int] = 0,
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

        if use_chat_template:
            prompt = self.processor.chat_template(messages=json.loads(prompt))

        inputs = self.processor.generation_inputs(
            text=prompt,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }

        outputs = super().generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            decoder_pad_token_id=decoder_pad_token_id,
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


@register_fastapi("core/fastapi/gemma")
class GemmaFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/gemma")
        self._section = "core/fastapi/gemma"
        router = config.getoption("router", "/core/fastapi/gemma")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, pretrained_name: str = "gemma-4-12b"):
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
            GemmaForGenerationPipeline.from_config(
                self.config,
                pretrained_name=pretrained_name,
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
        use_chat_template: Optional[bool] = False,
        max_seq_length: Optional[int] = 12800,
        num_beams: Optional[int] = 2,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
        decoder_pad_token_id: Optional[int] = 0,
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
        assert self._pipes is not None, "Service not started. Call /start first."
        pipe = self._pipes.acquire()
        try:
            outputs = pipe(
                prompt=text,
                use_chat_template=use_chat_template,
                max_seq_length=max_seq_length,
                num_beams=num_beams,
                decoder_start_token_id=decoder_start_token_id,
                decoder_end_token_id=decoder_end_token_id,
                decoder_pad_token_id=decoder_pad_token_id,
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
        finally:
            pipe.release()
        return outputs
