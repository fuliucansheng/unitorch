# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.bloom import (
    BloomForGeneration as _BloomForGeneration,
)
from unitorch.models.bloom import BloomProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.bloom import pretrained_bloom_infos


class BloomForGenerationPipeline(_BloomForGeneration):
    def __init__(
        self,
        config_path: str,
        tokenizer_file: str,
        max_seq_length: Optional[int] = 512,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = BloomProcessor(
            tokenizer_file=tokenizer_file,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/bloom")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "default-bloom",
        config_path: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/bloom")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        tokenizer_file = config.getoption("tokenizer_file", tokenizer_file)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        max_seq_length = config.getoption("max_seq_length", 512)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            tokenizer_file=tokenizer_file,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/bloom")
    def __call__(
        self,
        prompt: str,
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
        inputs = self.processor.prompt(
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

        return decoded[0].strip()
