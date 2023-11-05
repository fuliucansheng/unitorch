# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.minigpt4 import (
    MiniGPT4Blip2LlamaForGeneration as _MiniGPT4Blip2LlamaForGeneration,
)
from unitorch.models.minigpt4 import MiniGPT4Blip2LlamaProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos


class MiniGPT4Blip2LlamaForGenerationPipeline(_MiniGPT4Blip2LlamaForGeneration):
    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        vocab_path: str,
        vision_config_path: str,
        quant_config_path: Optional[str] = None,
        pad_token_id: Optional[int] = 0,
        max_seq_length: Optional[int] = 192,
        max_prefix_seq_length: Optional[int] = 64,
        max_suffix_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        if device == "cpu":
            quant_config_path = None
        super().__init__(
            blip2_config_path=blip2_config_path,
            llama_config_path=llama_config_path,
            quant_config_path=quant_config_path,
            pad_token_id=pad_token_id,
        )
        self.processor = MiniGPT4Blip2LlamaProcessor(
            vocab_file=vocab_path,
            vision_config_path=vision_config_path,
            max_prefix_seq_length=max_prefix_seq_length,
            max_suffix_seq_length=max_suffix_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/pipeline/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")

        blip2_config_path = config.getoption("blip2_config_path", None)
        blip2_config_path = pop_value(
            blip2_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "blip2_config_path"
            ),
        )
        blip2_config_path = cached_path(blip2_config_path)

        llama_config_path = config.getoption("llama_config_path", None)
        llama_config_path = pop_value(
            llama_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "llama_config_path"
            ),
        )
        llama_config_path = cached_path(llama_config_path)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 192)
        max_prefix_seq_length = config.getoption("max_prefix_seq_length", 64)
        max_suffix_seq_length = config.getoption("max_suffix_seq_length", 128)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        device = config.getoption("device", "cpu")
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            blip2_config_path,
            llama_config_path,
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            quant_config_path=quant_config_path,
            max_seq_length=max_seq_length,
            max_prefix_seq_length=max_prefix_seq_length,
            max_suffix_seq_length=max_suffix_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 192,
        max_prefix_seq_length: Optional[int] = 64,
        max_suffix_seq_length: Optional[int] = 128,
        num_beams: Optional[int] = 2,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = [835, 2277],
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
        prefix_text = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>"
        suffix_text = "</Img> {0} ###Assistant:".format(prompt)
        inputs = self.processor.prompt(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            image=image,
            max_prefix_seq_length=max_prefix_seq_length,
            max_suffix_seq_length=max_suffix_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        outputs = super().generate(
            pixel_values=inputs["pixel_values"],
            prefix_input_ids=inputs["prefix_input_ids"],
            suffix_input_ids=inputs["suffix_input_ids"],
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
        cleanup_string = lambda text: re.sub(r"###|\n", " ", text)
        if isinstance(decoded[0], list):
            decoded = [list(map(cleanup_string, sequence)) for sequence in decoded]
        elif isinstance(decoded[0], str):
            decoded = list(map(cleanup_string, decoded))

        return decoded[0]
