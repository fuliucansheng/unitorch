# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import is_remote_url
from unitorch.models.llava import (
    LlavaMistralClipForGeneration as _LlavaMistralClipForGeneration,
)
from unitorch.models.llava import LlavaMistralClipProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.llava import (
    pretrained_llava_infos,
    pretrained_llava_extensions_infos,
)


class LlavaMistralClipForGenerationPipeline(_LlavaMistralClipForGeneration):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        vocab_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        max_gen_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        if device == "cpu":
            quant_config_path = None
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = LlavaMistralClipProcessor(
            vocab_file=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/llava/mistral_clip")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "default-llava-v1.6",
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/llava/mistral_clip")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        quant_config_path = config.getoption("quant_config_path", quant_config_path)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 512)
        max_gen_seq_length = config.getoption("max_gen_seq_length", 512)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path=vision_config_path,
            vocab_path=vocab_path,
            quant_config_path=quant_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/llava/mistral_clip")
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
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        lora_urls: Optional[Union[str, List[str]]] = None,
        lora_files: Optional[Union[str, List[str]]] = None,
    ):
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

        if lora_checkpoints is not None:
            _lora_checkpoints = list(
                zip(
                    *[
                        (ckpt, weight, alpha)
                        for ckpt, weight, alpha in zip(
                            lora_checkpoints, lora_weights, lora_alphas
                        )
                        if ckpt is not None
                    ]
                )
            )
            if len(_lora_checkpoints) == 0:
                lora_checkpoints = None
            else:
                lora_checkpoints, lora_weights, lora_alphas = _lora_checkpoints
                lora_checkpoints = [
                    nested_dict_value(pretrained_llava_extensions_infos, ckpt)
                    for ckpt in lora_checkpoints
                ]
        if lora_urls is not None:
            _lora_urls = list(
                zip(
                    *[
                        (url, weight, alpha)
                        for url, weight, alpha in zip(
                            lora_urls, lora_weights, lora_alphas
                        )
                        if is_remote_url(url)
                    ]
                )
            )
            if len(_lora_urls) == 0:
                lora_urls = None
            else:
                lora_urls, lora_weights, lora_alphas = _lora_urls
        if lora_files is not None:
            _lora_files = list(
                zip(
                    *[
                        (file, weight, alpha)
                        for file, weight, alpha in zip(
                            lora_files, lora_weights, lora_alphas
                        )
                        if file is not None
                    ]
                )
            )
            if len(_lora_files) == 0:
                lora_files = None
            else:
                lora_files, lora_weights, lora_alphas = _lora_files

        lora_files = pop_value(
            lora_checkpoints,
            lora_urls,
            lora_files,
            check_none=False,
        )
        if lora_files is not None:
            self.load_lora_weights(
                lora_files, lora_weights=lora_weights, lora_alphas=lora_alphas
            )
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
        self.unload_lora_weights()
        decoded = self.processor.detokenize(outputs.sequences)

        return decoded[0].strip()
