# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import Qwen2Tokenizer, AddedToken, Qwen2TokenizerFast
from unitorch.utils import (
    pop_value,
    truncate_sequence_pair,
    read_json_file,
    get_added_token,
)
from unitorch.models import (
    HfTextClassificationProcessor,
    HfLlmProcessor,
    HfTextGenerationProcessor,
    GenericOutputs,
)


class QWenProcessor(HfLlmProcessor):
    def __init__(
        self,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        max_gen_seq_length: Optional[int] = 512,
    ):
        """
        Initializes the ClipProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to 262144.
        """
        tokenizer_config = read_json_file(tokenizer_config) if tokenizer_config else {}
        special_tokens_map = (
            read_json_file(special_tokens_map) if special_tokens_map else {}
        )
        added_tokens_decoder = tokenizer_config.pop("added_tokens_decoder", {})
        tokenizer_config = {
            k: (
                get_added_token(v)
                if isinstance(v, dict) and v.get("__type") == "AddedToken"
                else v
            )
            for k, v in tokenizer_config.items()
        }
        tokenizer = Qwen2TokenizerFast(
            tokenizer_file=tokenizer_file,
            **tokenizer_config,
        )
        for idx, spec in added_tokens_decoder.items():
            token = spec["content"]
            tokenizer.added_tokens_decoder[idx] = get_added_token(spec)
            tokenizer.added_tokens_encoder[token] = idx

        special_tokens = {}
        for name, spec in special_tokens_map.items():
            special_tokens[name] = get_added_token(spec)
        tokenizer.add_special_tokens(special_tokens)
        if chat_template:
            tokenizer.chat_template = read_json_file(chat_template)["chat_template"]
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
