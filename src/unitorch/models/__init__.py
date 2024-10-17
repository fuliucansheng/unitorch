# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import json
import torch
import logging
import torch.nn as nn
import safetensors
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict
from transformers.utils import is_remote_url, ModelOutput as GenericOutputs
from unitorch import hf_cached_path
from unitorch.utils import replace, load_weight, is_diffusers_available


class CheckpointMixin:
    checkpoint_name = "pytorch_model.bin"
    replace_keys_in_state_dict = {}
    prefix_keys_in_state_dict = {}

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load model weights from a checkpoint.

        Args:
            ckpt_dir (str): Directory path of the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return

        if weight_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(weight_path)
        else:
            state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict)
        logging.info(
            f"{type(self).__name__} model load weight from checkpoint {weight_path}"
        )

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Save the model's current state as a checkpoint.

        Args:
            ckpt_dir (str): Directory path to save the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        state_dict = self.state_dict()
        weight_path = os.path.join(ckpt_dir, weight_name)
        if weight_path.endswith(".safetensors"):
            safetensors.torch.save_file(state_dict, weight_path)
        else:
            torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} model save checkpoint to {weight_path}")

    def from_pretrained(
        self,
        weight_path: Union[str, List[str]] = None,
        state_dict: Union[Dict, List[Dict]] = None,
        replace_keys: Optional[Dict] = dict(),
        prefix_keys: Optional[Dict] = dict(),
    ):
        """
        Load pretrained weights into the model.

        Args:
            weight_path (str or List[str]): Path(s) to the pretrained weight file(s).
            state_dict (Dict or List[Dict]): Pretrained state_dict(s) to load weights from.
            replace_keys (Dict): Dictionary mapping keys in the pretrained state_dict to the model's keys.
            prefix_keys (Dict): Dictionary prefix keys in the pretrained state_dict to the model's keys.

        Returns:
            None
        """
        assert weight_path or state_dict, "weight_path or state_dict must be set"

        # Load state_dict(s) based on the provided weight_path or state_dict
        state_dicts = []
        if weight_path:
            if isinstance(weight_path, str):
                weight_path = [weight_path]
            for path in weight_path:
                logging.debug(f"Loading weights from {path}")
            state_dicts += [load_weight(path) for path in weight_path]

        if state_dict:
            state_dicts += state_dict if isinstance(state_dict, list) else [state_dict]

        self_state_dict = self.state_dict()  # Get the current state_dict of the model
        load_keys = []  # Keep track of the keys loaded from the state_dict(s)
        non_load_keys = []  # Keep track of the keys not loaded from the state_dict(s)

        if isinstance(self.replace_keys_in_state_dict, dict):
            replace_keys = {**self.replace_keys_in_state_dict, **replace_keys}

        if isinstance(self.prefix_keys_in_state_dict, dict):
            prefix_keys = {**self.prefix_keys_in_state_dict, **prefix_keys}

        # Iterate over the state_dict(s) and load the matching keys into the model's state_dict
        for _state_dict in state_dicts:
            if not _state_dict:
                continue
            for key, value in list(_state_dict.items()):
                for rkey, prefix in prefix_keys.items():
                    if re.match(rkey, key):
                        key = prefix + key
                        break

                for rkey, nkey in replace_keys.items():
                    key = re.sub(rkey, nkey, key)
                if key in self_state_dict and value.shape == self_state_dict[key].shape:
                    self_state_dict[key] = value
                    if key not in load_keys:
                        load_keys.append(key)
                else:
                    non_load_keys.append(key)

        self.load_state_dict(self_state_dict, False)
        load_percent = (
            len(load_keys) / len(self_state_dict) * 100
        )  # Calculate the percentage of loaded keys
        logging.debug(f"Non load keys in pretrain weights: {list(non_load_keys)}")
        logging.debug(
            f"{type(self).__name__} missed keys: {list(self_state_dict.keys() - load_keys)}"
        )
        logging.info(f"{type(self).__name__} loaded weights ({int(load_percent)}%)")


class GenericModel(nn.Module, CheckpointMixin):
    def __init__(self):
        super().__init__()
        pass

    def _init_weights(self, module):
        """
        Initialize the weights of the given module.

        Args:
            module (nn.Module): The module to initialize weights for.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        self.apply(self._init_weights)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the model's parameters.

        Returns:
            torch.dtype: The data type of the model's parameters.
        """
        return next(self.parameters()).dtype

    @property
    def device(self):
        """
        Returns the device of the model's parameters.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self.parameters()).device


from unitorch.models.processing_utils import (
    HfTextGenerationProcessor,
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
)
from unitorch.models.modeling_ema import ExponentialMovingAverage
from unitorch.models.quantization import QuantizationConfig, QuantizationMixin
from unitorch.models.onnx import GenericOnnxModel

# import models
import unitorch.models.bart
import unitorch.models.beit
import unitorch.models.bert
import unitorch.models.blip
import unitorch.models.bloom
import unitorch.models.chinese_clip
import unitorch.models.clip
import unitorch.models.dpt
import unitorch.models.dinov2
import unitorch.models.grounding_dino
import unitorch.models.llama
import unitorch.models.llava
import unitorch.models.mask2former
import unitorch.models.mbart
import unitorch.models.mistral
import unitorch.models.mt5
import unitorch.models.onnx
import unitorch.models.pegasus
import unitorch.models.roberta
import unitorch.models.segformer
import unitorch.models.swin
import unitorch.models.t5
import unitorch.models.visualbert
import unitorch.models.vit
import unitorch.models.xlm_roberta
import unitorch.models.xpegasus
import unitorch.models.peft

if is_diffusers_available():
    import unitorch.models.diffusers
