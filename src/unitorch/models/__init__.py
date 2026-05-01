# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
import os
import re
from typing import Dict, List, Optional, Union

import safetensors
import torch
import torch.nn as nn
from transformers.utils import ModelOutput as GenericOutputs

from unitorch import hf_cached_path
from unitorch.utils import (
    is_diffusers_available,
    is_megatron_available,
    load_weight,
    replace,
)


class CheckpointMixin:
    """Mixin that adds checkpoint save/load and pretrained-weight loading to a model."""

    checkpoint_name = "pytorch_model.bin"
    replace_keys_in_state_dict: Dict[str, str] = {}
    prefix_keys_in_state_dict: Dict[str, str] = {}

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Load model weights from *ckpt_dir*."""
        weight_name = weight_name or self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        if weight_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(weight_path)
        else:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict)
        logging.info("%s loaded weights from %s", type(self).__name__, weight_path)

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Save model weights to *ckpt_dir*."""
        weight_name = weight_name or self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        state_dict = self.state_dict()
        if weight_path.endswith(".safetensors"):
            safetensors.torch.save_file(state_dict, weight_path)
        else:
            torch.save(state_dict, weight_path)
        logging.info("%s saved checkpoint to %s", type(self).__name__, weight_path)

    def from_pretrained(
        self,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Union[Dict, List[Dict]]] = None,
        replace_keys: Optional[Dict[str, str]] = None,
        prefix_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """Load pretrained weights into the model.

        Args:
            weight_path: Path(s) to pretrained weight file(s).
            state_dict: Pretrained state dict(s) to load from.
            replace_keys: Regex substitution rules ``{pattern: replacement}``
                applied to each key before matching.
            prefix_keys: Regex prefix rules ``{pattern: prefix}`` — the first
                matching pattern prepends *prefix* to the key.
        """
        assert weight_path or state_dict, "weight_path or state_dict must be provided"

        replace_keys = {**self.replace_keys_in_state_dict, **(replace_keys or {})}
        prefix_keys = {**self.prefix_keys_in_state_dict, **(prefix_keys or {})}

        state_dicts: List[Dict] = []
        if weight_path:
            if isinstance(weight_path, str):
                weight_path = [weight_path]
            for path in weight_path:
                logging.debug("Loading weights from %s", path)
            state_dicts += [load_weight(p) for p in weight_path]
        if state_dict:
            state_dicts += state_dict if isinstance(state_dict, list) else [state_dict]

        self_state_dict = self.state_dict()
        load_keys: List[str] = []

        for sd in state_dicts:
            if not sd:
                continue
            for key, value in sd.items():
                for rkey, pfx in prefix_keys.items():
                    if re.match(rkey, key):
                        key = pfx + key
                        break
                for rkey, nkey in replace_keys.items():
                    key = re.sub(rkey, nkey, key)
                if key in self_state_dict and value.shape == self_state_dict[key].shape:
                    self_state_dict[key] = value
                    if key not in load_keys:
                        load_keys.append(key)
                else:
                    logging.debug(
                        "Key %s with shape %s does not match model shape %s",
                        key,
                        value.shape,
                        self_state_dict.get(key, torch.empty(0)).shape,
                    )

        self.load_state_dict(self_state_dict, strict=False)
        missed_keys = set(self_state_dict.keys()) - set(load_keys)
        for key in missed_keys:
            logging.debug(
                "%s key %s not in pretrained weights (shape %s)",
                type(self).__name__,
                key,
                self_state_dict[key].shape,
            )
        load_percent = len(load_keys) / len(self_state_dict) * 100 if self_state_dict else 0
        logging.info("%s loaded weights (%d%%)", type(self).__name__, int(load_percent))


class GenericModel(nn.Module, CheckpointMixin):
    """Base class for all unitorch models."""

    def __init__(self) -> None:
        super().__init__()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self) -> None:
        """Initialise all submodule weights with the default scheme."""
        self.apply(self._init_weights)

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the model's parameters."""
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Device of the model's parameters."""
        return next(self.parameters()).device


from unitorch.models.processing_utils import (
    HfImageClassificationProcessor,
    HfLlmProcessor,
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
)
from unitorch.models.modeling_ema import ExponentialMovingAverage
from unitorch.models.onnx import GenericOnnxModel

import unitorch.models.bart
import unitorch.models.beit
import unitorch.models.bert
import unitorch.models.blip
import unitorch.models.chinese_clip
import unitorch.models.clip
import unitorch.models.dinov2
import unitorch.models.dpt
import unitorch.models.grounding_dino
import unitorch.models.kolors
import unitorch.models.llama
import unitorch.models.llava
import unitorch.models.mask2former
import unitorch.models.mbart
import unitorch.models.mistral
import unitorch.models.onnx
import unitorch.models.pegasus
import unitorch.models.peft
import unitorch.models.qwen
import unitorch.models.roberta
import unitorch.models.segformer
import unitorch.models.siglip
import unitorch.models.swin
import unitorch.models.t5
import unitorch.models.visualbert
import unitorch.models.vit
import unitorch.models.xlm_roberta
import unitorch.models.xpegasus

if is_diffusers_available():
    import unitorch.models.diffusers

if is_megatron_available():
    import unitorch.models.megatron
