# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import warnings
import logging
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from megatron.core import mpu
from megatron.core import dist_checkpointing
from unitorch.utils import (
    replace,
    load_weight,
    is_diffusers_available,
)
from unitorch.models import CheckpointMixin


class MegatronCheckpointMixin(CheckpointMixin):
    checkpoint_name = "common.pt"

    modules_to_save_checkpoints = []

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: str = None,
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

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        sharded_state_dict = {}
        for module in self.modules_to_save_checkpoints:
            _module = getattr(self, module)
            sharded_state_dict.update(_module.sharded_state_dict())

        dist_checkpointing.save(sharded_state_dict, ckpt_dir)

        logging.info(f"{type(self).__name__} model save checkpoint to {ckpt_dir}")

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

        sharded_state_dict = {}
        for module in self.modules_to_save_checkpoints:
            _module = getattr(self, module)
            sharded_state_dict.update(_module.sharded_state_dict())

        checkpoint = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_dir
        )
        self.load_state_dict(checkpoint, strict=False)
        logging.info(
            f"{type(self).__name__} model load weight from checkpoint {ckpt_dir}"
        )


class GenericMegatronModel(nn.Module, MegatronCheckpointMixin):
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


from unitorch.models.megatron.modeling_gpt import MegatronGPTForGeneration
