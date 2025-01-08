# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import math
import logging
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class ExponentialMovingAverage(nn.Module):
    """
    Exponential Moving Average (EMA) for model parameters.
    """

    checkpoint_name = "pytorch_ema_model.bin"

    def __init__(
        self,
        model,
        decay: Optional[float] = 0.9999,
        tau: Optional[int] = 2000,
        num_steps: Optional[int] = 0,
    ):
        """
        Initializes the ExponentialMovingAverage.

        Args:
            model (nn.Module): The model to apply EMA to.
            decay (float, optional): Decay rate for the EMA. Defaults to 0.9999.
            tau (int, optional): Time constant for the EMA. Defaults to 2000.
            num_steps (int, optional): Number of steps taken for the EMA. Defaults to 0.
        """
        super().__init__()
        self.model = deepcopy(model)
        self.num_steps = num_steps
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        for p in self.model.parameters():
            p.requires_grad = False

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
        state_dict = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
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
        state_dict = self.model.state_dict()
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} model save checkpoint to {weight_path}")

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the model.
        """
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def step(self, model):
        """
        Performs a step of EMA.

        Args:
            model (nn.Module): The model to update the EMA with.
        """
        self.num_steps += 1
        rate = self.decay(self.num_steps)

        new_state = model.state_dict()
        for key, value in self.model.state_dict().items():
            if not value.dtype.is_floating_point:
                continue
            value *= rate
            value += (1 - rate) * new_state[key].detach()
