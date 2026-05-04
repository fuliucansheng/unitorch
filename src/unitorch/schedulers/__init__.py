# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
import os
import re
from typing import Dict, List, Optional, Union

import safetensors
import torch

from unitorch.utils import load_weight, replace
from unitorch.schedulers.warmup import CosineWarmupScheduler, LinearWarmupScheduler


class SchedulerCheckpointMixin:
    """Mixin that adds checkpoint save/load and pretrained-weight loading to schedulers."""

    checkpoint_name: str = "pytorch_scheduler.bin"
    replace_keys_in_state_dict: Dict[str, str] = {}
    prefix_keys_in_state_dict: Dict[str, str] = {}

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Load scheduler state from a checkpoint directory.

        Args:
            ckpt_dir: Path to the checkpoint directory.
            weight_name: Filename of the weight file; defaults to
                         :attr:`checkpoint_name`.
        """
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
        """Save the current scheduler state to a checkpoint directory.

        Args:
            ckpt_dir: Path to the checkpoint directory.
            weight_name: Filename of the weight file; defaults to
                         :attr:`checkpoint_name`.
        """
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
        weight_path: Union[str, List[str], None] = None,
        state_dict: Union[Dict, List[Dict], None] = None,
        replace_keys: Optional[Dict[str, str]] = None,
        prefix_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """Load pretrained weights into the scheduler.

        Args:
            weight_path: Path(s) to pretrained weight file(s).
            state_dict: Pre-loaded state dict(s) to merge in.
            replace_keys: Regex → replacement mapping applied to state-dict keys.
            prefix_keys: Regex → prefix mapping prepended to matching keys.
        """
        assert weight_path or state_dict, "weight_path or state_dict must be provided."

        replace_keys = {**self.replace_keys_in_state_dict, **(replace_keys or {})}
        prefix_keys = {**self.prefix_keys_in_state_dict, **(prefix_keys or {})}

        state_dicts: List[Dict] = []
        if weight_path:
            if isinstance(weight_path, str):
                weight_path = [weight_path]
            for path in weight_path:
                logging.debug("Loading weights from %s", path)
                state_dicts.append(load_weight(path))
        if state_dict:
            state_dicts += state_dict if isinstance(state_dict, list) else [state_dict]

        self_state_dict = self.state_dict()
        load_keys: List[str] = []

        for sd in state_dicts:
            if not sd:
                continue
            for key, value in sd.items():
                for pattern, prefix in prefix_keys.items():
                    if re.match(pattern, key):
                        key = prefix + key
                        break
                for pattern, replacement in replace_keys.items():
                    key = re.sub(pattern, replacement, key)

                if key in self_state_dict and value.shape == self_state_dict[key].shape:
                    self_state_dict[key] = value
                    if key not in load_keys:
                        load_keys.append(key)
                else:
                    logging.debug(
                        "Key %s (shape %s) does not match model's state_dict (shape %s).",
                        key,
                        value.shape,
                        self_state_dict.get(key, torch.empty(0)).shape,
                    )

        self.load_state_dict(self_state_dict, strict=False)
        load_percent = len(load_keys) / len(self_state_dict) * 100

        missed_keys = set(self_state_dict.keys()) - set(load_keys)
        for key in missed_keys:
            logging.debug(
                "%s: key '%s' (shape %s) not found in pretrained weights.",
                type(self).__name__, key, self_state_dict[key].shape,
            )

        logging.info("%s loaded weights (%.0f%%)", type(self).__name__, load_percent)
