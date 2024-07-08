# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import fire
import json
import torch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_controlnet_checkpoint,
    convert_ldm_bert_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
    convert_paint_by_example_checkpoint,
)
from unitorch.cli import cached_path
from unitorch.cli.models.diffusion_utils import load_weight


def stable_clip(
    config_file: str,
    checkpoint_file: str,
    output_checkpoint_file: str,
    base_checkpoint_file: Optional[str] = None,
    is_original_base_checkpoint: bool = False,
):
    def process_name(name: str) -> str:
        name = re.sub(r"lora_te_|lora_up\.|lora_down\.", "", name)
        name = re.sub(
            r"(_model|_attn|_proj|)|_",
            lambda m: "." if m.group(1) is None else m.group(0),
            name,
        )
        return name

    config_file = cached_path(config_file)
    checkpoint_file = cached_path(checkpoint_file)

    config = CLIPTextConfig.from_json_file(config_file)
    text = CLIPTextModel(config)
    text_state_dict = text.state_dict()
    state_dict = {}
    if base_checkpoint_file is not None:
        base_checkpoint_file = cached_path(base_checkpoint_file)
        state_dict = load_weight(base_checkpoint_file)
        if is_original_base_checkpoint:
            state_dict = convert_ldm_clip_checkpoint(
                state_dict, text_encoder=text
            ).state_dict()

    for k, v in state_dict.items():
        if "position_ids" in k:
            continue
        assert k in text_state_dict, f"base checkpoint error, {k} not in text"
        assert (
            text_state_dict[k].shape == v.shape
        ), f"base checkpoint error, {k} shape mismatch, {text_state_dict[k].shape} vs {v.shape}"

    original_state_dict = load_weight(checkpoint_file)
    for k, v in original_state_dict.items():
        if any(x in k for x in ["lora_down", "unet", "conv", "time", "alpha"]):
            continue
        new_k = process_name(k)
        if "lora_up" in k:
            v_up, v_down = v, original_state_dict[k.replace("lora_up", "lora_down")]
            alpha_name = k.replace("lora_up.", "").replace(".weight", ".alpha")
            alpha = original_state_dict[alpha_name].item()
            scale = v_up.shape[1] / alpha
            if v_up.dim() == 2:
                v = torch.matmul(v_up, v_down)
            elif v_up.dim() == 4:
                v = torch.einsum("bchw,cdhw->bdhw", v_up, v_down)
            else:
                assert False, f"dim {v_up.dim()} not supported"
            v = v * scale
        if new_k in text_state_dict:
            assert (
                text_state_dict[new_k].shape == v.shape
            ), f"{new_k} shape mismatch, {text_state_dict[new_k].shape} vs {v.shape}"
            if new_k in state_dict:
                state_dict[new_k] = state_dict[new_k] + v
            else:
                state_dict[new_k] = v
        else:
            logging.warning(f"Skip {new_k} in {checkpoint_file}")


def stable_open_clip(
    config_file: str,
    checkpoint_file: str,
    output_checkpoint_file: str,
    base_checkpoint_file: Optional[str] = None,
    is_original_base_checkpoint: bool = False,
):
    pass


def stable_unet(
    config_file: str,
    checkpoint_file: str,
    output_checkpoint_file: str,
    base_checkpoint_file: Optional[str] = None,
    is_original_base_checkpoint: bool = False,
):
    def process_name(name: str) -> str:
        name = re.sub(r"lora_unet_|lora_up\.|lora_down\.", "", name)
        name = re.sub(
            r"(_block|_k|_v|_q|_in|_out|)|_",
            lambda m: "." if m.group(1) is None else m.group(0),
            name,
        )
        return name

    config_file = cached_path(config_file)
    checkpoint_file = cached_path(checkpoint_file)

    config_dict = json.load(open(config_file))
    unet = UNet2DConditionModel.from_config(config_dict)
    unet_state_dict = unet.state_dict()
    state_dict = {}
    if base_checkpoint_file is not None:
        base_checkpoint_file = cached_path(base_checkpoint_file)
        state_dict = load_weight(base_checkpoint_file)
        if is_original_base_checkpoint:
            has_label_emb = any("label_emb" in k for k in state_dict.keys())
            config_dict["class_embed_type"] = "timestep" if has_label_emb else None
            config_dict["addition_embed_type"] = "text_time" if has_label_emb else None
            state_dict = convert_ldm_unet_checkpoint(state_dict, config=config_dict)

    for k, v in state_dict.items():
        assert k in unet_state_dict, f"base checkpoint error, {k} not in unet"
        assert (
            unet_state_dict[k].shape == v.shape
        ), f"base checkpoint error, {k} shape mismatch, {unet_state_dict[k].shape} vs {v.shape}"

    original_state_dict = load_weight(checkpoint_file)
    for k, v in original_state_dict.items():
        if any(x in k for x in ["lora_down", "text_model", "alpha"]):
            continue
        new_k = process_name(k)
        if "lora_up" in k:
            v_up, v_down = v, original_state_dict[k.replace("lora_up", "lora_down")]
            alpha_name = k.replace("lora_up.", "").replace(".weight", ".alpha")
            alpha = original_state_dict[alpha_name].item()
            scale = v_up.shape[1] / alpha
            if v_up.dim() == 2:
                v = torch.matmul(v_up, v_down)
            elif v_up.dim() == 4:
                v = torch.einsum("bchw,cdhw->bdhw", v_up, v_down)
            else:
                assert False, f"dim {v_up.dim()} not supported"
            v = v * scale
        if new_k in unet_state_dict:
            assert (
                unet_state_dict[new_k].shape == v.shape
            ), f"{new_k} shape mismatch, {unet_state_dict[new_k].shape} vs {v.shape}"
            if new_k in state_dict:
                state_dict[new_k] = state_dict[new_k] + v
            else:
                state_dict[new_k] = v
        else:
            logging.warning(f"Skip {new_k} in {checkpoint_file}")
    torch.save(state_dict, output_checkpoint_file)


def stable_vae(
    config_file: str,
    checkpoint_file: str,
    output_checkpoint_file: str,
    base_checkpoint_file: Optional[str] = None,
    is_original_base_checkpoint: bool = False,
):
    pass


def controlnet(
    config_file: str,
    checkpoint_file: str,
    output_checkpoint_file: str,
    base_checkpoint_file: Optional[str] = None,
    is_original_base_checkpoint: bool = False,
):
    pass


if __name__ == "__main__":
    fire.Fire()
