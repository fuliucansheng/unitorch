# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Strict, local loading of the native FLUX.3 base package and converted assets."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration


MODEL_ID = "black-forest-labs/flux-3-image-klein-base"
MODEL_REVISION = "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86"
DIFFUSERS_COMMIT = "b800f828f3951c4d206e154411bcbfa693cd322e"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    """Publish metadata only after its entire payload has reached disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".flux3-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def transformer_config_from_package(package):
    """Translate BFL's explicit DiT config using the pinned public conversion."""
    if package.get("model") != "flux-3-image-klein" or package.get("quantization") is not None:
        raise ValueError("Expected the native unquantized FLUX.3 image klein package")
    dit = package["dit"]
    if dit.get("flow_map", False) or dit.get("vec_in_dim") is not None or dit["out_modes"] != ["image"]:
        raise ValueError("Expected FLUX.3 base velocity model, not distilled or pooled-vector weights")
    if dit.get("image_rope_scale", 0.5) != 0.5:
        raise ValueError("Unsupported image RoPE scale")
    return dict(in_channels=dit["in_channels"]["image"], cond_in_channels=dit["in_channels"]["image_cond"],
                num_layers=dit["depth"], num_single_layers=dit["depth_single"],
                attention_head_dim=dit["hidden"] // dit["heads"], num_attention_heads=dit["heads"],
                joint_attention_dim=dit["in_channels"]["text"], mlp_ratio=dit["mlp_ratio"],
                axes_dims_rope=tuple(dit["axes_dim"]), rope_theta=dit["theta"], flow_map=False)


def convert_native_vae(state, num_blocks=4):
    """Map the FLUX.2 VAE used by FLUX.3; reject unknowns in the strict loader.

    Key mapping follows diffusers' scripts/convert_flux2_to_diffusers.py at
    DIFFUSERS_COMMIT (Apache-2.0, Hugging Face). Every input key is retained.
    """
    result = {}
    for key, value in state.items():
        name = key.replace("encoder.quant_conv.", "quant_conv.").replace(
            "decoder.post_quant_conv.", "post_quant_conv.")
        name = name.replace(".norm_out.", ".conv_norm_out.").replace(".nin_shortcut.", ".conv_shortcut.")
        name = re.sub(r"encoder.down.(\d+).block.", r"encoder.down_blocks.\1.resnets.", name)
        name = re.sub(r"encoder.down.(\d+).downsample.", r"encoder.down_blocks.\1.downsamplers.0.", name)
        name = re.sub(r"decoder.up.(\d+).block.",
                      lambda m: f"decoder.up_blocks.{num_blocks - 1 - int(m[1])}.resnets.", name)
        name = re.sub(r"decoder.up.(\d+).upsample.",
                      lambda m: f"decoder.up_blocks.{num_blocks - 1 - int(m[1])}.upsamplers.0.", name)
        name = re.sub(r".mid.block_(\d+).", lambda m: f".mid_block.resnets.{int(m[1]) - 1}.", name)
        name = name.replace(".mid.attn_1.", ".mid_block.attentions.0.")
        if ".attentions." in name:
            for old, new in (("norm", "group_norm"), ("q", "to_q"), ("k", "to_k"),
                             ("v", "to_v"), ("proj_out", "to_out.0")):
                name = name.replace(f".0.{old}.", f".0.{new}.")
            if value.ndim == 4:
                value = value[:, :, 0, 0]
        if name in result:
            raise ValueError(f"Duplicate converted VAE tensor: {name}")
        result[name] = value
    return result


def validate_state(state, expected, label="checkpoint"):
    """Validate every tensor before changing a single live parameter."""
    if set(state) != set(expected):
        raise ValueError(f"{label} coverage mismatch: missing={sorted(set(expected)-set(state))[:8]}, "
                         f"extra={sorted(set(state)-set(expected))[:8]}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"{label} tensor shape mismatch: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{label} tensor is not finite: {key}")


def _assign(module, state, dtype):
    expected = module.state_dict()
    # Older native VAE exports omit this unused training counter.
    if isinstance(module, AutoencoderKLFlux2) and "bn.num_batches_tracked" not in state:
        state["bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    validate_state(state, expected, type(module).__name__)
    module.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()},
                           strict=True, assign=True)
    return module


def _component_state(directory, stem):
    directory = Path(directory)
    index = directory / f"{stem}.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else [f"{stem}.safetensors"]
    state = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError("Component shard must be a local filename")
        shard = load_file(str(directory / name))
        if set(shard) & set(state):
            raise ValueError("Duplicate tensor in component shards")
        state.update(shard)
    return state


def load_flux3_components(path, dtype=torch.bfloat16, vae_dtype=torch.float32):
    """Load concrete classes with complete key/shape/finite checks, offline.

    Native BF16 packages load in memory through the public DiT conversion; no
    imaginary transformer/ folder, Hub resolution or credentials are required.
    Converted diffusers folders are also accepted for reproducible tiny tests.
    """
    root = Path(path)
    native = (root / "model.safetensors").is_file()
    package = json.loads((root / "config.json").read_text()) if native else None
    if native:
        config = transformer_config_from_package(package)
        manifest = json.loads((root / "manifest.json").read_text())
        actual = sha256_file(root / "model.safetensors")
        if manifest["sha256"]["model.safetensors"] != actual:
            raise ValueError("Native FLUX.3 transformer SHA256 mismatch")
        state = convert_flux3_transformer_checkpoint_to_diffusers(load_file(str(root / "model.safetensors")))
    else:
        config = Flux3Transformer2DModel.load_config(root / "transformer", local_files_only=True)
        state = _component_state(root / "transformer", "diffusion_pytorch_model")
        actual = hashlib.sha256(json.dumps(dict(config), sort_keys=True).encode())
        for file in sorted((root / "transformer").glob("*.safetensors")):
            actual.update(sha256_file(file).encode())
        actual = actual.hexdigest()
    with init_empty_weights():
        transformer = Flux3Transformer2DModel.from_config(config)
    if transformer.config.flow_map:
        raise ValueError("FLUX.3 base cannot load distilled flow-map weights")
    transformer = _assign(transformer, state, dtype)
    del state
    text_dir = root / "text_encoder"
    # Concrete Transformers loader handles published 4.x -> 5.x key renames
    # and tied embeddings. Loading diagnostics must be empty.
    text, info = Qwen3VLForConditionalGeneration.from_pretrained(
        text_dir, dtype=dtype, local_files_only=True, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete Qwen3-VL checkpoint: {info}")
    with init_empty_weights():
        if native:
            vae = AutoencoderKLFlux2()
        else:
            vae = AutoencoderKLFlux2.from_config(AutoencoderKLFlux2.load_config(root / "vae", local_files_only=True))
    state = convert_native_vae(load_file(str(root / "ae.safetensors"))) if native else _component_state(root / "vae", "diffusion_pytorch_model")
    vae = _assign(vae, state, vae_dtype)
    identity = hashlib.sha256(actual.encode())
    for file in sorted(text_dir.glob("*.safetensors")):
        identity.update(sha256_file(file).encode())
    for file in ([root / "ae.safetensors"] if native else sorted((root / "vae").glob("*.safetensors"))):
        identity.update(sha256_file(file).encode())
    return transformer, text, vae, identity.hexdigest(), package
