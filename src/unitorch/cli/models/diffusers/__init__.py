# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.cli import hf_endpoint_url


def __hf_hub_wan_v2_2_safetensors_dict__(
    name: str,
    transformer_shards: int = 12,
    text_shards: int = 3,
    transformer2_shards: int = 12,
):
    info = {
        "transformer": {
            "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
            "weight": [
                hf_endpoint_url(
                    f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(transformer_shards).rjust(5, '0')}.safetensors"
                )
                for i in range(1, transformer_shards + 1)
            ],
        },
        "text": {
            "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
            "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/spiece.model"),
            "weight": [
                hf_endpoint_url(
                    f"/{name}/resolve/main/text_encoder/model-{str(i).rjust(5, '0')}-of-{str(text_shards).rjust(5, '0')}.safetensors"
                )
                for i in range(1, text_shards + 1)
            ],
        },
        "scheduler": hf_endpoint_url(
            f"/{name}/resolve/main/scheduler/scheduler_config.json"
        ),
    }
    if transformer2_shards is not None:
        info["transformer2"] = {
            "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
            "weight": [
                hf_endpoint_url(
                    f"/{name}/resolve/main/transformer_2/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(transformer2_shards).rjust(5, '0')}.safetensors"
                )
                for i in range(1, transformer2_shards + 1)
            ],
        }
    return info

__hf_hub_lucy_edit_safetensors_dict__ = lambda name, n1=3: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/transformer/diffusion_pytorch_model.safetensors"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/spiece.model"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder/model-{str(i).rjust(5, '0')}-of-{str(n1).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n1 + 1)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
    "expand_timesteps": True,
}

__hf_hub_qwen_image_safetensors_dict__ = lambda name, n1=9, n2=4: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(n1).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n1 + 1)
        ],
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "tokenizer_config": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder/model-{str(i).rjust(5, '0')}-of-{str(n2).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n2 + 1)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

def __hf_hub_flux2_safetensors_dict__(
    name,
    transformer_shards: int = 7,
    text_shards: int = 10,
    transformer_weight: str = None,
    text_weight: str = None,
    tokenizer_class: str = None,
    has_vocab: bool = False,
    has_merge: bool = False,
    has_added_tokens: bool = False,
):
    if transformer_weight is None:
        transformer_weight = [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(transformer_shards).rjust(5, '0')}.safetensors"
            )
            for i in range(1, transformer_shards + 1)
        ]
    else:
        transformer_weight = hf_endpoint_url(f"/{name}/resolve/main/{transformer_weight}")

    if text_weight is None:
        text_weight = [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder/model-{str(i).rjust(5, '0')}-of-{str(text_shards).rjust(5, '0')}.safetensors"
            )
            for i in range(1, text_shards + 1)
        ]
    else:
        text_weight = hf_endpoint_url(f"/{name}/resolve/main/{text_weight}")

    text_info = {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "tokenizer": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/special_tokens_map.json"
        ),
        "chat_template": hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/chat_template.jinja"
        ),
        "weight": text_weight,
    }
    if tokenizer_class is not None:
        text_info["tokenizer_class"] = tokenizer_class
    if has_vocab:
        text_info["vocab"] = hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/vocab.json"
        )
    if has_merge:
        text_info["merge"] = hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/merges.txt"
        )
    if has_added_tokens:
        text_info["added_tokens"] = hf_endpoint_url(
            f"/{name}/resolve/main/tokenizer/added_tokens.json"
        )

    return {
        "repo_id": name,
        "transformer": {
            "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
            "weight": transformer_weight,
        },
        "text": text_info,
        "scheduler": hf_endpoint_url(
            f"/{name}/resolve/main/scheduler/scheduler_config.json"
        ),
    }

__hf_hub_vae_dict = lambda name: {
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.bin"
        ),
    },
}

__hf_hub_vae_safetensors_dict__ = lambda name: {
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
        ),
    },
}

__hf_hub_vae_safetensors_fp16_dict__ = lambda name: {
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors"
        ),
    },
}

__hf_hub_stable_video_safetensors_dict__ = lambda name: {
    "unet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/unet/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/unet/diffusion_pytorch_model.safetensors"
        ),
    },
    "image": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/image_encoder/config.json"),
        "vision_config": hf_endpoint_url(
            f"/{name}/resolve/main/feature_extractor/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/image_encoder/model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

pretrained_stable_infos = {
    "stable-video-diffusion-img2vid-xt": {
        **__hf_hub_stable_video_safetensors_dict__(
            "stabilityai/stable-video-diffusion-img2vid-xt"
        ),
        **__hf_hub_vae_safetensors_dict__(
            "stabilityai/stable-video-diffusion-img2vid-xt"
        ),
    },
    "stable-video-diffusion-img2vid-xt-1-1": {
        **__hf_hub_stable_video_safetensors_dict__(
            "vdo/stable-video-diffusion-img2vid-xt-1-1"
        ),
        **__hf_hub_vae_safetensors_dict__("vdo/stable-video-diffusion-img2vid-xt-1-1"),
    },
    "wan-v2.2-t2v-14b": {
        **__hf_hub_wan_v2_2_safetensors_dict__(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            transformer_shards=12,
            text_shards=3,
            transformer2_shards=12,
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        "boundary_ratio": 0.875,
    },
    "wan-v2.2-i2v-14b": {
        **__hf_hub_wan_v2_2_safetensors_dict__(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            transformer_shards=12,
            text_shards=3,
            transformer2_shards=12,
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        "boundary_ratio": 0.9,
    },
    "wan-v2.2-ti2v-5b": {
        **__hf_hub_wan_v2_2_safetensors_dict__(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            transformer_shards=5,
            text_shards=3,
            transformer2_shards=None,
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        "boundary_ratio": 1.0,
        "expand_timesteps": True,
    },
    "qwen-image": {
        **__hf_hub_qwen_image_safetensors_dict__("Qwen/Qwen-Image"),
        **__hf_hub_vae_safetensors_dict__("Qwen/Qwen-Image"),
    },
    "qwen-image-editing": {
        **__hf_hub_qwen_image_safetensors_dict__("Qwen/Qwen-Image-Edit"),
        **__hf_hub_vae_safetensors_dict__("Qwen/Qwen-Image-Edit"),
        **{
            "vision_config": hf_endpoint_url(
                f"/Qwen/Qwen-Image-Edit/resolve/main/processor/preprocessor_config.json"
            ),
        },
    },
    "flux2-dev": {
        **__hf_hub_flux2_safetensors_dict__(
            "black-forest-labs/FLUX.2-dev",
            tokenizer_class="LlamaTokenizerFast",
        ),
        **__hf_hub_vae_safetensors_dict__("black-forest-labs/FLUX.2-dev"),
    },
    "flux2-klein-4b": {
        **__hf_hub_flux2_safetensors_dict__(
            "black-forest-labs/FLUX.2-klein-4B",
            transformer_weight="transformer/diffusion_pytorch_model.safetensors",
            text_shards=2,
            tokenizer_class="Qwen2Tokenizer",
            has_vocab=True,
            has_merge=True,
            has_added_tokens=True,
        ),
        **__hf_hub_vae_safetensors_dict__("black-forest-labs/FLUX.2-klein-4B"),
    },
    "lucy-edit-v1.1-dev": {
        **__hf_hub_lucy_edit_safetensors_dict__("decart-ai/Lucy-Edit-1.1-Dev"),
        **__hf_hub_vae_safetensors_dict__("decart-ai/Lucy-Edit-1.1-Dev"),
    },
}

pretrained_stable_extensions_infos = {}

from unitorch.cli.models.diffusion_utils import load_weight


def load_wan_text_weight(
    weight_path,
    replace_keys=None,
    use_auth_token=None,
):
    state_dict = load_weight(
        weight_path,
        prefix_keys={"": "text."},
        replace_keys=replace_keys,
        use_auth_token=use_auth_token,
    )
    shared_key = "text.shared.weight"
    embed_key = "text.encoder.embed_tokens.weight"
    if shared_key in state_dict and embed_key not in state_dict:
        state_dict[embed_key] = state_dict[shared_key]
    return state_dict


import unitorch.cli.models.diffusers.modeling_qwen_image
import unitorch.cli.models.diffusers.modeling_flux2
import unitorch.cli.models.diffusers.modeling_wan
import unitorch.cli.models.diffusers.modeling_lucy
import unitorch.cli.models.diffusers.modeling_vae
import unitorch.cli.models.diffusers.processing_qwen_image
import unitorch.cli.models.diffusers.processing_flux2

if is_opencv_available():
    import unitorch.cli.models.diffusers.processing_wan
    import unitorch.cli.models.diffusers.processing_lucy
