# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.cli import hf_endpoint_url

__hf_hub_wan_v2_2_safetensors_dict__ = lambda name, n1=12, n2=12, n3=3: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(n1).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n1 + 1)
        ],
    },
    "transformer2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer_2/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(n2).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n2 + 1)
        ],
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/spiece.model"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder/model-{str(i).rjust(5, '0')}-of-{str(n3).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n3 + 1)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

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

__hf_hub_flux2_safetensors_dict__ = lambda name, n1=7, n2=10: {
    "repo_id": name,
    "processor": {
        "name": name,
        "subfolder": "tokenizer",
    },
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
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", n1=12, n2=12, n3=3
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        "boundary_ratio": 0.875,
    },
    "wan-v2.2-i2v-14b": {
        **__hf_hub_wan_v2_2_safetensors_dict__(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", n1=12, n2=12, n3=3
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        "boundary_ratio": 0.9,
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
        **__hf_hub_flux2_safetensors_dict__("black-forest-labs/FLUX.2-dev"),
        **__hf_hub_vae_safetensors_dict__("black-forest-labs/FLUX.2-dev"),
    },
    "lucy-edit-v1.1-dev": {
        **__hf_hub_lucy_edit_safetensors_dict__("decart-ai/Lucy-Edit-1.1-Dev"),
        **__hf_hub_vae_safetensors_dict__("decart-ai/Lucy-Edit-1.1-Dev"),
    },
}

pretrained_stable_extensions_infos = {}

from unitorch.cli.models.diffusion_utils import load_weight

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
