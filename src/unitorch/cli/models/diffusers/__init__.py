# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.cli import hf_endpoint_url

__hf_hub_stable_v1_5_dict__ = lambda name: {
    "unet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/unet/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/unet/diffusion_pytorch_model.bin"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/pytorch_model.bin"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_v1_5_safetensors_dict__ = lambda name: {
    "unet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/unet/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/unet/diffusion_pytorch_model.safetensors"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_v2_dict__ = __hf_hub_stable_v1_5_dict__
__hf_hub_stable_v2_safetensors_dict__ = __hf_hub_stable_v1_5_safetensors_dict__
__hf_hub_stable_v2_1_dict__ = __hf_hub_stable_v1_5_dict__
__hf_hub_stable_v2_1_safetensors_dict__ = __hf_hub_stable_v1_5_safetensors_dict__

__hf_hub_stable_xl_dict__ = lambda name: {
    "unet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/unet/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/unet/diffusion_pytorch_model.bin"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/pytorch_model.bin"
        ),
    },
    "text2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_2/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder_2/pytorch_model.bin"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_xl_safetensors_dict__ = lambda name: {
    "unet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/unet/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/model.fp16.safetensors"
        ),
    },
    "text2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_2/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder_2/model.fp16.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_3_safetensors_dict__ = lambda name: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/transformer/diffusion_pytorch_model.safetensors"
        ),
    },
    "text": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/model.safetensors"
        ),
    },
    "text2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_2/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/vocab.json"),
        "merge": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/merges.txt"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder_2/model.safetensors"
        ),
    },
    "text3": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_3/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_3/spiece.model"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder_3/model-{str(i).rjust(5, '0')}-of-00002.safetensors"
            )
            for i in range(1, 3)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_flux_safetensors_dict__ = lambda name, n1=3, n2=2: {
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
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/model.safetensors"
        ),
    },
    "text2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_2/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/spiece.model"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder_2/model-{str(i).rjust(5, '0')}-of-{str(n2).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n2 + 1)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_flux_ctrl_safetensors_dict__ = lambda name, n1=3, n2=4: {
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
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/text_encoder/model.safetensors"
        ),
    },
    "text2": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/text_encoder_2/config.json"),
        "vocab": hf_endpoint_url(f"/{name}/resolve/main/tokenizer_2/spiece.model"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/text_encoder_2/model-{str(i).rjust(5, '0')}-of-{str(n2).rjust(5, '0')}.safetensors"
            )
            for i in range(1, n2 + 1)
        ],
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

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

__hf_flux_dev_redux_safetensors_dict__ = lambda x: {
    "image": {
        "config": hf_endpoint_url(f"/{x}/resolve/main/image_encoder/config.json"),
        "vision_config": hf_endpoint_url(
            f"/{x}/resolve/main/feature_extractor/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(f"/{x}/resolve/main/image_encoder/model.safetensors"),
    },
    "redux_image": {
        "config": hf_endpoint_url(f"/{x}/resolve/main/image_embedder/config.json"),
        "weight": hf_endpoint_url(
            f"/{x}/resolve/main/image_embedder/diffusion_pytorch_model.safetensors"
        ),
    },
}

pretrained_stable_infos = {
    "stable-v1.5": {
        **__hf_hub_stable_v1_5_dict__("botp/stable-diffusion-v1-5"),
        **__hf_hub_vae_dict("botp/stable-diffusion-v1-5"),
    },
    "stable-v1.5-realistic-v5.1-no-vae": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ),
        **__hf_hub_vae_safetensors_dict__("SG161222/Realistic_Vision_V5.1_noVAE"),
    },
    "stable-v1.5-realistic-v5.1": {
        **__hf_hub_stable_v1_5_dict__("stablediffusionapi/realistic-vision-v51"),
        **__hf_hub_vae_dict("stablediffusionapi/realistic-vision-v51"),
    },
    "stable-v1.5-film": {
        **__hf_hub_stable_v1_5_safetensors_dict__("Yntec/Film"),
        **__hf_hub_vae_safetensors_dict__("Yntec/Film"),
    },
    "stable-v1.5-cyber-realistic": {
        **__hf_hub_stable_v1_5_safetensors_dict__("Yntec/CyberRealistic"),
        **__hf_hub_vae_safetensors_dict__("Yntec/CyberRealistic"),
    },
    "stable-v1.5-majicmix-realistic-v6": {
        **__hf_hub_stable_v1_5_dict__("digiplay/majicMIX_realistic_v6"),
        **__hf_hub_vae_dict("digiplay/majicMIX_realistic_v6"),
    },
    "stable-v1.5-nitrosocke-ghibli": {
        **__hf_hub_stable_v1_5_dict__("nitrosocke/Ghibli-Diffusion"),
        **__hf_hub_vae_dict("nitrosocke/Ghibli-Diffusion"),
    },
    "stable-v1.5-inpainting": {
        **__hf_hub_stable_v1_5_dict__("botp/stable-diffusion-v1-5-inpainting"),
        **__hf_hub_vae_dict("botp/stable-diffusion-v1-5-inpainting"),
    },
    "stable-v1.5-realistic-v5.1-inpainting": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "Uminosachi/realisticVisionV51_v51VAE-inpainting"
        ),
        **__hf_hub_vae_safetensors_dict__(
            "Uminosachi/realisticVisionV51_v51VAE-inpainting"
        ),
    },
    "stable-v1.5-dreamshaper-8-inpainting": {
        **__hf_hub_stable_v1_5_safetensors_dict__("Lykon/dreamshaper-8-inpainting"),
        **__hf_hub_vae_safetensors_dict__("Lykon/dreamshaper-8-inpainting"),
    },
    "stable-v1.5-anything-4.0-inpainting": {
        **__hf_hub_stable_v1_5_safetensors_dict__("Sanster/anything-4.0-inpainting"),
        **__hf_hub_vae_safetensors_dict__("Sanster/anything-4.0-inpainting"),
    },
    "stable-v1.5-x4-upscaler": {
        **__hf_hub_stable_v1_5_dict__("stabilityai/stable-diffusion-x4-upscaler"),
        **__hf_hub_vae_safetensors_dict__("stabilityai/stable-diffusion-x4-upscaler"),
    },
    "stable-v2": {
        **__hf_hub_stable_v2_dict__("stabilityai/stable-diffusion-2"),
        **__hf_hub_vae_dict("stabilityai/stable-diffusion-2"),
    },
    "stable-v2.1": {
        **__hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-1"),
        **__hf_hub_vae_dict("stabilityai/stable-diffusion-2-1"),
    },
    "stable-v2.1-inpainting": {
        **__hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-inpainting"),
        **__hf_hub_vae_dict("stabilityai/stable-diffusion-2-inpainting"),
    },
    "stable-xl-base": {
        **__hf_hub_stable_xl_safetensors_dict__(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        **__hf_hub_vae_safetensors_dict__("stabilityai/stable-diffusion-xl-base-1.0"),
    },
    "stable-xl-base-vae-fp16": {
        **__hf_hub_stable_xl_safetensors_dict__(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        **__hf_hub_vae_safetensors_dict__("madebyollin/sdxl-vae-fp16-fix"),
    },
    "stable-xl-turbo": {
        **__hf_hub_stable_xl_safetensors_dict__("stabilityai/sdxl-turbo"),
        **__hf_hub_vae_safetensors_dict__("stabilityai/sdxl-turbo"),
    },
    "stable-xl-realism-engine-v30": {
        **__hf_hub_stable_xl_safetensors_dict__("misri/realismEngineSDXL_v30VAE"),
        **__hf_hub_vae_safetensors_dict__("misri/realismEngineSDXL_v30VAE"),
    },
    "stable-xl-base-inpainting-0.1": {
        **__hf_hub_stable_xl_safetensors_dict__(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        ),
        **__hf_hub_vae_safetensors_fp16_dict__(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        ),
    },
    "stable-xl-opendalle-v1.1": {
        **__hf_hub_stable_xl_safetensors_dict__("dataautogpt3/OpenDalleV1.1"),
        **__hf_hub_vae_safetensors_dict__("dataautogpt3/OpenDalleV1.1"),
    },
    "stable-xl-realvis-v4.0": {
        **__hf_hub_stable_xl_safetensors_dict__("SG161222/RealVisXL_V4.0"),
        **__hf_hub_vae_safetensors_dict__("SG161222/RealVisXL_V4.0"),
    },
    "stable-xl-realvis-v4.0-inpainting": {
        **__hf_hub_stable_xl_safetensors_dict__("OzzyGT/RealVisXL_V4.0_inpainting"),
        **__hf_hub_vae_safetensors_fp16_dict__("OzzyGT/RealVisXL_V4.0_inpainting"),
    },
    "stable-xl-juggernaut-v8": {
        **__hf_hub_stable_xl_dict__("RunDiffusion/Juggernaut-XL-v8"),
        **__hf_hub_vae_dict("RunDiffusion/Juggernaut-XL-v8"),
    },
    "stable-xl-playground-v2-aesthetic": {
        **__hf_hub_stable_xl_safetensors_dict__(
            "playgroundai/playground-v2-1024px-aesthetic"
        ),
        **__hf_hub_vae_safetensors_dict__(
            "playgroundai/playground-v2-1024px-aesthetic"
        ),
    },
    "stable-v3-medium": {
        **__hf_hub_stable_3_safetensors_dict__(
            "ckpt/stable-diffusion-3-medium-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("ckpt/stable-diffusion-3-medium-diffusers"),
    },
    "stable-v3.5-medium": {
        **__hf_hub_stable_3_safetensors_dict__("ckpt/stable-diffusion-3.5-medium"),
        **__hf_hub_vae_safetensors_dict__("ckpt/stable-diffusion-3.5-medium"),
    },
    "stable-v3.5-large": {
        **__hf_hub_stable_3_safetensors_dict__("yuvraj108c/stable-diffusion-3.5-large"),
        **__hf_hub_vae_safetensors_dict__("yuvraj108c/stable-diffusion-3.5-large"),
    },
    "stable-flux-schnell": {
        **__hf_hub_stable_flux_safetensors_dict__("lzyvegetable/FLUX.1-schnell"),
        **__hf_hub_vae_safetensors_dict__("lzyvegetable/FLUX.1-schnell"),
    },
    "stable-flux-dev": {
        **__hf_hub_stable_flux_safetensors_dict__("camenduru/FLUX.1-dev-diffusers"),
        **__hf_hub_vae_safetensors_dict__("camenduru/FLUX.1-dev-diffusers"),
    },
    "stable-flux-dev-krea": {
        **__hf_hub_stable_flux_safetensors_dict__("NikolaSigmoid/FLUX.1-Krea-dev"),
        **__hf_hub_vae_safetensors_dict__("NikolaSigmoid/FLUX.1-Krea-dev"),
    },
    "stable-flux-dev-fill": {
        **__hf_hub_stable_flux_safetensors_dict__(
            "fuliucansheng/FLUX.1-Fill-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Fill-dev-diffusers"),
    },
    "stable-flux-dev-canny": {
        **__hf_hub_stable_flux_ctrl_safetensors_dict__(
            "fuliucansheng/FLUX.1-Canny-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Canny-dev-diffusers"),
    },
    "stable-flux-dev-depth": {
        **__hf_hub_stable_flux_ctrl_safetensors_dict__(
            "fuliucansheng/FLUX.1-Depth-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Depth-dev-diffusers"),
    },
    "stable-flux-dev-kontext": {
        **__hf_hub_stable_flux_safetensors_dict__(
            "fuliucansheng/FLUX.1-Kontext-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Kontext-dev-diffusers"),
    },
    "stable-flux-dev-redux": {
        **__hf_hub_stable_flux_safetensors_dict__("camenduru/FLUX.1-dev-diffusers"),
        **__hf_hub_vae_safetensors_dict__("camenduru/FLUX.1-dev-diffusers"),
        **__hf_flux_dev_redux_safetensors_dict__("tentpole/flux1-dev-redux"),
    },
    "stable-flux-dev-redux-fill": {
        **__hf_hub_stable_flux_safetensors_dict__(
            "fuliucansheng/FLUX.1-Fill-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Fill-dev-diffusers"),
        **__hf_flux_dev_redux_safetensors_dict__("tentpole/flux1-dev-redux"),
    },
    "stable-flux-dev-redux-canny": {
        **__hf_hub_stable_flux_ctrl_safetensors_dict__(
            "fuliucansheng/FLUX.1-Canny-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Canny-dev-diffusers"),
        **__hf_flux_dev_redux_safetensors_dict__("tentpole/flux1-dev-redux"),
    },
    "stable-flux-dev-redux-depth": {
        **__hf_hub_stable_flux_ctrl_safetensors_dict__(
            "fuliucansheng/FLUX.1-Depth-dev-diffusers"
        ),
        **__hf_hub_vae_safetensors_dict__("fuliucansheng/FLUX.1-Depth-dev-diffusers"),
        **__hf_flux_dev_redux_safetensors_dict__("tentpole/flux1-dev-redux"),
    },
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
    },
    "wan-v2.2-i2v-14b": {
        **__hf_hub_wan_v2_2_safetensors_dict__(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", n1=12, n2=12, n3=3
        ),
        **__hf_hub_vae_safetensors_dict__("Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
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
}

pretrained_stable_extensions_infos = {
    # sdxl refiner
    "stable-xl-refiner-1.0": {
        "refiner": {
            "unet": {
                "config": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/config.json"
                ),
                "weight": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
                ),
            },
            "text2": {
                "config": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/config.json"
                ),
                "vocab": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/vocab.json"
                ),
                "merge": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/merges.txt"
                ),
                "weight": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/model.fp16.safetensors"
                ),
            },
            "vae": {
                "config": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/config.json"
                ),
                "weight": hf_endpoint_url(
                    f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors"
                ),
            },
            "scheduler": hf_endpoint_url(
                f"/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/scheduler/scheduler_config.json"
            ),
        },
    },
    # sdxl vae
    "stable-xl-vae-fp16": {
        "vae": {
            "config": hf_endpoint_url(
                "/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json"
            ),
            "weight": hf_endpoint_url(
                "/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors"
            ),
        },
    },
}

from unitorch.cli.models.diffusion_utils import load_weight

import unitorch.cli.models.diffusers.modeling_stable
import unitorch.cli.models.diffusers.modeling_stable_xl
import unitorch.cli.models.diffusers.modeling_stable_3
import unitorch.cli.models.diffusers.modeling_stable_flux
import unitorch.cli.models.diffusers.modeling_qwen_image
import unitorch.cli.models.diffusers.modeling_wan
import unitorch.cli.models.diffusers.modeling_vae
import unitorch.cli.models.diffusers.processing_stable
import unitorch.cli.models.diffusers.processing_stable_xl
import unitorch.cli.models.diffusers.processing_stable_3
import unitorch.cli.models.diffusers.processing_stable_flux
import unitorch.cli.models.diffusers.processing_qwen_image

if is_opencv_available():
    import unitorch.cli.models.diffusers.processing_wan
