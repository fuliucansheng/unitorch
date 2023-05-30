# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

__hf_hub_stable_dict__ = lambda name: {
    "unet": {
        "config": f"https://huggingface.co/{name}/resolve/main/unet/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/unet/diffusion_pytorch_model.bin",
    },
    "text": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder/pytorch_model.bin",
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.bin",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

__hf_hub_stable_controlnet_dict__ = lambda name: {
    **__hf_hub_stable_dict__("runwayml/stable-diffusion-v1-5"),
    **{
        "controlnet": {
            "config": f"https://huggingface.co/{name}/resolve/main/config.json",
            "weight": f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.bin",
        }
    },
}

pretrained_diffusers_infos = {
    "ddpm-ema-pokemon-64": {
        "config": "https://huggingface.co/anton-l/ddpm-ema-pokemon-64/resolve/main/unet/config.json",
        "weight": "https://huggingface.co/anton-l/ddpm-ema-pokemon-64/resolve/main/unet/diffusion_pytorch_model.bin",
    },
    "stable-diffusion-v1.5": __hf_hub_stable_dict__("runwayml/stable-diffusion-v1-5"),
    "stable-diffusion-v2": __hf_hub_stable_dict__("stabilityai/stable-diffusion-2"),
    "stable-diffusion-v2.1": __hf_hub_stable_dict__("stabilityai/stable-diffusion-2-1"),
    "stable-diffusion-v2-inpainting": __hf_hub_stable_dict__(
        "stabilityai/stable-diffusion-2-inpainting"
    ),
    "stable-diffusion-x4-upscaler": __hf_hub_stable_dict__(
        "stabilityai/stable-diffusion-x4-upscaler"
    ),
    "stable-diffusion-controlnet-canny": __hf_hub_stable_controlnet_dict__(
        "lllyasviel/sd-controlnet-canny"
    ),
    "stable-diffusion-controlnet-openpose": __hf_hub_stable_controlnet_dict__(
        "lllyasviel/sd-controlnet-openpose"
    ),
    "stable-diffusion-controlnet-depth": __hf_hub_stable_controlnet_dict__(
        "lllyasviel/sd-controlnet-depth"
    ),
}
from unitorch import is_diffusers_available

if is_diffusers_available():
    import unitorch.cli.models.diffusers.modeling_controlnet
    import unitorch.cli.models.diffusers.modeling_stable
    import unitorch.cli.models.diffusers.processing_controlnet
    import unitorch.cli.models.diffusers.processing_stable
