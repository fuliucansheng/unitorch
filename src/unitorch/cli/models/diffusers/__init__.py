# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

__hf_hub_stable_v1_5_dict__ = lambda name: {
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

__hf_hub_stable_v1_5_safetensors_dict__ = lambda name: {
    "unet": {
        "config": f"https://huggingface.co/{name}/resolve/main/unet/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/unet/diffusion_pytorch_model.safetensors",
    },
    "text": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder/model.safetensors",
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

__hf_hub_stable_v2_dict__ = __hf_hub_stable_v1_5_dict__
__hf_hub_stable_v2_safetensors_dict__ = __hf_hub_stable_v1_5_safetensors_dict__
__hf_hub_stable_v2_1_dict__ = __hf_hub_stable_v1_5_dict__
__hf_hub_stable_v2_1_safetensors_dict__ = __hf_hub_stable_v1_5_safetensors_dict__

__hf_hub_stable_xl_dict__ = lambda name: {
    "unet": {
        "config": f"https://huggingface.co/{name}/resolve/main/unet/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
    },
    "text": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder/model.fp16.safetensors",
    },
    "text2": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/model.fp16.safetensors",
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

__hf_hub_stable_v1_5_controlnet_dict__ = lambda name: {
    **__hf_hub_stable_v1_5_dict__("runwayml/stable-diffusion-v1-5"),
    **{
        "controlnet": {
            "config": f"https://huggingface.co/{name}/resolve/main/config.json",
            "weight": f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.bin",
        }
    },
}

__hf_hub_stable_xl_controlnet_dict__ = lambda name: {
    **__hf_hub_stable_xl_dict__("stabilityai/stable-diffusion-xl-base-1.0"),
    **{
        "controlnet": {
            "config": f"https://huggingface.co/{name}/resolve/main/config.json",
            "weight": f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.bin",
        }
    },
}

pretrained_diffusers_infos = {
    "stable-v1.5": __hf_hub_stable_v1_5_dict__("runwayml/stable-diffusion-v1-5"),
    "stable-v1.5-realistic-v5.1-no-vae": __hf_hub_stable_v1_5_safetensors_dict__(
        "SG161222/Realistic_Vision_V5.1_noVAE"
    ),
    "stable-v1.5-realistic-v5.1": __hf_hub_stable_v1_5_dict__(
        "stablediffusionapi/realistic-vision-v51"
    ),
    "stable-v1.5-cyber-realistic": __hf_hub_stable_v1_5_safetensors_dict__(
        "Yntec/CyberRealistic"
    ),
    "stable-v1.5-majicmix-realistic-v6": __hf_hub_stable_v1_5_dict__(
        "digiplay/majicMIX_realistic_v6"
    ),
    "stable-v1.5-chilloutmix": __hf_hub_stable_v1_5_dict__(
        "emilianJR/chilloutmix_NiPrunedFp32Fix"
    ),
    "stable-v1.5-dreamshaper-8": __hf_hub_stable_v1_5_safetensors_dict__(
        "Lykon/dreamshaper-8"
    ),
    "stable-v1.5-nitrosocke-ghibli": __hf_hub_stable_v1_5_dict__(
        "nitrosocke/Ghibli-Diffusion"
    ),
    "stable-v1.5-inpainting": __hf_hub_stable_v1_5_dict__(
        "runwayml/stable-diffusion-inpainting"
    ),
    "stable-v1.5-dreamshaper-8-inpainting": __hf_hub_stable_v1_5_safetensors_dict__(
        "Lykon/dreamshaper-8-inpainting"
    ),
    "stable-v1.5-deliberate-v2-inpainting": __hf_hub_stable_v1_5_dict__(
        "5w4n/deliberate-v2-inpainting"
    ),
    "stable-v1.5-x4-upscaler": __hf_hub_stable_v1_5_dict__(
        "stabilityai/stable-diffusion-x4-upscaler"
    ),
    "stable-v1.5-dreambooth-cat-toy": __hf_hub_stable_v1_5_dict__(
        "sd-dreambooth-library/cat-toy"
    ),
    "stable-v1.5-controlnet-canny": __hf_hub_stable_v1_5_controlnet_dict__(
        "lllyasviel/sd-controlnet-canny"
    ),
    "stable-v1.5-controlnet-openpose": __hf_hub_stable_v1_5_controlnet_dict__(
        "lllyasviel/sd-controlnet-openpose"
    ),
    "stable-v1.5-controlnet-inpainting": __hf_hub_stable_v1_5_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_inpaint"
    ),
    "stable-v1.5-blipdiffusion": {
        **__hf_hub_stable_v1_5_dict__("salesforce/blipdiffusion"),
        **{
            "qformer": {
                "config": f"https://huggingface.co/salesforce/blipdiffusion/resolve/main/qformer/config.json",
                "weight": f"https://huggingface.co/salesforce/blipdiffusion/resolve/main/qformer/pytorch_model.bin",
            },
            "vision_encoder": {
                "config": f"https://huggingface.co/salesforce/blipdiffusion/resolve/main/vision_encoder/config.json",
                "weight": f"https://huggingface.co/salesforce/blipdiffusion/resolve/main/vision_encoder/pytorch_model.bin",
            },
            "vision_config": "https://huggingface.co/salesforce/blipdiffusion/resolve/main/image_processor/preprocessor_config.json",
        },
    },
    "stable-v1.5-blipdiffusion-controlnet": {
        **__hf_hub_stable_v1_5_dict__("salesforce/blipdiffusion-controlnet"),
        **{
            "qformer": {
                "config": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/qformer/config.json",
                "weight": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/qformer/pytorch_model.bin",
            },
            "vision_encoder": {
                "config": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/vision_encoder/config.json",
                "weight": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/vision_encoder/pytorch_model.bin",
            },
            "vision_config": "https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/image_processor/preprocessor_config.json",
            "controlnet": {
                "config": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/controlnet/config.json",
                "weight": f"https://huggingface.co/salesforce/blipdiffusion-controlnet/resolve/main/controlnet/diffusion_pytorch_model.bin",
            },
        },
    },
    "stable-v1.5-realistic-animate-v1.5": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ),
        **{
            "motion": {
                "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5/resolve/main/config.json",
                "weight": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5/resolve/main/diffusion_pytorch_model.safetensors",
            },
        },
    },
    "stable-v1.5-realistic-animate-v1.5.2": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ),
        **{
            "motion": {
                "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
                "weight": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/diffusion_pytorch_model.safetensors",
            },
        },
    },
    "stable-v1.5-realistic-animate-v1.5.2-zoom-in": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ),
        **{
            "motion": {
                "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
                "weight": f"https://huggingface.co/fuliucansheng/diffusers/resolve/main/animatediff/diffusion_pytorch_model.zoom_in.safetensors",
            },
        },
    },
    "stable-v1.5-realistic-animate-v1.5.2-zoom-out": {
        **__hf_hub_stable_v1_5_safetensors_dict__(
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ),
        **{
            "motion": {
                "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
                "weight": f"https://huggingface.co/fuliucansheng/diffusers/resolve/main/animatediff/diffusion_pytorch_model.zoom_out.safetensors",
            },
        },
    },
    "stable-v2": __hf_hub_stable_v2_dict__("stabilityai/stable-diffusion-2"),
    "stable-v2.1": __hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-1"),
    "stable-xl-base-1.0": __hf_hub_stable_xl_dict__(
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),
    "stable-xl-dreamshaper-xl-1.0": __hf_hub_stable_xl_dict__(
        "Lykon/dreamshaper-xl-1-0"
    ),
    "stable-xl-base-refiner-1.0": {
        **__hf_hub_stable_xl_dict__("stabilityai/stable-diffusion-xl-base-1.0"),
        **{
            "refiner_unet": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
            },
            "refiner_text2": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/config.json",
                "vocab": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/vocab.json",
                "merge": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/merges.txt",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/model.fp16.safetensors",
            },
            "refiner_vae": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
            },
            "refiner_scheduler": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/scheduler/scheduler_config.json",
        },
    },
    "stable-xl-dreamshaper-xl-1.0-refiner": {
        **__hf_hub_stable_xl_dict__("Lykon/dreamshaper-xl-1-0"),
        **{
            "refiner_unet": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
            },
            "refiner_text2": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/config.json",
                "vocab": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/vocab.json",
                "merge": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/merges.txt",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/model.fp16.safetensors",
            },
            "refiner_vae": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
            },
            "refiner_scheduler": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/scheduler/scheduler_config.json",
        },
    },
    "stable-xl-base-1.0-controlnet-canny": __hf_hub_stable_xl_controlnet_dict__(
        "diffusers/controlnet-canny-sdxl-1.0"
    ),
    "stable-xl-base-1.0-controlnet-depth-small": __hf_hub_stable_xl_controlnet_dict__(
        "diffusers/controlnet-depth-sdxl-1.0-small"
    ),
}

from unitorch.cli.models.diffusion_utils import load_weight

from unitorch.cli.models.diffusers.modeling_stable import (
    StableForText2ImageGeneration,
    StableForImage2ImageGeneration,
    StableForImageInpainting,
    StableForImageResolution,
)
from unitorch.cli.models.diffusers.modeling_stable_xl import (
    StableXLForText2ImageGeneration,
    StableXLForImage2ImageGeneration,
    StableXLForImageInpainting,
)
from unitorch.cli.models.diffusers.modeling_stable_xl_refiner import (
    StableXLRefinerForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_dreambooth import (
    DreamboothForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_dreambooth_xl import (
    DreamboothXLForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_controlnet import (
    ControlNetForText2ImageGeneration,
    ControlNetForImage2ImageGeneration,
    ControlNetForImageInpainting,
)
from unitorch.cli.models.diffusers.modeling_controlnet_xl import (
    ControlNetXLForText2ImageGeneration,
    ControlNetXLForImage2ImageGeneration,
    ControlNetXLForImageInpainting,
)
from unitorch.cli.models.diffusers.modeling_blip2 import (
    Blip2ForText2ImageGeneration,
)

from unitorch.cli.models.diffusers.processing_stable import StableProcessor
from unitorch.cli.models.diffusers.processing_stable_xl import StableXLProcessor
from unitorch.cli.models.diffusers.processing_stable_xl_refiner import (
    StableXLRefinerProcessor,
)
from unitorch.cli.models.diffusers.processing_dreambooth import DreamboothProcessor
from unitorch.cli.models.diffusers.processing_dreambooth_xl import DreamboothXLProcessor
from unitorch.cli.models.diffusers.processing_controlnet import ControlNetProcessor
from unitorch.cli.models.diffusers.processing_controlnet_xl import ControlNetXLProcessor
from unitorch.cli.models.diffusers.processing_blip2 import Blip2Processor

import unitorch.cli.models.diffusers.modeling_animate
