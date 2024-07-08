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
        "weight": f"https://huggingface.co/{name}/resolve/main/unet/diffusion_pytorch_model.bin",
    },
    "text": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder/pytorch_model.bin",
    },
    "text2": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/pytorch_model.bin",
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.bin",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

__hf_hub_stable_xl_safetensors_dict__ = lambda name: {
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

__hf_hub_stable_3_safetensors_dict__ = lambda name: {
    "transformer": {
        "config": f"https://huggingface.co/{name}/resolve/main/transformer/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/transformer/diffusion_pytorch_model.safetensors",
    },
    "text": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder/model.safetensors",
    },
    "text2": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/vocab.json",
        "merge": f"https://huggingface.co/{name}/resolve/main/tokenizer_2/merges.txt",
        "weight": f"https://huggingface.co/{name}/resolve/main/text_encoder_2/model.safetensors",
    },
    "text3": {
        "config": f"https://huggingface.co/{name}/resolve/main/text_encoder_3/config.json",
        "vocab": f"https://huggingface.co/{name}/resolve/main/tokenizer_3/spiece.model",
        "weight": [
            f"https://huggingface.co/{name}/resolve/main/text_encoder_3/model-{str(i).rjust(5, '0')}-of-00002.safetensors"
            for i in range(1, 3)
        ],
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

__hf_hub_controlnet_dict__ = lambda name: {
    "controlnet": {
        "config": f"https://huggingface.co/{name}/resolve/main/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.bin",
    }
}

__hf_hub_controlnet_safetensors_dict__ = lambda name: {
    "controlnet": {
        "config": f"https://huggingface.co/{name}/resolve/main/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.safetensors",
    }
}

__hf_hub_stable_video_safetensors_dict__ = lambda name: {
    "unet": {
        "config": f"https://huggingface.co/{name}/resolve/main/unet/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/unet/diffusion_pytorch_model.safetensors",
    },
    "image": {
        "config": f"https://huggingface.co/{name}/resolve/main/image_encoder/config.json",
        "vision_config": f"https://huggingface.co/{name}/resolve/main/feature_extractor/preprocessor_config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/image_encoder/model.safetensors",
    },
    "vae": {
        "config": f"https://huggingface.co/{name}/resolve/main/vae/config.json",
        "weight": f"https://huggingface.co/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors",
    },
    "scheduler": f"https://huggingface.co/{name}/resolve/main/scheduler/scheduler_config.json",
}

pretrained_stable_infos = {
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
    "stable-v1.5-nitrosocke-ghibli": __hf_hub_stable_v1_5_dict__(
        "nitrosocke/Ghibli-Diffusion"
    ),
    "stable-v1.5-inpainting": __hf_hub_stable_v1_5_dict__(
        "runwayml/stable-diffusion-inpainting"
    ),
    "stable-v1.5-x4-upscaler": __hf_hub_stable_v1_5_dict__(
        "stabilityai/stable-diffusion-x4-upscaler"
    ),
    "stable-v2": __hf_hub_stable_v2_dict__("stabilityai/stable-diffusion-2"),
    "stable-v2.1": __hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-1"),
    "stable-xl-base": __hf_hub_stable_xl_safetensors_dict__(
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),
    "stable-xl-turbo": __hf_hub_stable_xl_safetensors_dict__("stabilityai/sdxl-turbo"),
    "stable-xl-realism-v30": __hf_hub_stable_xl_dict__(
        "stablediffusionapi/realism-engine-sdxl-v30"
    ),
    "stable-xl-opendalle-v1.1": __hf_hub_stable_xl_safetensors_dict__(
        "dataautogpt3/OpenDalleV1.1"
    ),
    "stable-xl-realvis-v3.0": __hf_hub_stable_xl_safetensors_dict__(
        "SG161222/RealVisXL_V3.0"
    ),
    "stable-xl-juggernaut-v8": __hf_hub_stable_xl_dict__(
        "stablediffusionapi/juggernaut-xl-v8"
    ),
    "stable-xl-playground-v2-aesthetic": __hf_hub_stable_xl_safetensors_dict__(
        "playgroundai/playground-v2-1024px-aesthetic"
    ),
    "stable-3-medium": __hf_hub_stable_3_safetensors_dict__(
        "ckpt/stable-diffusion-3-medium-diffusers"
    ),
    "stable-video-diffusion-img2vid-xt": __hf_hub_stable_video_safetensors_dict__(
        "stabilityai/stable-video-diffusion-img2vid-xt"
    ),
    "stable-video-diffusion-img2vid-xt-1-1": __hf_hub_stable_video_safetensors_dict__(
        "vdo/stable-video-diffusion-img2vid-xt-1-1"
    ),
}

pretrained_stable_extensions_infos = {
    "stable-v1.5-controlnet-canny": __hf_hub_controlnet_dict__(
        "lllyasviel/sd-controlnet-canny"
    ),
    "stable-v1.5-controlnet-openpose": __hf_hub_controlnet_dict__(
        "lllyasviel/sd-controlnet-openpose"
    ),
    "stable-v1.5-controlnet-softedge": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_softedge",
    ),
    "stable-v1.5-controlnet-inpainting": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_inpaint"
    ),
    "stable-v1.5-controlnet-depth": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11f1p_sd15_depth"
    ),
    "stable-xl-controlnet-canny": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-canny-sdxl-1.0"
    ),
    "stable-xl-controlnet-softedge-dexined": __hf_hub_controlnet_dict__(
        "SargeZT/controlnet-sd-xl-1.0-softedge-dexined"
    ),
    "stable-xl-controlnet-depth": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-depth-sdxl-1.0"
    ),
    "stable-xl-controlnet-softedge": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-softedge-sdxl-1.0"
    ),
    "stable-xl-controlnet-depth-small": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-depth-sdxl-1.0-small"
    ),
    "stable-xl-adapter-t2i-canny": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-canny-sdxl-1.0"
    ),
    "stable-xl-adapter-t2i-sketch": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0"
    ),
    "stable-xl-adapter-t2i-openpose": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0"
    ),
    "stable-xl-refiner-1.0": {
        "refiner": {
            "unet": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
            },
            "text2": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/config.json",
                "vocab": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/vocab.json",
                "merge": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/tokenizer_2/merges.txt",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/model.fp16.safetensors",
            },
            "vae": {
                "config": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/config.json",
                "weight": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
            },
            "scheduler": f"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/scheduler/scheduler_config.json",
        },
    },
    "stable-xl-vae-fp16": {
        "vae": {
            "config": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json",
            "weight": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors",
        },
    },
    "stable-3-controlnet-canny": __hf_hub_controlnet_safetensors_dict__(
        "InstantX/SD3-Controlnet-Canny"
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
from unitorch.cli.models.diffusers.modeling_stable_3 import (
    Stable3ForText2ImageGeneration,
    Stable3ForImage2ImageGeneration,
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
from unitorch.cli.models.diffusers.modeling_controlnet_3 import (
    ControlNet3ForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.processing_stable import StableProcessor
from unitorch.cli.models.diffusers.processing_stable_xl import StableXLProcessor
from unitorch.cli.models.diffusers.processing_controlnet import ControlNetProcessor
from unitorch.cli.models.diffusers.processing_controlnet_xl import ControlNetXLProcessor
from unitorch.cli.models.diffusers.processing_controlnet_3 import ControlNet3Processor
