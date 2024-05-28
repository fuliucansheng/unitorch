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

__hf_hub_multicontrolnet_dict__ = lambda *names: {
    "controlnet": {
        "config": [
            f"https://huggingface.co/{name}/resolve/main/config.json" for name in names
        ],
        "weight": [
            f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.bin"
            for name in names
        ],
    }
}

__hf_hub_multicontrolnet_safetensors_dict__ = lambda *names: {
    "controlnet": {
        "config": [
            f"https://huggingface.co/{name}/resolve/main/config.json" for name in names
        ],
        "weight": [
            f"https://huggingface.co/{name}/resolve/main/diffusion_pytorch_model.safetensors"
            for name in names
        ],
    }
}

__hf_hub_stable_v1_5_controlnet_dict__ = (
    lambda name, base="runwayml/stable-diffusion-v1-5": {
        **__hf_hub_stable_v1_5_dict__(base),
        **__hf_hub_controlnet_dict__(name),
    }
)

__hf_hub_stable_v1_5_safetensors_controlnet_dict__ = (
    lambda name, base="runwayml/stable-diffusion-v1-5": {
        **__hf_hub_stable_v1_5_safetensors_dict__(base),
        **__hf_hub_controlnet_dict__(name),
    }
)

__hf_hub_stable_xl_controlnet_dict__ = (
    lambda name, base="stabilityai/stable-diffusion-xl-base-1.0": {
        **__hf_hub_stable_xl_dict__(base),
        **__hf_hub_controlnet_dict__(name),
    }
)

__hf_hub_stable_xl_safetensors_controlnet_dict__ = (
    lambda name, base="stabilityai/stable-diffusion-xl-base-1.0": {
        **__hf_hub_stable_xl_safetensors_dict__(base),
        **__hf_hub_controlnet_dict__(name),
    }
)

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
    "stable-v2": __hf_hub_stable_v2_dict__("stabilityai/stable-diffusion-2"),
    "stable-v2.1": __hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-1"),
    "stable-xl-base": __hf_hub_stable_xl_safetensors_dict__(
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),
    "stable-xl-dreamshaper": __hf_hub_stable_xl_safetensors_dict__(
        "Lykon/dreamshaper-xl-1-0"
    ),
    "stable-xl-turbo": __hf_hub_stable_xl_safetensors_dict__("stabilityai/sdxl-turbo"),
    "stable-xl-realism-v30": __hf_hub_stable_xl_dict__(
        "stablediffusionapi/realism-engine-sdxl-v30"
    ),
    "stable-xl-realism-v30-fp16": {
        **__hf_hub_stable_xl_dict__("stablediffusionapi/realism-engine-sdxl-v30"),
        **{
            "vae": {
                "config": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json",
                "weight": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors",
            },
        },
    },
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
    "stable-v1.5-multicontrolnet-canny-depth": __hf_hub_multicontrolnet_dict__(
        "lllyasviel/sd-controlnet-canny",
        "lllyasviel/control_v11f1p_sd15_depth",
    ),
    "stable-v1.5-animate-v1.5": {
        "motion": {
            "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5/resolve/main/config.json",
            "weight": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5/resolve/main/diffusion_pytorch_model.safetensors",
        },
    },
    "stable-v1.5-animate-v1.5.2": {
        "motion": {
            "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
            "weight": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/diffusion_pytorch_model.safetensors",
        },
    },
    "stable-v1.5-animate-v1.5.2-zoom-in": {
        "motion": {
            "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
            "weight": f"https://huggingface.co/fuliucansheng/diffusers/resolve/main/animatediff/diffusion_pytorch_model.zoom_in.safetensors",
        },
    },
    "stable-v1.5-animate-v1.5.2-zoom-out": {
        "motion": {
            "config": f"https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/config.json",
            "weight": f"https://huggingface.co/fuliucansheng/diffusers/resolve/main/animatediff/diffusion_pytorch_model.zoom_out.safetensors",
        },
    },
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
    "stable-xl-multicontrolnet-canny-depth": __hf_hub_multicontrolnet_dict__(
        "diffusers/controlnet-canny-sdxl-1.0",
        "diffusers/controlnet-depth-sdxl-1.0",
    ),
    "stable-xl-t2i-adapter-canny": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-canny-sdxl-1.0"
    ),
    "stable-xl-t2i-adapter-sketch": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0"
    ),
    "stable-xl-t2i-adapter-openpose": __hf_hub_controlnet_safetensors_dict__(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0"
    ),
    "stable-xl-refiner-1.0": {
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
}

stable_version = lambda name: "-".join(name.split("-")[:2])
stable_name = lambda name: "-".join(name.split("-")[2:])
stable_combine_name = (
    lambda base, *names: f"{base}-{'-'.join(stable_name(n) for n in names)}"
)

pretrained_diffusers_infos = {
    **pretrained_stable_infos,
    **{
        stable_combine_name(base, name): {**dict1, **dict2}
        for base, dict1 in pretrained_stable_infos.items()
        for name, dict2 in pretrained_stable_extensions_infos.items()
        if stable_version(name) == stable_version(base)
    },
    "stable-v1.5-x4-upscaler": __hf_hub_stable_v1_5_dict__(
        "stabilityai/stable-diffusion-x4-upscaler"
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
from unitorch.cli.models.diffusers.modeling_multicontrolnet import (
    MultiControlNetForText2ImageGeneration,
    MultiControlNetForImage2ImageGeneration,
    MultiControlNetForImageInpainting,
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
from unitorch.cli.models.diffusers.processing_multicontrolnet import (
    MultiControlNetProcessor,
)

import unitorch.cli.models.diffusers.modeling_animate
