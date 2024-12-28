# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.bin"
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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.bin"
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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors"
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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_flux_safetensors_dict__ = lambda name: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-00003.safetensors"
            )
            for i in range(1, 4)
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
                f"/{name}/resolve/main/text_encoder_2/model-{str(i).rjust(5, '0')}-of-00002.safetensors"
            )
            for i in range(1, 3)
        ],
    },
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_stable_flux_ctrl_safetensors_dict__ = lambda name: {
    "transformer": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/transformer/config.json"),
        "weight": [
            hf_endpoint_url(
                f"/{name}/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-00003.safetensors"
            )
            for i in range(1, 4)
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
                f"/{name}/resolve/main/text_encoder_2/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            )
            for i in range(1, 5)
        ],
    },
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

__hf_hub_controlnet_dict__ = lambda name: {
    "controlnet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/config.json"),
        "weight": hf_endpoint_url(f"/{name}/resolve/main/diffusion_pytorch_model.bin"),
    }
}

__hf_hub_controlnet_safetensors_dict__ = lambda name: {
    "controlnet": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/diffusion_pytorch_model.safetensors"
        ),
    }
}

__hf_hub_adapter_dict__ = lambda name: {
    "adapter": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/config.json"),
        "weight": hf_endpoint_url(f"/{name}/resolve/main/diffusion_pytorch_model.bin"),
    }
}

__hf_hub_adapter_safetensors_dict__ = lambda name: {
    "adapter": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/diffusion_pytorch_model.safetensors"
        ),
    }
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
    "vae": {
        "config": hf_endpoint_url(f"/{name}/resolve/main/vae/config.json"),
        "weight": hf_endpoint_url(
            f"/{name}/resolve/main/vae/diffusion_pytorch_model.safetensors"
        ),
    },
    "scheduler": hf_endpoint_url(
        f"/{name}/resolve/main/scheduler/scheduler_config.json"
    ),
}

pretrained_stable_infos = {
    "stable-v1.5": __hf_hub_stable_v1_5_dict__("botp/stable-diffusion-v1-5"),
    "stable-v1.5-realistic-v5.1-no-vae": __hf_hub_stable_v1_5_safetensors_dict__(
        "SG161222/Realistic_Vision_V5.1_noVAE"
    ),
    "stable-v1.5-realistic-v5.1": __hf_hub_stable_v1_5_dict__(
        "stablediffusionapi/realistic-vision-v51"
    ),
    "stable-v1.5-film": __hf_hub_stable_v1_5_safetensors_dict__("Yntec/Film"),
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
        "botp/stable-diffusion-v1-5-inpainting"
    ),
    "stable-v1.5-realistic-v5.1-inpainting": __hf_hub_stable_v1_5_safetensors_dict__(
        "Uminosachi/realisticVisionV51_v51VAE-inpainting"
    ),
    "stable-v1.5-dreamshaper-8-inpainting": __hf_hub_stable_v1_5_safetensors_dict__(
        "Lykon/dreamshaper-8-inpainting"
    ),
    "stable-v1.5-x4-upscaler": __hf_hub_stable_v1_5_dict__(
        "stabilityai/stable-diffusion-x4-upscaler"
    ),
    "stable-v2": __hf_hub_stable_v2_dict__("stabilityai/stable-diffusion-2"),
    "stable-v2.1": __hf_hub_stable_v2_1_dict__("stabilityai/stable-diffusion-2-1"),
    "stable-xl-base": __hf_hub_stable_xl_safetensors_dict__(
        "stabilityai/stable-diffusion-xl-base-1.0"
    ),
    "stable-xl-base-vae-fp16": {
        **__hf_hub_stable_xl_safetensors_dict__(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ),
        **{
            "vae": {
                "config": hf_endpoint_url(
                    "/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json"
                ),
                "weight": hf_endpoint_url(
                    "/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors"
                ),
            },
        },
    },
    "stable-xl-turbo": __hf_hub_stable_xl_safetensors_dict__("stabilityai/sdxl-turbo"),
    "stable-xl-realism-engine-v30": __hf_hub_stable_xl_safetensors_dict__(
        "misri/realismEngineSDXL_v30VAE"
    ),
    "stable-xl-opendalle-v1.1": __hf_hub_stable_xl_safetensors_dict__(
        "dataautogpt3/OpenDalleV1.1"
    ),
    "stable-xl-realvis-v3.0": __hf_hub_stable_xl_safetensors_dict__(
        "SG161222/RealVisXL_V3.0"
    ),
    "stable-xl-juggernaut-v8": __hf_hub_stable_xl_dict__(
        "RunDiffusion/Juggernaut-XL-v8"
    ),
    "stable-xl-playground-v2-aesthetic": __hf_hub_stable_xl_safetensors_dict__(
        "playgroundai/playground-v2-1024px-aesthetic"
    ),
    "stable-v3-medium": __hf_hub_stable_3_safetensors_dict__(
        "ckpt/stable-diffusion-3-medium-diffusers"
    ),
    "stable-v3.5-medium": __hf_hub_stable_3_safetensors_dict__(
        "ckpt/stable-diffusion-3.5-medium"
    ),
    "stable-v3.5-large": __hf_hub_stable_3_safetensors_dict__(
        "yuvraj108c/stable-diffusion-3.5-large"
    ),
    "stable-flux-schnell": __hf_hub_stable_flux_safetensors_dict__(
        "black-forest-labs/FLUX.1-schnell"
    ),
    "stable-flux-dev": __hf_hub_stable_flux_safetensors_dict__(
        "camenduru/FLUX.1-dev-diffusers"
    ),
    "stable-flux-dev-redux": {
        **__hf_hub_stable_flux_safetensors_dict__("camenduru/FLUX.1-dev-diffusers"),
        **{
            "image": {
                "config": hf_endpoint_url(
                    "/tentpole/flux1-dev-redux/resolve/main/image_encoder/config.json"
                ),
                "vision_config": hf_endpoint_url(
                    "/tentpole/flux1-dev-redux/resolve/main/feature_extractor/preprocessor_config.json"
                ),
                "weight": hf_endpoint_url(
                    "/tentpole/flux1-dev-redux/resolve/main/image_encoder/model.safetensors"
                ),
            },
            "redux_image": {
                "config": hf_endpoint_url(
                    "/tentpole/flux1-dev-redux/resolve/main/image_embedder/config.json"
                ),
                "weight": hf_endpoint_url(
                    "/tentpole/flux1-dev-redux/resolve/main/image_embedder/diffusion_pytorch_model.safetensors"
                ),
            },
        },
    },
    "stable-flux-dev-fill": __hf_hub_stable_flux_safetensors_dict__(
        "fuliucansheng/FLUX.1-Fill-dev-diffusers"
    ),
    "stable-flux-dev-canny": __hf_hub_stable_flux_ctrl_safetensors_dict__(
        "fuliucansheng/FLUX.1-Canny-dev-diffusers"
    ),
    "stable-video-diffusion-img2vid-xt": __hf_hub_stable_video_safetensors_dict__(
        "stabilityai/stable-video-diffusion-img2vid-xt"
    ),
    "stable-video-diffusion-img2vid-xt-1-1": __hf_hub_stable_video_safetensors_dict__(
        "vdo/stable-video-diffusion-img2vid-xt-1-1"
    ),
}

pretrained_stable_extensions_infos = {
    # sd 1.5 controlnet
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
    "stable-v1.5-controlnet-tile": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11f1e_sd15_tile"
    ),
    "stable-v1.5-controlnet-lineart": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_lineart"
    ),
    "stable-v1.5-controlnet-softedge": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_softedge"
    ),
    "stable-v1.5-controlnet-scribble": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_scribble"
    ),
    "stable-v1.5-controlnet-openpose": __hf_hub_controlnet_dict__(
        "lllyasviel/control_v11p_sd15_openpose"
    ),
    # sd 1.5 adapter
    "stable-v1.5-adapter-canny": __hf_hub_adapter_dict__(
        "diffusers/t2iadapter_canny_sd15v2"
    ),
    "stable-v1.5-adapter-depth": __hf_hub_adapter_dict__(
        "diffusers/t2iadapter_depth_sd15v2"
    ),
    "stable-v1.5-adapter-sketch": __hf_hub_adapter_dict__(
        "diffusers/t2iadapter_sketch_sd15v2"
    ),
    # sdxl controlnet
    "stable-xl-controlnet-canny": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-canny-sdxl-1.0"
    ),
    "stable-xl-controlnet-softedge-dexined": __hf_hub_controlnet_dict__(
        "SargeZT/controlnet-sd-xl-1.0-softedge-dexined"
    ),
    "stable-xl-controlnet-depth": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-depth-sdxl-1.0"
    ),
    "stable-xl-controlnet-depth-small": __hf_hub_controlnet_dict__(
        "diffusers/controlnet-depth-sdxl-1.0-small"
    ),
    "stable-xl-controlnet-tile": __hf_hub_controlnet_safetensors_dict__(
        "xinsir/controlnet-tile-sdxl-1.0"
    ),
    "stable-xl-controlnet-openpose": __hf_hub_controlnet_safetensors_dict__(
        "xinsir/controlnet-openpose-sdxl-1.0"
    ),
    "stable-xl-controlnet-scribble": __hf_hub_controlnet_safetensors_dict__(
        "xinsir/controlnet-scribble-sdxl-1.0"
    ),
    # sdxl adapter
    "stable-xl-adapter-canny": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-canny-sdxl-1.0"
    ),
    "stable-xl-adapter-sketch": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0"
    ),
    "stable-xl-adapter-depth": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0"
    ),
    "stable-xl-adapter-depth-midas": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
    ),
    "stable-xl-adapter-openpose": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0"
    ),
    "stable-xl-adapter-lineart": __hf_hub_adapter_safetensors_dict__(
        "TencentARC/t2i-adapter-lineart-sdxl-1.0"
    ),
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
    # stable 3 controlnet
    "stable-v3-controlnet-canny": __hf_hub_controlnet_safetensors_dict__(
        "InstantX/SD3-Controlnet-Canny"
    ),
    "stable-v3-controlnet-tile": __hf_hub_controlnet_safetensors_dict__(
        "InstantX/SD3-Controlnet-Tile"
    ),
    # stable flux controlnet
    "stable-flux-controlnet-dev-union": __hf_hub_controlnet_safetensors_dict__(
        "InstantX/FLUX.1-dev-Controlnet-Union"
    ),
    "stable-flux-controlnet-dev-union-pro": __hf_hub_controlnet_safetensors_dict__(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
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
from unitorch.cli.models.diffusers.modeling_stable_flux import (
    StableFluxForText2ImageGeneration,
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
from unitorch.cli.models.diffusers.modeling_controlnet_flux import (
    ControlNetFluxForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_adapter import (
    StableAdapterForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_adapter_xl import (
    StableXLAdapterForText2ImageGeneration,
)
from unitorch.cli.models.diffusers.modeling_vae import VAEForDiffusion
from unitorch.cli.models.diffusers.processing_stable import StableProcessor
from unitorch.cli.models.diffusers.processing_stable_xl import StableXLProcessor
from unitorch.cli.models.diffusers.processing_stable_3 import Stable3Processor
from unitorch.cli.models.diffusers.processing_stable_flux import StableFluxProcessor
from unitorch.cli.models.diffusers.processing_controlnet import ControlNetProcessor
from unitorch.cli.models.diffusers.processing_controlnet_xl import ControlNetXLProcessor
from unitorch.cli.models.diffusers.processing_controlnet_3 import ControlNet3Processor
from unitorch.cli.models.diffusers.processing_controlnet_flux import (
    ControlNetFluxProcessor,
)
from unitorch.cli.models.diffusers.processing_adapter import AdapterProcessor
from unitorch.cli.models.diffusers.processing_adapter_xl import AdapterXLProcessor
