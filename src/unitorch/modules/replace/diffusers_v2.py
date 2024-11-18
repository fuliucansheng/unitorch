# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import diffusers
from diffusers.pipelines import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    VaeImageProcessor,
)
from unitorch.models import GenericOutputs
from unitorch.utils.decorators import replace


@replace(
    diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline
)
class StableDiffusion3Img2ImgPipelineV2(StableDiffusion3Img2ImgPipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer
            if tokenizer is not None
            else GenericOutputs(model_max_length=77),
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2
            if tokenizer_2 is not None
            else GenericOutputs(model_max_length=77),
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3
            if tokenizer_3 is not None
            else GenericOutputs(model_max_length=77),
        )


@replace(
    diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline
)
class StableDiffusion3InpaintPipelineV2(StableDiffusion3InpaintPipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer
            if tokenizer is not None
            else GenericOutputs(model_max_length=77),
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2
            if tokenizer_2 is not None
            else GenericOutputs(model_max_length=77),
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3
            if tokenizer_3 is not None
            else GenericOutputs(model_max_length=77),
        )
