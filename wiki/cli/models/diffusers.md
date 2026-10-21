# unitorch.cli.models.diffusers

## FLUX.2

`core/process/diffusion/flux2` produces tensors for text-to-image and per-sample
image editing. `core/model/diffusers/text2image/flux2` and
`core/model/diffusers/editing/flux2` train the full diffusion transformer.
See [data, configuration and checkpoint commands](../../guides/flux2_finetuning.md).

::: unitorch.cli.models.diffusers.processing_flux2.Flux2Processor

::: unitorch.cli.models.diffusers.modeling_flux2.Flux2ForText2ImageGeneration

::: unitorch.cli.models.diffusers.modeling_flux2.Flux2ForImageEditingGeneration

## WanProcessor

!!! tip
    **`core/process/diffusion/wan`** is the section for configuration of WanProcessor.

::: unitorch.cli.models.diffusers.processing_wan.WanProcessor

## WanForText2VideoGeneration

!!! tip
    **`core/model/diffusers/text2video/wan`** is the section for configuration of WanForText2VideoGeneration.

The Wan example configs in this repository default to the unified `wan-v2.2-ti2v-5b` checkpoint for real text-to-video and image-to-video runs.

::: unitorch.cli.models.diffusers.modeling_wan.WanForText2VideoGeneration

## WanForImage2VideoGeneration

!!! tip
    **`core/model/diffusers/image2video/wan`** is the section for configuration of WanForImage2VideoGeneration.

::: unitorch.cli.models.diffusers.modeling_wan.WanForImage2VideoGeneration

## LucyProcessor

!!! tip
    **`core/process/diffusion/lucy`** is the section for configuration of LucyProcessor.

::: unitorch.cli.models.diffusers.processing_lucy.LucyProcessor

## LucyForVideoEditingGeneration

!!! tip
    **`core/model/diffusers/video_editing/lucy`** is the section for configuration of LucyForVideoEditingGeneration.

::: unitorch.cli.models.diffusers.modeling_lucy.LucyForVideoEditingGeneration

## QWenImageProcessor

!!! tip
    **`core/process/diffusion/qwen_image`** is the section for configuration of QWenImageProcessor.

::: unitorch.cli.models.diffusers.processing_qwen_image.QWenImageProcessor

## QWenImageText2ImageGeneration

!!! tip
    **`core/model/diffusers/text2image/qwen_image`** is the section for configuration of QWenImageText2ImageGeneration.

::: unitorch.cli.models.diffusers.modeling_qwen_image.QWenImageText2ImageGeneration

## QWenImageEditingGeneration

!!! tip
    **`core/model/diffusers/editing/qwen_image`** is the section for configuration of QWenImageEditingGeneration.

::: unitorch.cli.models.diffusers.modeling_qwen_image.QWenImageEditingGeneration
