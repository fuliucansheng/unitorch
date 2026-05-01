# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from typing import Dict

from unitorch import is_diffusers_available, is_megatron_available, is_opencv_available
from unitorch.cli.models.modeling_utils import (
    LossOutputs,
    ModelInputs,
    ModelOutputs,
    ModelTargets,
    TensorBatchMixin,
    TensorInputs,
    TensorMixInputs,
    TensorMixMixin,
    TensorMixOutputs,
    TensorMixTargets,
    TensorOutputs,
    TensorSeqInputs,
    TensorSeqMixin,
    TensorSeqOutputs,
    TensorSeqTargets,
    TensorTargets,
)

ACT2FN: Dict = {
    "relu": lambda v: np.maximum(v, 0),
    "tanh": np.tanh,
    "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
    "softmax": lambda v: np.exp(v) / np.sum(np.exp(v), axis=-1, keepdims=True),
}

from unitorch.cli.models.classification_utils import (
    ClassificationOutputs,
    ClassificationTargets,
    EmbeddingOutputs,
)
from unitorch.cli.models.detection_utils import (
    DetectionOutputs,
    DetectionTargets,
    detection_model_decorator,
)
from unitorch.cli.models.generation_utils import (
    GenerationOutputs,
    GenerationTargets,
    generation_model_decorator,
)
from unitorch.cli.models.ranking_utils import RankingOutputs, RankingTargets
from unitorch.cli.models.segmentation_utils import (
    SegmentationOutputs,
    SegmentationTargets,
    segmentation_model_decorator,
)

import unitorch.cli.models.image_utils
import unitorch.cli.models.label_utils
import unitorch.cli.models.processing_utils
import unitorch.cli.models.random_utils

if is_opencv_available():
    import unitorch.cli.models.video_utils

if is_diffusers_available():
    from unitorch.cli.models.diffusion_utils import (
        DiffusionOutputs,
        DiffusionTargets,
        diffusion_model_decorator,
    )
    import unitorch.cli.models.diffusers

if is_megatron_available():
    import unitorch.cli.models.megatron

import unitorch.cli.models.bart
import unitorch.cli.models.beit
import unitorch.cli.models.bert
import unitorch.cli.models.blip
import unitorch.cli.models.bria
import unitorch.cli.models.chinese_clip
import unitorch.cli.models.clip
import unitorch.cli.models.detr
import unitorch.cli.models.dinov2
import unitorch.cli.models.dpt
import unitorch.cli.models.grounding_dino
import unitorch.cli.models.kolors
import unitorch.cli.models.llama
import unitorch.cli.models.llava
import unitorch.cli.models.mask2former
import unitorch.cli.models.mbart
import unitorch.cli.models.mistral
import unitorch.cli.models.pegasus
import unitorch.cli.models.peft
import unitorch.cli.models.qwen
import unitorch.cli.models.roberta
import unitorch.cli.models.sam
import unitorch.cli.models.segformer
import unitorch.cli.models.siglip
import unitorch.cli.models.swin
import unitorch.cli.models.t5
import unitorch.cli.models.visualbert
import unitorch.cli.models.vit
import unitorch.cli.models.xlm_roberta
import unitorch.cli.models.xpegasus
