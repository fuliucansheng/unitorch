# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random

ACT2FN = {
    "relu": lambda v: np.maximum(v, 0),
    "tanh": np.tanh,
    "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
    "softmax": lambda v: np.exp(v) / np.sum(np.exp(v), axis=-1, keepdims=True),
}
from unitorch import is_diffusers_available

# import modeling utils
from unitorch.cli.models.modeling_utils import (
    ModelInputs,
    ModelOutputs,
    ModelTargets,
    TensorsInputs,
    ListTensorsInputs,
    CombineTensorsInputs,
    TensorsOutputs,
    ListTensorsOutputs,
    CombineTensorsOutputs,
    TensorsTargets,
    ListTensorsTargets,
    CombineTensorsTargets,
    LossOutputs,
)
from unitorch.cli.models.classification_utils import (
    ClassificationOutputs,
    EmbeddingOutputs,
    ClassificationTargets,
)
from unitorch.cli.models.detection_utils import (
    DetectionOutputs,
    DetectionTargets,
)
from unitorch.cli.models.detection_utils import detection_model_decorator
from unitorch.cli.models.generation_utils import (
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.generation_utils import generation_model_decorator
from unitorch.cli.models.ranking_utils import RankingOutputs, RankingTargets
from unitorch.cli.models.segmentation_utils import (
    SegmentationOutputs,
    SegmentationTargets,
)
from unitorch.cli.models.segmentation_utils import segmentation_model_decorator

# import processing utils
import unitorch.cli.models.image_utils
import unitorch.cli.models.random_utils
import unitorch.cli.models.label_utils
import unitorch.cli.models.processing_utils

if is_diffusers_available():
    from unitorch.cli.models.diffusion_utils import DiffusionOutputs, DiffusionTargets
    from unitorch.cli.models.diffusion_utils import diffusion_model_decorator
    import unitorch.cli.models.diffusers

# import model classes & process functions
import unitorch.cli.models.bart
import unitorch.cli.models.bert
import unitorch.cli.models.beit
import unitorch.cli.models.blip
import unitorch.cli.models.bloom
import unitorch.cli.models.bria
import unitorch.cli.models.chinese_clip
import unitorch.cli.models.clip
import unitorch.cli.models.detr
import unitorch.cli.models.dinov2
import unitorch.cli.models.dpt
import unitorch.cli.models.grounding_dino
import unitorch.cli.models.llama
import unitorch.cli.models.llava
import unitorch.cli.models.mask2former
import unitorch.cli.models.mbart
import unitorch.cli.models.mistral
import unitorch.cli.models.mt5
import unitorch.cli.models.pegasus
import unitorch.cli.models.peft
import unitorch.cli.models.roberta
import unitorch.cli.models.swin
import unitorch.cli.models.sam
import unitorch.cli.models.segformer
import unitorch.cli.models.t5
import unitorch.cli.models.visualbert
import unitorch.cli.models.vit
import unitorch.cli.models.xlm_roberta
import unitorch.cli.models.xpegasus
