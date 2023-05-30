# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.models.deberta.modeling import DebertaForClassification, DebertaForMaskLM
from unitorch.models.deberta.processing import DebertaProcessor, get_deberta_tokenizer
from unitorch.models.deberta.modeling_v2 import DebertaV2ForClassification
from unitorch.models.deberta.processing_v2 import (
    DebertaV2Processor,
    get_deberta_v2_tokenizer,
)
