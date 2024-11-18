# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_xlm_roberta_infos = {
    "xlm-roberta-base": {
        "config": hf_endpoint_url("/xlm-roberta-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url(
            "/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"
        ),
        "weight": hf_endpoint_url("/xlm-roberta-base/resolve/main/pytorch_model.bin"),
    },
    "xlm-roberta-large": {
        "config": hf_endpoint_url("/xlm-roberta-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url(
            "/xlm-roberta-large/resolve/main/sentencepiece.bpe.model"
        ),
        "weight": hf_endpoint_url("/xlm-roberta-large/resolve/main/pytorch_model.bin"),
    },
    "xlm-roberta-large": {
        "config": hf_endpoint_url("/xlm-roberta-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url(
            "/xlm-roberta-large/resolve/main/sentencepiece.bpe.model"
        ),
        "weight": hf_endpoint_url("/xlm-roberta-large/resolve/main/pytorch_model.bin"),
    },
    "xlm-roberta-large-finetuned-conll03-english": {
        "config": hf_endpoint_url(
            "/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model"
        ),
        "weight": hf_endpoint_url(
            "/xlm-roberta-large-finetuned-conll03-english/resolve/main/pytorch_model.bin"
        ),
    },
    "xlm-roberta-xl": {
        "config": hf_endpoint_url("/facebook/xlm-roberta-xl/resolve/main/config.json"),
        "vocab": hf_endpoint_url(
            "/facebook/xlm-roberta-xl/resolve/main/sentencepiece.bpe.model"
        ),
        "weight": hf_endpoint_url(
            "/facebook/xlm-roberta-xl/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.xlm_roberta.modeling
import unitorch.cli.models.xlm_roberta.processing
from unitorch.cli.models.xlm_roberta.modeling import (
    XLMRobertaForClassification,
    XLMRobertaXLForClassification,
)
from unitorch.cli.models.xlm_roberta.processing import XLMRobertaProcessor
