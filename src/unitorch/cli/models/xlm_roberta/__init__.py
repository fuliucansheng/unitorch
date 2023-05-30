# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos

pretrained_xlm_roberta_infos = {
    "default-xlm-roberta": {
        "config": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
    },
    "xlm-roberta-base": {
        "config": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin",
    },
    "xlm-roberta-large": {
        "config": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/xlm-roberta-large/resolve/main/pytorch_model.bin",
    },
    "xlm-roberta-large": {
        "config": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/xlm-roberta-large/resolve/main/pytorch_model.bin",
    },
    "xlm-roberta-large-finetuned-conll03-english": {
        "config": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json",
        "vocab": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/pytorch_model.bin",
    },
    "xlm-roberta-xl": {
        "config": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/sentencepiece.bpe.model",
        "weight": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.xlm_roberta.modeling
import unitorch.cli.models.xlm_roberta.processing
from unitorch.cli.models.xlm_roberta.modeling import (
    XLMRobertaForClassification,
    XLMRobertaXLForClassification,
)
from unitorch.cli.models.xlm_roberta.processing import XLMRobertaProcessor
