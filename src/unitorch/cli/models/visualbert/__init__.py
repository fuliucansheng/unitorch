# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos

pretrained_visualbert_infos = {
    "default-visualbert": {
        "config": "https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    },
    "visualbert-vqa-coco-pre": {
        "config": "https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/resolve/main/pytorch_model.bin",
    },
    "visualbert-nlvr2-coco-pre": {
        "config": "https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.visualbert.modeling
import unitorch.cli.models.visualbert.processing
from unitorch.cli.models.visualbert.modeling import (
    VisualBertForClassification,
    VisualBertForPretrain,
)
from unitorch.cli.models.visualbert.processing import VisualBertProcessor
