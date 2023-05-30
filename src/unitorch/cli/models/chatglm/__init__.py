# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_chatglm_infos = {
    # chatglm
    "default-chatglm": {
        "config": "https://huggingface.co/THUDM/chatglm-6b/resolve/main/config.json",
        "vocab": "https://huggingface.co/THUDM/chatglm-6b/resolve/main/ice_text.model",
        "tokenizer": "https://huggingface.co/THUDM/chatglm-6b/raw/main/tokenizer_config.json",
    },
    "chatglm-6b": {
        "config": "https://huggingface.co/THUDM/chatglm-6b/resolve/main/config.json",
        "vocab": "https://huggingface.co/THUDM/chatglm-6b/resolve/main/ice_text.model",
        "tokenizer": "https://huggingface.co/THUDM/chatglm-6b/raw/main/tokenizer_config.json",
        "weight": [
            f"https://huggingface.co/THUDM/chatglm-6b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00008.bin"
            for i in range(1, 9)
        ],
    },
}

import unitorch.cli.models.chatglm.modeling
import unitorch.cli.models.chatglm.processing
from unitorch.cli.models.chatglm.modeling import (
    ChatGLMForClassification,
    ChatGLMForPretrain,
    ChatGLMForGeneration,
)
from unitorch.cli.models.chatglm.processing import ChatGLMProcessor