# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_swin_infos = {
    "swin-tiny-patch4-window7-224": {
        "config": hf_endpoint_url(
            "/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-tiny-patch4-window7-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-tiny-patch4-window7-224/resolve/main/pytorch_model.bin"
        ),
    },
    "swin-base-patch4-window7-224": {
        "config": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224/resolve/main/pytorch_model.bin"
        ),
    },
    "swin-large-patch4-window7-224": {
        "config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224/resolve/main/pytorch_model.bin"
        ),
    },
    "swin-large-patch4-window12-384": {
        "config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window12-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window12-384/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window12-384/resolve/main/pytorch_model.bin"
        ),
    },
    "swin-base-patch4-window7-224-in22k": {
        "config": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/pytorch_model.bin"
        ),
    },
    "swin-large-patch4-window7-224-in22k": {
        "config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.swin.modeling
import unitorch.cli.models.swin.processing
from unitorch.cli.models.swin.modeling import SwinForImageClassification
from unitorch.cli.models.swin.processing import SwinProcessor
