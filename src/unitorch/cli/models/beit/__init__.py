# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_beit_infos = {
    "beit-base-patch16-224-pt22k-ft22k": {
        "config": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/pytorch_model.bin"
        ),
    },
    "beit-base-patch16-224": {
        "config": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/microsoft/beit-base-patch16-224/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.beit.modeling
import unitorch.cli.models.beit.processing
from unitorch.cli.models.beit.modeling import BeitForImageClassification
from unitorch.cli.models.beit.processing import BeitProcessor
