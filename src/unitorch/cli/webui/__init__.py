# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def matched_pretrained_names(
    pretrained_names: List[str],
    match_patterns: List[str],
    block_patterns: List[str] = [],
) -> List[str]:
    matched = [
        p for p in pretrained_names if any([re.match(m, p) for m in match_patterns])
    ]
    blacked = [p for p in matched if not any([re.match(b, p) for b in block_patterns])]
    return blacked


import unitorch.cli.webui.image_utils
import unitorch.cli.webui.animate
import unitorch.cli.webui.blip2
import unitorch.cli.webui.controlnet
import unitorch.cli.webui.controlnet_xl
import unitorch.cli.webui.stable
import unitorch.cli.webui.stable_xl
import unitorch.cli.webui.stable_xl_refiner
import unitorch.cli.webui.llama
import unitorch.cli.webui.minigpt4
