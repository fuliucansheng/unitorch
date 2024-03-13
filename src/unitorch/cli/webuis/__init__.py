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


import unitorch.cli.webuis.image_utils
import unitorch.cli.webuis.animate
import unitorch.cli.webuis.blip2
import unitorch.cli.webuis.bloom
import unitorch.cli.webuis.stable
import unitorch.cli.webuis.controlnet
import unitorch.cli.webuis.stable_xl
import unitorch.cli.webuis.stable_xl_refiner
import unitorch.cli.webuis.controlnet_xl
import unitorch.cli.webuis.llama
import unitorch.cli.webuis.minigpt4
import unitorch.cli.webuis.mistral
import unitorch.cli.webuis.sam
