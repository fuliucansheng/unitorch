# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def matched_pretrained_names(
    pretrained_names: List[str],
    match_patterns: Union[str, List[str]],
    block_patterns: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    if isinstance(match_patterns, str):
        match_patterns = [match_patterns]
    if block_patterns is None:
        block_patterns = []
    elif isinstance(block_patterns, str):
        block_patterns = [block_patterns]
    matched = [
        p for p in pretrained_names if any([re.match(m, p) for m in match_patterns])
    ]
    blacked = [p for p in matched if not any([re.match(b, p) for b in block_patterns])]
    return blacked


from unitorch.utils import is_diffusers_available
from unitorch.cli import GenericWebUI
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines import Schedulers

supported_scheduler_names = list(Schedulers.keys())


class SimpleWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser, iname="Simple", iface=None):
        self._config = config
        self._iname = iname
        self._iface = iface

    @property
    def iname(self):
        return self._iname

    @property
    def iface(self):
        return self._iface

    def start(self):
        pass

    def stop(self):
        pass


from unitorch.cli.webuis.utils import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
    create_controlnet_layout,
    create_lora_layout,
    create_freeu_layout,
)

import unitorch.cli.webuis.tools
import unitorch.cli.webuis.blip
import unitorch.cli.webuis.bloom
import unitorch.cli.webuis.bria
import unitorch.cli.webuis.detr
import unitorch.cli.webuis.dpt
import unitorch.cli.webuis.llama
import unitorch.cli.webuis.mistral
import unitorch.cli.webuis.sam

if is_diffusers_available():
    import unitorch.cli.webuis.stable
    import unitorch.cli.webuis.stable_xl
    import unitorch.cli.webuis.stable_3
