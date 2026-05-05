# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Optional, Set

import torch


def convert_tensor_to_strings(inputs: torch.Tensor) -> Any:
    """Recursively convert a tensor (scalar, 1-D, or N-D) to nested string lists."""
    if inputs.dim() == 0:
        return str(inputs.item())
    if inputs.dim() == 1:
        return [str(element.item()) for element in inputs]
    return [convert_tensor_to_strings(tensor) for tensor in inputs]


def remove_strings_ignore_tokens(inputs: Any, ignore_tokens: Optional[Set[str]]) -> Any:
    """Recursively remove strings in *ignore_tokens* from any nested structure.

    When *ignore_tokens* is ``None`` the input is returned as-is.
    """
    if ignore_tokens is None:
        return inputs
    if isinstance(inputs, list):
        return [remove_strings_ignore_tokens(e, ignore_tokens) for e in inputs]
    if isinstance(inputs, dict):
        return {k: remove_strings_ignore_tokens(v, ignore_tokens) for k, v in inputs.items()}
    if isinstance(inputs, tuple):
        return tuple(remove_strings_ignore_tokens(e, ignore_tokens) for e in inputs)
    if isinstance(inputs, frozenset):
        return frozenset(remove_strings_ignore_tokens(e, ignore_tokens) for e in inputs)
    if isinstance(inputs, set):
        return {remove_strings_ignore_tokens(e, ignore_tokens) for e in inputs}
    return inputs
