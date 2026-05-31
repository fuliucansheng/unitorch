# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import collections.abc
from itertools import repeat
from typing import Optional


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: Optional[int] = None,
    round_limit: float = 0.9,
) -> int:
    """Round *v* up to the nearest multiple of *divisor*.

    Args:
        v: Value to round.
        divisor: Target divisor. Defaults to 8.
        min_value: Floor for the result; defaults to *divisor*.
        round_limit: If the rounded value would be less than
            ``round_limit * v``, add one more *divisor* step. Defaults to 0.9.

    Returns:
        Smallest multiple of *divisor* that is ≥ *min_value* and does not
        decrease *v* by more than ``(1 - round_limit) * 100`` percent.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
