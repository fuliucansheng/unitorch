# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import collections.abc
from itertools import repeat


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
    v,
    divisor=8,
    min_value=None,
    round_limit=0.9,
):
    """
    Rounds a value to the nearest multiple of a divisor while ensuring it satisfies certain constraints.

    Args:
        v (int): The input value to be made divisible.
        divisor (int, optional): The divisor value. Defaults to 8.
        min_value (int, optional): The minimum value allowed. Defaults to None.
        round_limit (float, optional): The rounding limit, expressed as a fraction of the original value.
            Specifies the maximum allowed decrease in the rounded value compared to the original value.
            Defaults to 0.9, indicating a limit of 10%.

    Returns:
        int: The rounded value that is divisible by the given divisor and satisfies the constraints.

    Notes:
        - If min_value is not provided (None), it defaults to the value of the divisor.
        - The function rounds the input value (v) to the nearest multiple of the divisor.
        - If the rounded value is less than round_limit times the original value, the function increments
          the rounded value by the divisor to ensure the round down does not go down by more than the limit.

    Example:
        >>> make_divisible(25)
        24
        >>> make_divisible(32, divisor=10)
        30
        >>> make_divisible(10, divisor=3, min_value=6)
        12
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
