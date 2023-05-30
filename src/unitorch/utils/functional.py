# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def rpartial(
    func,
    *args,
    **kwargs,
):
    """
    Create a new function that partially applies the given arguments to the provided callable.

    Args:
        func: The callable object or function to be partially applied.
        *args: Positional arguments to be partially applied.
        **kwargs: Keyword arguments to be partially applied.

    Returns:
        A new function that, when called, will invoke `func` with the original and partially applied arguments.
    """
    return lambda *a, **kw: func(*(args + a), **{**kwargs, **kw})


def pop_value(
    *args,
    msg: Optional[str] = "default error msg",
    first: Optional[bool] = True,
    last: Optional[bool] = False,
    check_none: Optional[bool] = True,
) -> Any:
    """
    Returns the first non-None value from the list of arguments.

    Args:
        *args: A list of Python values.
        msg (str, optional): Error message to be raised if no non-None value is found (default: "default error msg").
        first (bool, optional): If True, returns the first non-None value. If False, returns the last non-None value (default: True).
        last (bool, optional): If True, returns the last non-None value. If False, returns the first non-None value (default: False).
        check_none (bool, optional): If True, raises a ValueError if no non-None value is found. If False, returns None (default: True).

    Raises:
        ValueError: If check_none is True and no non-None value is found.

    Returns:
        The first non-None value if found, None otherwise.
    """
    assert first is True or last is True

    if first:
        for arg in args:
            if arg is not None:
                return arg
    else:
        for arg in reversed(args):
            if arg is not None:
                return arg

    if check_none:
        raise ValueError(f"{msg} Can't find non-None value")

    return None


def truncate_sequence_pair(
    tokens: List[str], tokens_pair: List[str], max_length: int
) -> None:
    while True:
        total_length = len(tokens) + len(tokens_pair)
        if total_length <= max_length:
            break
        if len(tokens) > len(tokens_pair):
            tokens.pop()
        else:
            tokens_pair.pop()


nested_dict_value = (
    lambda a, b, *c: (
        nested_dict_value(a[b], *c) if isinstance(a[b], dict) and len(c) > 0 else a[b]
    )
    if b in a
    else None
)
