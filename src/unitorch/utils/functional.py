# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Dict, List


def rpartial(func, *args, **kwargs):
    """Return a callable that prepends *args* and *kwargs* to any future call.

    Unlike :func:`functools.partial`, the partially-applied positional arguments
    come *before* the arguments supplied at call time.

    Args:
        func: The callable to wrap.
        *args: Positional arguments prepended on each call.
        **kwargs: Keyword arguments merged (overridable) on each call.

    Returns:
        A lambda that calls ``func(*(args + call_args), **{**kwargs, **call_kwargs})``.
    """
    return lambda *a, **kw: func(*(args + a), **{**kwargs, **kw})


def pop_value(
    *args,
    msg: str = "default error msg",
    first: bool = True,
    last: bool = False,
    check_none: bool = True,
) -> Any:
    """Return the first (or last) non-``None`` value from *args*.

    Args:
        *args: Candidate values to inspect.
        msg: Prefix for the :exc:`ValueError` message when no value is found.
        first: When ``True`` (the default), return the first non-``None`` value.
        last: When ``True``, return the last non-``None`` value.
               At least one of *first* or *last* must be ``True``.
        check_none: When ``True``, raise :exc:`ValueError` if every argument is
                    ``None``; otherwise return ``None`` silently.

    Returns:
        The selected non-``None`` value, or ``None`` when *check_none* is ``False``
        and no such value exists.

    Raises:
        AssertionError: If neither *first* nor *last* is ``True``.
        ValueError: If *check_none* is ``True`` and no non-``None`` value is found.
    """
    assert first or last, "At least one of 'first' or 'last' must be True."

    candidates = args if first else reversed(args)
    for arg in candidates:
        if arg is not None:
            return arg

    if check_none:
        raise ValueError(f"{msg}: no non-None value found.")
    return None


def truncate_sequence_pair(
    tokens: List[Any],
    tokens_pair: List[Any],
    max_length: int,
) -> None:
    """Truncate two token lists in-place until their combined length ≤ *max_length*.

    The longer list is always shortened first; ties favour *tokens_pair*.

    Args:
        tokens: First token list (modified in-place).
        tokens_pair: Second token list (modified in-place).
        max_length: Target combined maximum length.
    """
    while len(tokens) + len(tokens_pair) > max_length:
        if len(tokens) > len(tokens_pair):
            tokens.pop()
        else:
            tokens_pair.pop()


def nested_dict_value(mapping: Dict, key: Any, *keys: Any) -> Any:
    """Retrieve a value from an arbitrarily nested dictionary.

    Args:
        mapping: The top-level dictionary to search.
        key: Key at the current nesting level.
        *keys: Additional keys for deeper nesting levels.

    Returns:
        The value at the specified path, or ``None`` if any key is missing.
    """
    if key not in mapping:
        return None
    if isinstance(mapping[key], dict) and keys:
        return nested_dict_value(mapping[key], *keys)
    return mapping[key]


def update_nested_dict(mapping: Dict, key: Any, value: Any, *keys: Any) -> None:
    """Set a value inside an arbitrarily nested dictionary, creating sub-dicts as needed.

    Args:
        mapping: The top-level dictionary to update (modified in-place).
        key: Key at the current nesting level.
        value: Value to assign, or the next-level key when *keys* is non-empty.
        *keys: Additional ``(key, …, value)`` path components for deeper nesting.
    """
    if key not in mapping:
        mapping[key] = {}
    if isinstance(mapping[key], dict) and keys:
        update_nested_dict(mapping[key], value, *keys)
    else:
        mapping[key] = value
