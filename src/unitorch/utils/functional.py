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


def update_nested_dict(mapping: Dict, *keys_and_value: Any) -> None:
    """Set a value inside an arbitrarily nested dictionary, creating sub-dicts as needed.

    The last element of *keys_and_value* is treated as the value to assign;
    all preceding elements are the key path.  Example::

        update_nested_dict(d, "a", "b", 42)  # d["a"]["b"] = 42

    Args:
        mapping: The top-level dictionary to update (modified in-place).
        *keys_and_value: One or more keys followed by the final value.

    Raises:
        ValueError: If fewer than two arguments are provided (need at least one
                    key and a value).
    """
    if len(keys_and_value) < 2:
        raise ValueError("update_nested_dict requires at least one key and a value.")
    *keys, value = keys_and_value
    for key in keys[:-1]:
        if not isinstance(mapping.get(key), dict):
            mapping[key] = {}
        mapping = mapping[key]
    mapping[keys[-1]] = value
