# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import inspect
import logging
from typing import Dict, Optional


def add_default_section_for_init(section: str, default_params: Optional[Dict] = None):
    """Class decorator for ``from_core_configure`` classmethods.

    Populates missing ``__init__`` parameters from *section* in the config,
    then attaches the config as ``__unitorch_setting__`` on the returned instance.
    """
    _defaults = default_params or {}

    def _build_instance(cls, config, **kwargs):
        for name, param in inspect.signature(cls).parameters.items():
            if kwargs.get(name) is None:
                kwargs[name] = config.getdefault(section, name, param.default)
        for name, fallback in _defaults.items():
            kwargs[name] = config.getdefault(section, name, fallback)
        obj = cls(**kwargs)
        obj.__unitorch_setting__ = config
        return obj

    def decorator(from_core_configure):
        def wrapped(cls, config, **kwargs):
            result = from_core_configure(cls, config, **kwargs)
            if isinstance(result, cls):
                result.__unitorch_setting__ = config
                return result
            assert result is None or isinstance(result, dict)
            if result is not None:
                kwargs.update(result)
            return _build_instance(cls, config, **kwargs)

        return wrapped

    return decorator


def add_default_section_for_function(section: str, default_params: Optional[Dict] = None):
    """Method decorator that fills missing parameters from the instance's config.

    Reads ``self.__unitorch_setting__`` (a ``CoreConfigureParser``) to resolve
    any argument not explicitly passed by the caller.
    """
    _defaults = default_params or {}

    def _resolve_kwargs(func, config, args, kwargs):
        for name, fallback in _defaults.items():
            if name not in kwargs:
                kwargs[name] = config.getdefault(section, name, fallback)
        for i, (name, param) in enumerate(inspect.signature(func).parameters.items()):
            if name == "self" or name in kwargs:
                continue
            positional = args[i] if i < len(args) else param.default
            kwargs[name] = config.getdefault(section, name, positional)
        return kwargs

    def decorator(func):
        def wrapped(*args, **kwargs):
            if args and hasattr(args[0], "__unitorch_setting__"):
                kwargs = _resolve_kwargs(func, args[0].__unitorch_setting__, args, kwargs)
                return func(args[0], **kwargs)
            logging.warning("__unitorch_setting__ not found; using default parameters.")
            return func(*args, **kwargs)

        return wrapped

    return decorator
