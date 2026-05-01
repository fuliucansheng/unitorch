# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import functools
import inspect
import logging
import random
import sys
import time
from typing import Tuple, Type

OPTIMIZED_CLASSES: dict = {}


def replace(target_obj):
    """Replace a class or function in-place across all loaded modules.

    After decoration, every module that referenced *target_obj* will see the
    new object instead, including uses as a base class.

    Example::

        class A:
            def f(self):
                print('class A')

        @replace(A)
        class B:
            def f(self):
                print('class B')

    Args:
        target_obj: The class, method, or function to be replaced.

    Returns:
        A decorator that performs the replacement and returns the new object.
    """

    def decorator(new_obj):
        if target_obj in OPTIMIZED_CLASSES:
            logging.warning("%s has been replaced more than once.", target_obj)

        setattr(new_obj, "__replaced_class__", target_obj)
        OPTIMIZED_CLASSES[target_obj] = new_obj

        for module_name, module in list(sys.modules.items()):
            module_dict = module.__dict__

            # Replace direct references to target_obj.
            if (
                target_obj.__name__ in module_dict
                and module_dict[target_obj.__name__] is target_obj
            ):
                setattr(module, target_obj.__name__, new_obj)
                logging.debug(
                    "Module %s: replaced %s with %s.",
                    module_name, target_obj, new_obj,
                )

            # Replace target_obj where it appears as a base class.
            for attr_name in list(module_dict.keys()):
                attr = module_dict[attr_name]
                if (
                    inspect.isclass(attr)
                    and attr is not new_obj
                    and target_obj in attr.__bases__
                ):
                    bases = list(attr.__bases__)
                    bases[bases.index(target_obj)] = new_obj
                    attr.__bases__ = tuple(bases)
                    logging.debug(
                        "Module %s: base class of %s replaced with %s.",
                        module_name, attr, new_obj,
                    )

        return new_obj

    return decorator


def retry(
    times: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    jitter: bool = True,
):
    """Retry a function call with exponential back-off on failure.

    Args:
        times: Maximum number of attempts before re-raising the exception.
        base_delay: Initial delay in seconds between retries.
        max_delay: Upper bound on the computed delay.
        exceptions: Exception types that trigger a retry.
        jitter: When ``True``, multiply the delay by a random factor in
                ``[0.5, 1.5]`` to avoid thundering-herd behaviour.

    Returns:
        A decorator that wraps the target function with retry logic.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    time.sleep(delay)

        return wrapper

    return decorator
