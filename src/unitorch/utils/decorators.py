# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging
import inspect

OPTIMIZED_CLASSES = dict()


# replace decorator from fastseq
def replace(target_obj):
    """A decorator to replace the specified obj.

    `target_obj` can be a class or a function.

    Example:

    ```python
    class A:
        def f(self):
            print('class A')
    @replace(A)
    class B:
        def f(self):
            print('class B')
    ```

    Args:
        target_obj (class/func/method): a class, method, or function to be
                                        replaced.

    Returns:
        A decorator function to replace the input object.
    """

    def decorator(new_obj):
        if target_obj in OPTIMIZED_CLASSES:
            logging.warning("{} has been optimized again.".format(target_obj))
        setattr(new_obj, "__replaced_class__", target_obj)
        OPTIMIZED_CLASSES[target_obj] = new_obj
        for k, v in list(sys.modules.items()):
            if (
                target_obj.__name__ in v.__dict__
                and v.__dict__[target_obj.__name__] is target_obj
            ):
                delattr(sys.modules[k], target_obj.__name__)
                setattr(sys.modules[k], target_obj.__name__, new_obj)
                logging.debug(
                    "In module {}, {} is replaced by {}".format(k, target_obj, new_obj)
                )
            # replace target_obj if it is used as the base classes.
            for key in list(v.__dict__.keys()):
                if (
                    inspect.isclass(v.__dict__[key])
                    and v.__dict__[key] != new_obj
                    and target_obj in v.__dict__[key].__bases__
                ):
                    idx = v.__dict__[key].__bases__.index(target_obj)
                    bases = list(v.__dict__[key].__bases__)
                    bases[idx] = new_obj
                    v.__dict__[key].__bases__ = tuple(bases)
                    logging.debug(
                        "In module {}, the base class of {} is replaced by {}".format(
                            k, v.__dict__[key], new_obj
                        )
                    )
        return new_obj

    return decorator
