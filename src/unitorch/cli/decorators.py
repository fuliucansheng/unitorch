# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import inspect
import logging
from unitorch.cli import get_global_config

__global_config__ = get_global_config()


def add_default_section_for_init(section, default_params=dict()):
    def default_init_func(cls, config, **kwargs):
        sign = inspect.signature(cls)
        params = sign.parameters
        for k, v in params.items():
            dvalue = v.default
            if kwargs.get(k) is not None:
                continue
            kwargs[k] = config.getdefault(section, k, dvalue)
        for k, v in default_params.items():
            kwargs[k] = config.getdefault(section, k, v)
        obj = cls(**kwargs)
        setattr(obj, "__unitorch_setting__", config)
        return obj

    def add_func(init_func):
        def _init_func(cls, config, **kwargs):
            ret = init_func(cls, config, **kwargs)
            if isinstance(ret, cls):
                setattr(ret, "__unitorch_setting__", config)
                return ret
            assert isinstance(ret, dict) or ret is None
            if ret is not None:
                kwargs.update(ret)
            ret = default_init_func(cls, config, **kwargs)
            return ret

        return _init_func

    return add_func


def add_default_section_for_function(
    section,
    default_params=dict(),
):
    def get_func_params(
        func,
        config,
        args,
        kwargs,
    ):
        sign = inspect.signature(func)
        params = sign.parameters
        for k, v in default_params.items():
            if k in kwargs:
                continue
            kwargs[k] = config.getdefault(section, k, v)

        for i, (k, v) in enumerate(params.items()):
            if k == "self" or k in kwargs:
                continue
            dvalue = args[i] if i < len(args) else v.default
            kwargs[k] = config.getdefault(section, k, dvalue)
        return kwargs

    def add_func(func):
        def _new_func(*args, **kwargs):
            if len(args) > 0 and hasattr(args[0], "__unitorch_setting__"):
                kwargs = get_func_params(
                    func,
                    args[0].__unitorch_setting__,
                    args,
                    kwargs,
                )
                ret = func(args[0], **kwargs)
            else:
                kwargs = get_func_params(
                    func,
                    __global_config__,
                    args,
                    kwargs,
                )
                ret = func(args[0], **kwargs)
            return ret

        return _new_func

    return add_func


# import gradio as gr
# def save_and_load_latest_state(init_func):
#     dtypes = [
#         gr.Dropdown,
#         gr.Slider,
#         gr.Checkbox,
#         gr.Radio,
#         gr.Textbox,
#         gr.Image,
#         gr.Gallery,
#         gr.Audio,
#         gr.Video,
#         gr.File,
#     ]
#     states = {}

#     def save_state(v, i):
#         states[i] = v

#     def load_state(i):
#         return states.get(i, None)

#     def actual_func(*args, **kwargs):
#         inst = init_func(*args, **kwargs)
#         iface = inst.iface
#         ignore_elements = inst.ignore_elements
#         ignore_indexes = [element._id for element in ignore_elements]
#         with iface:
#             indexes = []

#             for index, block in iface.blocks.items():
#                 if (
#                     any([isinstance(block, dtype) for dtype in dtypes])
#                     and index not in ignore_indexes
#                 ):
#                     indexes.append(index)

#             for index in indexes:
#                 block = iface.blocks[index]
#                 block.change(
#                     fn=lambda x: save_state(x, f"blockid-{index}"), inputs=[block]
#                 )
#                 iface.load(fn=lambda: load_state(f"blockid-{index}"), outputs=[block])

#     return actual_func
