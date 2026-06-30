# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import ast
import json
from typing import Any, Dict

import fire
import unitorch.cli
import unitorch.cli.fastapis
from unitorch.cli.copilots import (
    copilot_tool_metadata,
    get_copilot_tool,
    list_copilot_tools,
    serialize_copilot_output,
)


def _print_json(value: Any) -> None:
    print(json.dumps(serialize_copilot_output(value), ensure_ascii=False, indent=2))


def _literal(value: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _coerce_kwargs(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    tool = get_copilot_tool(name)
    signature = tool.signature
    coerced = {}
    for key, value in kwargs.items():
        if key not in signature.parameters:
            coerced[key] = _literal(value)
            continue
        parameter = signature.parameters[key]
        annotation = parameter.annotation
        value = _literal(value)
        if annotation is int and value is not None:
            value = int(value)
        elif annotation is float and value is not None:
            value = float(value)
        elif annotation is bool and value is not None:
            value = bool(value)
        coerced[key] = value
    return coerced


@fire.decorators.SetParseFn(str)
def copilot(**kwargs):
    pass


@fire.decorators.SetParseFn(str)
def cli(tool: str = None, **kwargs):
    if tool is None:
        _print_json(list_copilot_tools())
        return
    copilot_tool = get_copilot_tool(tool)
    _print_json(copilot_tool.invoke(**_coerce_kwargs(tool, kwargs)))


def main():
    fire.Fire(copilot)


def cli_main():
    fire.Fire(cli)
