# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from __future__ import annotations

import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import fire

from unitorch.cli.copilots import (
    copilot_tool_metadata,
    get_copilot_tool,
    list_copilot_tools,
)


_COPILOT_TOOLS_SKILL_NAME = "unitorch-copilot-tools"
_COPILOT_TOOLS_DESCRIPTION = (
    "Use when an agent needs UniTorch-provided tools for model, algorithm, "
    "package info, and registered component workflows."
)
_COPILOT_TOOLS_OVERVIEW = (
    "`unitorch-copilot-tools` is a collection of UniTorch-provided tools for "
    "model and algorithm related workflows. It also includes UniTorch "
    "introspection helpers such as package info and registered component "
    "metadata. Each child directory documents one registered copilot tool, "
    "including CLI usage through `unitorch-copilot-cli`, Python invocation, "
    "parameters, and any remote FastAPI adapter."
)
_COPILOT_TOOLS_SUBSKILL_NOTE = (
    "This is a subskill of `unitorch-copilot-tools`. Use the parent skill "
    "index to discover other UniTorch model, algorithm, and package info "
    "tools."
)
_SKIP_PARAMETERS = {"pipeline", "kwargs"}


def _slug(name: str, default: str = "copilot-tool") -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or default


def _copilot_tool_dir_name(name: str) -> str:
    return _slug(name)


def _copilot_tool_skill_name(name: str) -> str:
    return f"{_COPILOT_TOOLS_SKILL_NAME}-{_copilot_tool_dir_name(name)}"


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _type_name(tp: Any) -> str:
    if tp is inspect._empty:
        return "Any"
    if getattr(tp, "__module__", "") == "typing":
        return str(tp).replace("typing.", "")
    if getattr(tp, "__name__", None):
        return tp.__name__
    return str(tp).replace("typing.", "")


def _default_value(value: Any):
    if value is inspect._empty:
        return None
    return value


def _parameters(copilot_tool) -> Dict[str, Dict[str, Any]]:
    signature = copilot_tool.signature
    hints = copilot_tool.type_hints
    parameters = {}
    for name, parameter in signature.parameters.items():
        if name in _SKIP_PARAMETERS:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        hint = hints.get(name, parameter.annotation)
        parameters[name] = {
            "type": _type_name(hint),
            "required": parameter.default is inspect._empty,
            "default": _default_value(parameter.default),
        }
    return parameters


def _format_default(value: Any) -> str:
    if value is None:
        return ""
    return f"`{value!r}`"


def _parameters_markdown(parameters: Dict[str, Dict[str, Any]]) -> str:
    if not parameters:
        return "This copilot tool does not define explicit parameters."

    lines = [
        "| Name | Type | Required | Default |",
        "|------|------|----------|---------|",
    ]
    for name, info in parameters.items():
        required = "yes" if info["required"] else "no"
        default = _format_default(info["default"])
        lines.append(
            f"| `{name}` | `{info['type']}` | {required} | {default} |"
        )
    return "\n".join(lines)


def _remote_markdown(remote: Optional[Dict[str, Any]]) -> str:
    if remote is None:
        return "This copilot tool does not declare a remote FastAPI adapter."

    lines = [
        "This copilot tool can call a service started by `unitorch-fastapi`.",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Route | `{remote['route']}` |",
        f"| Method | `{remote['method']}` |",
        f"| Response | `{remote['resp_type']}` |",
    ]
    if remote.get("param_fields"):
        lines.append(f"| Param fields | `{remote['param_fields']}` |")
    if remote.get("image_fields"):
        lines.append(f"| Image fields | `{remote['image_fields']}` |")
    if remote.get("video_fields"):
        lines.append(f"| Video fields | `{remote['video_fields']}` |")
    return "\n".join(lines)


def render_copilot_skill_markdown(name: str) -> str:
    copilot_tool = get_copilot_tool(name)
    metadata = copilot_tool_metadata(name)
    parameters = _parameters(copilot_tool)
    description = copilot_tool.description or f"Invoke the {name} copilot tool."
    skill_name = _copilot_tool_skill_name(name)

    lines = [
        "---",
        f"name: {_yaml_string(skill_name)}",
        "description: "
        + _yaml_string(
            f"Use when an agent needs to invoke the `{name}` unitorch copilot tool."
        ),
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        _COPILOT_TOOLS_SUBSKILL_NOTE,
        "",
        "## When To Use",
        "",
        f"Use this skill when you need to call `{name}` through unitorch copilot.",
        "",
        "## CLI",
        "",
        "```bash",
        metadata["cli"]["command"],
        "```",
        "",
        "## Python",
        "",
        "```python",
        "from unitorch.cli.copilots import get_copilot_tool",
        "",
        f'tool = get_copilot_tool("{name}")',
        "result = tool.invoke()",
        "```",
        "",
        "## Parameters",
        "",
        _parameters_markdown(parameters),
        "",
        "## Remote FastAPI",
        "",
        _remote_markdown(metadata["remote"]),
        "",
    ]

    if metadata["remote"] is not None:
        lines.extend(
            [
                "```python",
                "from unitorch.cli.copilots import CopilotClient",
                "",
                'client = CopilotClient(endpoint="http://127.0.0.1:5000")',
                f'result = client.invoke("{name}")',
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def render_copilot_skill_index_markdown(name: Optional[str] = None) -> str:
    return _render_copilot_skill_index_markdown(_copilot_tool_names(name))


def _render_copilot_skill_index_markdown(names: list[str]) -> str:
    lines = [
        "---",
        f"name: {_yaml_string(_COPILOT_TOOLS_SKILL_NAME)}",
        "description: " + _yaml_string(_COPILOT_TOOLS_DESCRIPTION),
        "---",
        "",
        f"# {_COPILOT_TOOLS_SKILL_NAME}",
        "",
        _COPILOT_TOOLS_OVERVIEW,
        "",
        "## Usage",
        "",
        "```bash",
        "unitorch-copilot-cli",
        "unitorch-copilot-cli <tool-name> --arg=value",
        "```",
        "",
        "## Registered Tools",
        "",
    ]

    if not names:
        lines.append("No copilot tools are currently registered.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Tool | Skill | Description |",
            "|------|-------|-------------|",
        ]
    )
    for tool_name in names:
        metadata = copilot_tool_metadata(tool_name)
        description = metadata["description"] or f"Invoke the `{tool_name}` copilot tool."
        skill_path = f"{_copilot_tool_dir_name(tool_name)}/SKILL.md"
        skill_link = f"[{_copilot_tool_dir_name(tool_name)}]({skill_path})"
        lines.append(
            f"| `{tool_name}` | {skill_link} | {description} |"
        )
    lines.append("")
    return "\n".join(lines)


def export_copilot_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
) -> Dict[str, str]:
    folder_path = Path(folder)
    names = _copilot_tool_names(name)
    index_names = _indexed_copilot_tool_names(
        folder_path,
        names,
        include_all=_include_all(name),
    )
    generated = {
        _COPILOT_TOOLS_SKILL_NAME: str(_copilot_skill_document_path(folder_path)),
    }

    skill_dir = _copilot_skill_dir(folder_path)
    skill_dir.mkdir(parents=True, exist_ok=True)
    _copilot_skill_document_path(folder_path).write_text(
        _render_copilot_skill_index_markdown(index_names),
        encoding="utf-8",
    )

    for copilot_tool_name in names:
        skill_path = _copilot_tool_skill_document_path(folder_path, copilot_tool_name)
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            render_copilot_skill_markdown(copilot_tool_name),
            encoding="utf-8",
        )
        generated[copilot_tool_name] = str(skill_path)

    return generated


def install_copilot_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
    force: bool = False,
) -> Dict[str, str]:
    folder_path = Path(folder)
    skill_dir = _copilot_skill_dir(folder_path)
    if force and skill_dir.exists():
        shutil.rmtree(skill_dir)
    return export_copilot_skill_documents(name=name, folder=folder)


def uninstall_copilot_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
) -> Dict[str, str]:
    folder_path = Path(folder)
    skill_dir = _copilot_skill_dir(folder_path)
    removed = {}

    if _include_all(name):
        skill_path = _copilot_skill_document_path(folder_path)
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            removed[_COPILOT_TOOLS_SKILL_NAME] = str(skill_path)
        return removed

    for copilot_tool_name in _copilot_tool_names(name):
        skill_path = _copilot_tool_skill_document_path(folder_path, copilot_tool_name)
        if not skill_path.exists():
            continue
        shutil.rmtree(skill_path.parent)
        removed[copilot_tool_name] = str(skill_path)

    if not removed:
        return removed

    remaining_names = _installed_copilot_tool_names(folder_path)
    if remaining_names:
        _copilot_skill_document_path(folder_path).write_text(
            _render_copilot_skill_index_markdown(remaining_names),
            encoding="utf-8",
        )
    elif skill_dir.exists():
        shutil.rmtree(skill_dir)
        removed[_COPILOT_TOOLS_SKILL_NAME] = str(
            _copilot_skill_document_path(folder_path)
        )

    return removed


def _include_all(name: Optional[str]) -> bool:
    return name is None or name == "all"


def _copilot_tool_names(name: Optional[str]) -> list[str]:
    if _include_all(name):
        return list_copilot_tools()
    return [name]


def _installed_copilot_tool_names(folder: Path) -> list[str]:
    skill_dir = _copilot_skill_dir(folder)
    if not skill_dir.exists():
        return []

    names = []
    for copilot_tool_name in list_copilot_tools():
        if _copilot_tool_skill_document_path(folder, copilot_tool_name).is_file():
            names.append(copilot_tool_name)
    return names


def _indexed_copilot_tool_names(
    folder: Path,
    names: list[str],
    include_all: bool,
) -> list[str]:
    if include_all:
        return names

    indexed = set(_installed_copilot_tool_names(folder))
    indexed.update(names)
    ordered = [name for name in list_copilot_tools() if name in indexed]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _copilot_skill_dir(folder: Path) -> Path:
    return folder / _COPILOT_TOOLS_SKILL_NAME


def _copilot_skill_document_path(folder: Path) -> Path:
    return _copilot_skill_dir(folder) / "SKILL.md"


def _copilot_tool_skill_document_path(folder: Path, name: str) -> Path:
    return _copilot_skill_dir(folder) / _copilot_tool_dir_name(name) / "SKILL.md"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Unsupported boolean value: {value!r}.")
    return bool(value)


@fire.decorators.SetParseFn(str)
def main(
    command: Optional[str] = None,
    name: Optional[str] = None,
    folder: Optional[str] = None,
    force: bool = False,
) -> None:
    if command == "install":
        outputs = install_copilot_skill_documents(
            name=name,
            folder=folder or "./skills",
            force=_parse_bool(force),
        )
    elif command == "uninstall":
        outputs = uninstall_copilot_skill_documents(
            name=name,
            folder=folder or "./skills",
        )
    else:
        raise ValueError(
            "Unsupported skills command. Use `install` or `uninstall`."
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
