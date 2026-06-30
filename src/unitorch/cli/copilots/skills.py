# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from __future__ import annotations

import inspect
import json
import pkgutil
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


_SKIP_PARAMETERS = {"pipeline", "kwargs"}


def _skill_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower()
    normalized = normalized or "copilot-tool"
    if normalized.startswith("unitorch-"):
        return normalized
    return f"unitorch-{normalized}"


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
    skill_name = _skill_name(name)

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


def export_copilot_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
) -> Dict[str, str]:
    folder_path = Path(folder)
    names = _copilot_tool_names(name)
    generated = {}

    for copilot_tool_name in names:
        skill_path = _skill_document_path(folder_path, copilot_tool_name)
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
) -> Dict[str, str]:
    installed = {}
    if _include_all(name):
        installed.update(_install_manual_skill_documents(folder=folder))
        installed.update(export_copilot_skill_documents(name=name, folder=folder))
        return installed

    manual_name = _manual_skill_name(name)
    if manual_name is not None:
        return _install_manual_skill_documents(name=manual_name, folder=folder)
    return export_copilot_skill_documents(name=name, folder=folder)


def uninstall_copilot_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
) -> Dict[str, str]:
    folder_path = Path(folder)
    removed = {}

    for manual_name in _manual_skill_names(name):
        skill_path = _skill_document_path(folder_path, manual_name)
        if not skill_path.exists():
            continue
        shutil.rmtree(skill_path.parent)
        removed[manual_name] = str(skill_path)

    for copilot_tool_name in _copilot_tool_names(name):
        skill_path = _skill_document_path(folder_path, copilot_tool_name)
        if not skill_path.exists():
            continue
        skill_path.unlink()
        removed[copilot_tool_name] = str(skill_path)
        try:
            skill_path.parent.rmdir()
        except OSError:
            pass

    return removed


def _include_all(name: Optional[str]) -> bool:
    return name is None or name == "all"


def _copilot_tool_names(name: Optional[str]) -> list[str]:
    if _include_all(name):
        return list_copilot_tools()
    return [name]


def _manual_skill_sources() -> Dict[str, Path]:
    sources = {}
    for skills_dir in _manual_skill_dirs():
        if not skills_dir.exists() or not skills_dir.is_dir():
            continue
        for skill_dir in sorted(skills_dir.iterdir()):
            if (skill_dir / "SKILL.md").is_file():
                sources[skill_dir.name] = skill_dir
    return sources


def _manual_skill_dirs() -> list[Path]:
    import unitorch.cli.skills as skills

    return [Path(path) for path in pkgutil.extend_path(skills.__path__, skills.__name__)]


def _manual_skill_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    sources = _manual_skill_sources()
    if name in sources:
        return name
    for manual_name in sources:
        if name == _skill_name(manual_name):
            return manual_name
    return None


def _manual_skill_names(name: Optional[str]) -> list[str]:
    if _include_all(name):
        return list(_manual_skill_sources().keys())
    manual_name = _manual_skill_name(name)
    return [manual_name] if manual_name is not None else []


def _install_manual_skill_documents(
    name: Optional[str] = None,
    folder: str = "./skills",
) -> Dict[str, str]:
    folder_path = Path(folder)
    installed = {}
    sources = _manual_skill_sources()
    names = _manual_skill_names(name)

    for manual_name in names:
        source_dir = sources[manual_name]
        target_dir = folder_path / _skill_name(manual_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        skill_path = target_dir / "SKILL.md"
        skill_path.write_text(
            _prefix_skill_frontmatter_name(skill_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        installed[manual_name] = str(skill_path)

    return installed


def _prefix_skill_frontmatter_name(content: str) -> str:
    match = re.match(r"(?s)\A---\n(.*?)\n---\n", content)
    if match is None:
        return content

    frontmatter = match.group(1)
    if not re.search(r"(?m)^name:\s*", frontmatter):
        return content

    frontmatter = re.sub(
        r"(?m)^name:\s*(.+?)\s*$",
        lambda m: f"name: {_yaml_string(_skill_name(_strip_yaml_quotes(m.group(1))))}",
        frontmatter,
        count=1,
    )
    return f"---\n{frontmatter}\n---\n{content[match.end():]}"


def _strip_yaml_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _skill_document_path(folder: Path, name: str) -> Path:
    return folder / _skill_name(name) / "SKILL.md"


@fire.decorators.SetParseFn(str)
def main(
    command: Optional[str] = None,
    name: Optional[str] = None,
    folder: Optional[str] = None,
) -> None:
    if command == "install":
        outputs = install_copilot_skill_documents(
            name=name,
            folder=folder or "./skills",
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
