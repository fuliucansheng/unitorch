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

from unitorch import VERSION
from unitorch.cli.copilots import (
    copilot_tool_metadata,
    get_copilot_tool,
    list_copilot_tools,
)


_COPILOT_TOOLS_SKILL_NAME = "unitorch-copilot-tools"
_SKILL_AUTHOR = "FULIUCANSHENG"
_SKILL_LICENSE = "MIT"
_COPILOT_TOOLS_DESCRIPTION = (
    "Use when an agent needs to discover or invoke UniTorch copilot tools, "
    "inspect registered models, processors, tasks, FastAPI services, and "
    "writers, or automate ML workflows through unitorch-copilot-cli."
)
_COPILOT_TOOLS_OVERVIEW = (
    "`unitorch-copilot-tools` is the generated skill index for UniTorch "
    "copilot tools. Use it to discover registered components, invoke tools "
    "through `unitorch-copilot-cli`, call the same tools from Python, and "
    "bridge to remote services exposed by `unitorch-fastapi` when a tool "
    "declares a FastAPI adapter. It also preserves access to model and "
    "algorithm related workflows, including package info discovery for "
    "registered UniTorch components."
)
_COPILOT_TOOLS_SUBSKILL_NOTE = (
    "This is a subskill of `unitorch-copilot-tools`. Use the parent skill "
    "index to discover other UniTorch model, algorithm, and package info "
    "tools."
)
_COPILOT_TOOLS_TAGS = (
    "unitorch",
    "copilot",
    "cli",
    "skills",
    "ml",
    "clawhub",
    "hermeshub",
)
_COPILOT_TOOLS_RELATED_SKILLS = (
    "unitorch-config-ini",
    "unitorch-train-model",
    "unitorch-infer-model",
    "unitorch-serve-fastapi",
)
_SERVE_FASTAPI_SKILL_NAME = "unitorch-serve-fastapi"
_SKIP_PARAMETERS = {"pipeline", "kwargs"}
_FRONTMATTER_BASE_REQUIRED_KEYS = (
    "name",
    "description",
)
_FRONTMATTER_GENERATED_REQUIRED_KEYS = (
    "version",
    "author",
    "license",
)
_MAX_DESCRIPTION_LENGTH = 1024


def _slug(name: str, default: str = "copilot-tool") -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or default


def _tag(name: str) -> str:
    return _slug(name.replace("_", "-"), default="unitorch")


def _dedupe(values) -> list[str]:
    seen = set()
    results = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def _copilot_tool_dir_name(name: str) -> str:
    return _slug(name)


def _copilot_tool_skill_name(name: str) -> str:
    return f"{_COPILOT_TOOLS_SKILL_NAME}-{_copilot_tool_dir_name(name)}"


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(values) -> str:
    return json.dumps(_dedupe(values), ensure_ascii=False)


def _skill_description(description: str) -> str:
    description = re.sub(r"\s+", " ", description).strip()
    if len(description) <= _MAX_DESCRIPTION_LENGTH:
        return description
    return description[: _MAX_DESCRIPTION_LENGTH - 3].rstrip() + "..."


def _frontmatter(
    name: str,
    description: str,
    tags,
    related_skills=(),
) -> str:
    related_skills = _dedupe(related_skills)
    return "\n".join(
        [
            "---",
            f"name: {_yaml_string(name)}",
            f"description: {_yaml_string(_skill_description(description))}",
            f"version: {_yaml_string(VERSION)}",
            f"author: {_yaml_string(_SKILL_AUTHOR)}",
            f"license: {_yaml_string(_SKILL_LICENSE)}",
            "metadata:",
            "  hermes:",
            f"    tags: {_yaml_list(tags)}",
            f"    related_skills: {_yaml_list(related_skills)}",
            "  clawhub:",
            f"    tags: {_yaml_list(tags)}",
            f"related_skills: {_yaml_list(related_skills)}",
            "---",
        ]
    )


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


def _parameter_placeholder(name: str, info: Dict[str, Any]) -> str:
    if info["default"] is not None:
        return repr(info["default"])
    if name == "name":
        return '"model"'
    type_name = info["type"].lower()
    if "bool" in type_name:
        return "true"
    if "int" in type_name:
        return "1"
    if "float" in type_name:
        return "1.0"
    if "list" in type_name or "tuple" in type_name:
        return "[]"
    if "dict" in type_name:
        return "{}"
    if "path" in name:
        return '"path/to/input"'
    return '"value"'


def _cli_command(name: str, parameters: Dict[str, Dict[str, Any]]) -> str:
    command = f"unitorch-copilot-cli {name}"
    required_parameters = {
        key: value for key, value in parameters.items() if value["required"]
    }
    if not required_parameters and parameters:
        key, value = next(iter(parameters.items()))
        return f"{command} --{key} {_parameter_placeholder(key, value)}"
    if required_parameters:
        args = [
            f"--{key} {_parameter_placeholder(key, value)}"
            for key, value in required_parameters.items()
        ]
        return f"{command} {' '.join(args)}"
    return command


def _python_invocation(
    name: str,
    parameters: Dict[str, Dict[str, Any]],
    indent: str = "",
) -> list[str]:
    required_parameters = {
        key: value for key, value in parameters.items() if value["required"]
    }
    if not required_parameters:
        return [f'{indent}result = tool.invoke()']

    lines = [f"{indent}result = tool.invoke("]
    for key, value in required_parameters.items():
        lines.append(f"{indent}    {key}={_parameter_placeholder(key, value)},")
    lines.append(f"{indent})")
    return lines


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


def _tool_description(name: str, description: str) -> str:
    if description:
        return (
            f"Use when an agent needs to invoke the {name} UniTorch copilot "
            f"tool: {description}"
        )
    return f"Use when an agent needs to invoke the {name} UniTorch copilot tool."


def _tool_tags(name: str, metadata: Dict[str, Any]) -> list[str]:
    name_tags = [_tag(part) for part in re.split(r"[/._-]+", name) if part]
    tags = ["unitorch", "copilot", "cli", "tool", *name_tags]
    tags.extend(_tag(tag) for tag in metadata.get("tags", ()))
    if metadata.get("remote") is not None:
        tags.append("fastapi")
    return _dedupe(tags)


def _tool_related_skills(metadata: Dict[str, Any]) -> list[str]:
    related = [_COPILOT_TOOLS_SKILL_NAME]
    if metadata.get("remote") is not None:
        related.append(_SERVE_FASTAPI_SKILL_NAME)
    return related


def _tool_when_to_use(name: str, has_remote: bool) -> list[str]:
    lines = [
        f"- Invoke `{name}` from an agent workflow.",
        "- Inspect the tool's parameter contract before calling it.",
    ]
    if has_remote:
        lines.append(
            "- Prefer the remote adapter when the tool wraps a `unitorch-fastapi` service."
        )
    else:
        lines.append(
            "- Use local CLI or Python invocation because this tool has no remote adapter."
        )
    return lines


def _tool_verification_checklist(has_remote: bool) -> list[str]:
    lines = [
        "- Confirm `unitorch-copilot-cli` is available in the active Python environment.",
        "- Run the CLI example with the smallest useful inputs before wiring it into a larger agent workflow.",
    ]
    if has_remote:
        lines.append(
            "- Confirm `unitorch-fastapi` is healthy and the route matches the generated metadata."
        )
    return lines


def _tool_common_pitfalls(has_remote: bool) -> list[str]:
    lines = [
        "- Do not invent parameter names; use the table above or `tool.signature`.",
        "- Import extension libraries before discovery when they register additional copilot tools.",
    ]
    if has_remote:
        lines.append(
            "- Keep local and remote invocation payloads aligned when media fields are involved."
        )
    return lines


def render_copilot_skill_markdown(name: str) -> str:
    copilot_tool = get_copilot_tool(name)
    metadata = copilot_tool_metadata(name)
    parameters = _parameters(copilot_tool)
    description = copilot_tool.description or f"Invoke the {name} copilot tool."
    skill_name = _copilot_tool_skill_name(name)
    cli_command = _cli_command(name, parameters)
    tool_description = _tool_description(name, description)
    has_remote = metadata["remote"] is not None

    lines = [
        _frontmatter(
            name=skill_name,
            description=tool_description,
            tags=_tool_tags(name, metadata),
            related_skills=_tool_related_skills(metadata),
        ),
        "",
        f"# {name}",
        "",
        "## Overview",
        "",
        description,
        "",
        _COPILOT_TOOLS_SUBSKILL_NOTE,
        "",
        "## When To Use",
        "",
        *_tool_when_to_use(name, has_remote),
        "",
        "## CLI",
        "",
        "Run the tool through `unitorch-copilot-cli`:",
        "",
        "```bash",
        cli_command,
        "```",
        "",
        "## Python",
        "",
        "For tools without required parameters, the minimal call is `result = tool.invoke()`.",
        "",
        "```python",
        "from unitorch.cli.copilots import get_copilot_tool",
        "",
        f'tool = get_copilot_tool("{name}")',
        *_python_invocation(name, parameters),
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
                "Use `CopilotClient` when the corresponding FastAPI server is already running:",
                "",
                "```python",
                "from unitorch.cli.copilots import CopilotClient",
                "",
                'client = CopilotClient(endpoint="http://127.0.0.1:5000")',
                f'result = client.invoke("{name}")',
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Verification Checklist",
            "",
            *_tool_verification_checklist(has_remote),
            "",
            "## Common Pitfalls",
            "",
            *_tool_common_pitfalls(has_remote),
            "",
        ]
    )

    return "\n".join(lines)


def render_copilot_skill_index_markdown(name: Optional[str] = None) -> str:
    return _render_copilot_skill_index_markdown(_copilot_tool_names(name))


def _render_copilot_skill_index_markdown(names: list[str]) -> str:
    lines = [
        _frontmatter(
            name=_COPILOT_TOOLS_SKILL_NAME,
            description=_COPILOT_TOOLS_DESCRIPTION,
            tags=_COPILOT_TOOLS_TAGS,
            related_skills=_COPILOT_TOOLS_RELATED_SKILLS,
        ),
        "",
        f"# {_COPILOT_TOOLS_SKILL_NAME}",
        "",
        "## Overview",
        "",
        _COPILOT_TOOLS_OVERVIEW,
        "",
        "## Install",
        "",
        "Generate the canonical project skills under the root `skills/` directory:",
        "",
        "Install UniTorch first; for local development, use an editable install "
        "such as `python3 -m pip install -e .` and run without `PYTHONPATH=src`:",
        "",
        "```bash",
        "npm run generate-skills",
        "```",
        "",
        "Install the published root skills into an agent skill folder with the "
        "external `skills` npm package:",
        "",
        "```bash",
        "npx skills add fuliucansheng/unitorch",
        "npx skills add fuliucansheng/unitorch --folder ./agent-skills",
        "```",
        "",
        "That external installer copies the root `skills/` package and can report:",
        "",
        "```json",
        "{",
        '  "repo": "fuliucansheng/unitorch",',
        '  "folder": "/home/decu/.hermes/skills",',
        '  "copied": [',
        '    "unitorch-config-ini",',
        '    "unitorch-copilot-tools",',
        '    "unitorch-infer-model",',
        '    "unitorch-replace-decorator",',
        '    "unitorch-serve-fastapi",',
        '    "unitorch-train-model"',
        "  ]",
        "}",
        "```",
        "",
        "Generate, export, and validate from Python when updating this repository:",
        "",
        "```bash",
        "python3 -m unitorch.cli.copilots.skills install all --folder ./skills --force true",
        "python3 -m unitorch.cli.copilots.skills validate --folder ./skills",
        "```",
        "",
        "## When To Use",
        "",
        "- Discover registered UniTorch models, processors, tasks, writers, and services.",
        "- Invoke small agent-facing utilities through `unitorch-copilot-cli`.",
        "- Install the packaged root `skills/` tree into an agent-local skill folder with the external `skills` npm package.",
        "",
        "## CLI",
        "",
        "```bash",
        "unitorch-copilot-cli",
        "unitorch-copilot-cli <tool-name> --arg=value",
        "```",
        "",
        "## Python",
        "",
        "```python",
        "from unitorch.cli.copilots import get_copilot_tool",
        "",
        "tool = get_copilot_tool(\"core/copilot/pkg_infos\")",
        "result = tool.invoke(name=\"model\")",
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
    lines.extend(
        [
            "## Verification Checklist",
            "",
            "- Run `python3 -m unitorch.cli.copilots.skills validate --folder ./skills` after generation.",
            "- Confirm the parent index lists every generated child skill.",
            "- Confirm the root `skills/` package contains `skills/unitorch-copilot-tools/SKILL.md` and child `SKILL.md` files before installing it elsewhere.",
            "",
            "## Common Pitfalls",
            "",
            "- Generate into `skills/` so the generated copilot skill sits beside the hand-written root skills.",
            "- Use `npx skills add fuliucansheng/unitorch` only through the external open-agent skills ecosystem; this repository does not publish or alias that installer.",
            "- The external installer copies published skills into an agent-local folder; it does not regenerate this repository's skill markdown.",
            "- Install UniTorch and any extension packages before generation; do not rely on `PYTHONPATH=src` as the normal path.",
            "- Keep the checked-in root `skills/` tree in sync when copilot tool metadata or installation guidance changes.",
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


def validate_copilot_skill_documents(folder: str = "./skills") -> Dict[str, Any]:
    folder_path = Path(folder)
    skill_paths = sorted(folder_path.rglob("SKILL.md"))
    if not skill_paths:
        raise ValueError(f"No SKILL.md files found under {folder_path}.")

    errors = []
    validated = []
    for skill_path in skill_paths:
        try:
            markdown = skill_path.read_text(encoding="utf-8")
            frontmatter = _extract_frontmatter(markdown)
            metadata = _load_frontmatter(frontmatter)
            _validate_frontmatter(skill_path, metadata, markdown)
            validated.append(str(skill_path))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        joined = "\n- ".join(errors)
        raise ValueError(f"Skill validation failed:\n- {joined}")

    return {
        "valid": True,
        "count": len(validated),
        "skills": validated,
    }


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


def _extract_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md does not start with YAML frontmatter.")
    end = text.find("\n---", 4)
    if end == -1:
        raise ValueError("SKILL.md frontmatter is not closed.")
    return text[4:end].strip()


def _load_frontmatter(frontmatter: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return _load_frontmatter_without_yaml(frontmatter)

    loaded = yaml.safe_load(frontmatter)
    if not isinstance(loaded, dict):
        raise ValueError("SKILL.md frontmatter must parse as a mapping.")
    return loaded


def _load_frontmatter_without_yaml(frontmatter: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in (*_FRONTMATTER_BASE_REQUIRED_KEYS, *_FRONTMATTER_GENERATED_REQUIRED_KEYS):
        match = re.search(rf"^{key}:\s*(.+)$", frontmatter, re.MULTILINE)
        if match is None:
            continue
        metadata[key] = _parse_frontmatter_value(match.group(1).strip())

    related = re.search(r"^related_skills:\s*(.+)$", frontmatter, re.MULTILINE)
    if related is not None:
        metadata["related_skills"] = _parse_frontmatter_value(related.group(1).strip())

    hermes_block = re.search(
        r"^\s{2}hermes:\s*\n((?:^\s{4}.+\n?)*)",
        frontmatter,
        re.MULTILINE,
    )
    if hermes_block is not None:
        hermes: Dict[str, Any] = {}
        for key in ("tags", "related_skills"):
            match = re.search(
                rf"^\s{{4}}{key}:\s*(.+)$",
                hermes_block.group(1),
                re.MULTILINE,
            )
            if match is not None:
                hermes[key] = _parse_frontmatter_value(match.group(1).strip())
        if hermes:
            metadata["metadata"] = {"hermes": hermes}

    return metadata


def _parse_frontmatter_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("'\"")


def _validate_frontmatter(
    skill_path: Path,
    metadata: Dict[str, Any],
    markdown: str,
) -> None:
    required_keys = list(_FRONTMATTER_BASE_REQUIRED_KEYS)
    strict_generated_metadata = _is_generated_copilot_skill_path(skill_path)
    if strict_generated_metadata:
        required_keys.extend(_FRONTMATTER_GENERATED_REQUIRED_KEYS)

    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise ValueError(f"{skill_path}: missing frontmatter keys {missing}.")

    description = metadata["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{skill_path}: description must be a non-empty string.")
    if len(description) > _MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"{skill_path}: description exceeds {_MAX_DESCRIPTION_LENGTH} chars."
        )

    if not strict_generated_metadata:
        return

    hermes = (metadata.get("metadata") or {}).get("hermes") or {}
    tags = hermes.get("tags")
    if not isinstance(tags, list) or not tags:
        raise ValueError(f"{skill_path}: metadata.hermes.tags must be a non-empty list.")

    related_skills = hermes.get("related_skills", metadata.get("related_skills"))
    if related_skills is None:
        raise ValueError(
            f"{skill_path}: metadata.hermes.related_skills must be present."
        )
    if not isinstance(related_skills, list):
        raise ValueError(
            f"{skill_path}: metadata.hermes.related_skills must be a list."
        )
    if any(not isinstance(skill, str) or not skill.strip() for skill in related_skills):
        raise ValueError(
            f"{skill_path}: metadata.hermes.related_skills must only contain "
            "non-empty strings."
        )
    legacy_related_skills = metadata.get("related_skills")
    if legacy_related_skills is not None and legacy_related_skills != related_skills:
        raise ValueError(
            f"{skill_path}: top-level related_skills must match "
            "metadata.hermes.related_skills when both are present."
        )

    if "## Overview" not in markdown:
        raise ValueError(f"{skill_path}: missing Overview section.")


def _is_generated_copilot_skill_path(skill_path: Path) -> bool:
    return _COPILOT_TOOLS_SKILL_NAME in skill_path.parts


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
    elif command == "export":
        outputs = export_copilot_skill_documents(
            name=name,
            folder=folder or "./skills",
        )
    elif command == "uninstall":
        outputs = uninstall_copilot_skill_documents(
            name=name,
            folder=folder or "./skills",
        )
    elif command == "validate":
        outputs = validate_copilot_skill_documents(
            folder=folder or "./skills",
        )
    else:
        raise ValueError(
            "Unsupported skills command. Use `install`, `export`, `uninstall`, "
            "or `validate`."
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
