# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, List

from unitorch.cli import (
    GenericCopilotRemoteSpec,
    GenericCopilotTool,
    register_copilot_tool,
    registered_copilot_tool,
    serialize_copilot_output,
)


def list_copilot_tools() -> List[str]:
    return sorted(registered_copilot_tool.keys())


def get_copilot_tool(name: str) -> GenericCopilotTool:
    if name not in registered_copilot_tool:
        raise KeyError(f"Copilot tool {name!r} is not registered.")
    return registered_copilot_tool[name]


def copilot_tool_metadata(name: str) -> Dict:
    copilot_tool = get_copilot_tool(name)
    remote = None
    if copilot_tool.remote is not None:
        remote = {
            "route": copilot_tool.remote.route,
            "method": copilot_tool.remote.method,
            "req_type": copilot_tool.remote.req_type,
            "resp_type": copilot_tool.remote.resp_type,
            "param_fields": copilot_tool.remote.param_fields,
            "file_fields": copilot_tool.remote.file_fields,
            "image_fields": copilot_tool.remote.image_fields,
            "video_fields": copilot_tool.remote.video_fields,
            "binary": copilot_tool.remote.binary,
        }
    return {
        "name": copilot_tool.name,
        "description": copilot_tool.description,
        "tags": list(copilot_tool.tags),
        "python": {
            "module": copilot_tool.python_module,
            "function": copilot_tool.python_function,
        },
        "cli": {
            "command": f"unitorch-copilot-cli {copilot_tool.name}",
        },
        "remote": remote,
    }


from unitorch.cli.copilots.client import CopilotClient, call_fastapi
from unitorch.cli.copilots.skills import (
    export_copilot_skill_documents,
    install_copilot_skill_documents,
    render_copilot_skill_markdown,
    uninstall_copilot_skill_documents,
    validate_copilot_skill_documents,
)

from unitorch.cli.copilots.pkg_infos import pkg_infos
