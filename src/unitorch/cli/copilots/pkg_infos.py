# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional

from unitorch.cli import register_copilot_tool, registered_copilot_tool


def _safe_import(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception:
        return


@register_copilot_tool(
    name="core/copilot/pkg_infos",
    description="List registered unitorch packages and copilot tools.",
    tags=("metadata",),
)
def pkg_infos(name: Optional[str] = None):
    import unitorch.cli as cli

    pkg_infos = {
        "process": cli.registered_process,
        "copilot_tool": getattr(cli, "registered_copilot_tool", {}),
        "model": cli.registered_model,
        "fastapi": cli.registered_fastapi,
        "score": cli.registered_score,
        "dataset": cli.registered_dataset,
        "loss": cli.registered_loss,
        "optimizer": cli.registered_optim,
        "scheduler": cli.registered_scheduler,
        "task": cli.registered_task,
        "writer": cli.registered_writer,
    }
    normalized = {
        pkg_type: sorted(pkg_dict.keys()) for pkg_type, pkg_dict in pkg_infos.items()
    }
    if name is not None:
        if name not in normalized:
            raise KeyError(
                f"Package type {name!r} not found. Available types: {sorted(normalized.keys())}"
            )
        return {name: normalized[name]}
    return normalized
