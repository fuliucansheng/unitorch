# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
from unitorch.cli import GenericCopilotTool
from unitorch.cli import (
    register_copilot_tool,
    registered_process,
    registered_copilot_tool,
    registered_model,
    registered_fastapi,
    registered_service,
    registered_script,
    registered_score,
    registered_dataset,
    registered_loss,
    registered_optim,
    registered_scheduler,
    registered_task,
    registered_writer,
)


@register_copilot_tool("core/copilot/pkg_infos")
class PkgInfosCopilotTool(GenericCopilotTool):
    def launch(self, name=None,**kwargs):
        pkg_infos = {
            "process": registered_process,
            "copilot_tool": registered_copilot_tool,
            "model": registered_model,
            "fastapi": registered_fastapi,
            "service": registered_service,
            "script": registered_script,
            "score": registered_score,
            "dataset": registered_dataset,
            "loss": registered_loss,
            "optimizer": registered_optim,
            "scheduler": registered_scheduler,
            "task": registered_task,
            "writer": registered_writer,
        }
        if name is not None:
            if name in pkg_infos:
                logging.info(f"Registered {name}s: {list(pkg_infos[name].keys())}")
            else:
                logging.warning(f"Package type '{name}' not found. Available types: {list(pkg_infos.keys())}")
        else:
            for pkg_type, pkg_dict in pkg_infos.items():
                logging.info(f"Registered {pkg_type}s: {list(pkg_dict.keys())}")

    def describe(self):
        return "Provides information about registered packages in the unitorch package, including process, copilot_tool, model, fastapi, service, script, score, dataset, loss, optimizer, scheduler, task, and writer."

    def usage(self):
        return "unitorch-copilot-cli core/copilot/pkg_infos [--name <package_type>]"