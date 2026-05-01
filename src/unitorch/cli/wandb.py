# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import logging
import os
import time

from unitorch.utils import is_wandb_available, read_json_file
from unitorch.cli.core import CoreConfigureParser

if is_wandb_available():
    import wandb
else:
    wandb = None

_wandb_available: bool = False

# Expose wandb.config at module level (None when wandb is unavailable).
config = getattr(wandb, "config", None)


def setup(config: CoreConfigureParser) -> None:
    """Initialise a wandb run from *config* (section ``core/wandb``)."""
    global _wandb_available

    config.set_default_section("core/wandb")
    team = config.getoption("team", None)
    project = config.getoption("project", None)
    token = config.getoption("token", None)
    name = config.getoption("name", time.strftime("%m/%d/%Y/%H", time.localtime()))
    db_file = config.getoption("db_file", None)

    if wandb is None:
        logging.warning("wandb is not available.")
        return

    if int(os.environ.get("RANK", 0)) > 0:
        logging.warning("wandb is disabled for non-zero ranks in distributed mode.")
        return

    if any(x is None for x in (team, project, token)):
        return

    try:
        wandb.login(key=token, relogin=True)

        run_id = None
        if db_file is not None:
            db_info = read_json_file(db_file)
            run_id = db_info.get("run_id")
            if run_id is None:
                run_id = wandb.util.generate_id()
                db_info["run_id"] = run_id
                with open(db_file, "w") as f:
                    json.dump(db_info, f)

        wandb.init(
            entity=team,
            project=project,
            name=name,
            group=name,
            reinit=True,
            id=run_id,
            resume="allow",
        )
        _wandb_available = True
        logging.info("Logged in to wandb successfully.")
    except Exception as e:
        logging.error("Failed to login to wandb: %s", e)


def is_available() -> bool:
    return _wandb_available


def _guard() -> bool:
    return wandb is not None and _wandb_available


def log(*args, **kwargs) -> None:
    if _guard():
        wandb.log(*args, **kwargs)


def save(*args, **kwargs) -> None:
    if _guard():
        wandb.save(*args, **kwargs)


def finish(*args, **kwargs) -> None:
    if _guard():
        wandb.finish(*args, **kwargs)
