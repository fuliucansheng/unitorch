# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import json
import logging
from unitorch.utils import is_wandb_available, read_json_file
from unitorch.cli.core import CoreConfigureParser

_wandb_available = False

if is_wandb_available():
    import wandb
else:
    wandb = None


def setup(config):
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

    rank = int(os.environ.get("RANK", 0))
    if rank > 0:
        logging.warning("wandb is not available in distributed mode.")
        return

    if any(x is None for x in [team, project, token]):
        return

    try:
        wandb.login(key=token, relogin=True)
        if db_file is not None:
            db_info = read_json_file(db_file)
            run_id = db_info.get("run_id", None)
            if run_id is None:
                run_id = wandb.util.generate_id()
                db_info["run_id"] = run_id
                with open(db_file, "w") as f:
                    f.write(json.dumps(db_info))
        else:
            run_id = None
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
        logging.info("Login to wandb successfully.")
    except Exception as e:
        logging.error(f"Failed to login to wandb: {e}")
        return


def is_available():
    return _wandb_available


config = getattr(wandb, "config", None)


def log(*args, **kwargs):
    if wandb is None:
        return
    if not is_available():
        return
    wandb.log(*args, **kwargs)


def save(*args, **kwargs):
    if wandb is None:
        return
    if not is_available():
        return
    wandb.save(*args, **kwargs)


def finish(*args, **kwargs):
    if wandb is None:
        return
    if not is_available():
        return
    wandb.finish(*args, **kwargs)
