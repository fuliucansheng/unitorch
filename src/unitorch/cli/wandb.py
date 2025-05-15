# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import logging
from unitorch.utils import is_wandb_available
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
        wandb.init(
            entity=team,
            project=project,
            name=name,
            group=name,
            reinit=True,
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
