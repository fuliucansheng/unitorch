"""Exercise assets-task STOP with real two-rank CPU/Gloo collectives."""
import json
import os
from pathlib import Path

import fire
import pytest
import torch
import torch.distributed as dist

from flux3_reliability_utils import patch_runtime
from unitorch.tasks import flux3


def main(config, output, stop_rank=0, save_every=3, pilot_steps=5, stop_step=2, grad_acc_step=1):
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    events, observations = [], []
    diverged = False
    def trace(event):
        nonlocal diverged
        if diverged:
            return
        gathered = [None, None]
        dist.all_gather_object(gathered, event)
        events.append(event)
        if gathered != [event, event]:
            diverged = True
            raise AssertionError("collective sequence diverged: " + repr(gathered))

    original_destroy = dist.destroy_process_group
    def finish():
        try:
            trace("return")
        finally:
            original_destroy()

    with pytest.MonkeyPatch.context() as patch:
        models = patch_runtime(patch, trace)
        exists = Path.exists
        def stop_exists(path):
            if str(path) == "CPU-FIXTURE-STOP":
                flag = rank == stop_rank and models[0].train_calls >= stop_step
                observations.append(flag)
                return flag
            return exists(path)
        patch.setattr(Path, "exists", stop_exists)
        patch.setattr(dist, "destroy_process_group", finish)
        kwargs = json.loads(Path(config).read_text())
        kwargs.update(save_every=save_every, pilot_steps=pilot_steps, grad_acc_step=grad_acc_step,
                      stop_file="CPU-FIXTURE-STOP")
        result = flux3.train_assets(**kwargs)
        Path(output, f"rank-{rank}.json").write_text(json.dumps(dict(
            result=result, events=events, observations=observations)))


if __name__ == "__main__":
    fire.Fire(main)
