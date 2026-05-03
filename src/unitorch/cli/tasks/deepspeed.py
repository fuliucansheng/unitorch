# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import json
import logging
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from collections.abc import Iterable
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from typing import List, Optional, Union
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Queue
from unitorch import set_seed
from unitorch.models import ExponentialMovingAverage
from unitorch.utils import (
    get_local_rank,
    nested_dict_value,
    update_nested_dict,
)
from unitorch.utils import (
    DistributedSkipSampler,
    RandomSkipSampler,
    SequentialSkipSampler,
    PostProcess,
    IOProcess,
    GENERATE_FINISHED,
)
from unitorch.cli import (
    cached_path,
    register_task,
    registered_model,
    registered_dataset,
    registered_loss,
    registered_score,
    registered_writer,
    init_registered_module,
    init_registered_process,
    config_defaults_init,
    config_defaults_method,
)
from unitorch.cli.models import LossOutputs
from unitorch.cli.tasks.supervised import (
    DatasetFeature,
    collate_fn,
    infer,
    monitor,
    save_snapshot,
)
import unitorch.cli.wandb as wandb


def _strip_model_prefix(state_dict):
    """Remove the 'model.' prefix added by DeepSpeed wrappers."""
    return {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}


def save_snapshot_zero_3(
    model,
    ckpt_dir,
    iter_dev,
    score_fn,
    monitor_fns,
    save_checkpoint="default",  # checkpoint policy: default/best/latest/every/all
    merge_checkpoint=False,     # merge ZeRO-3 shards into a single fp32 bin file
    exclude_freeze_parameters=True,  # skip frozen parameters when merging
    best_score=-np.inf,         # best validation score seen so far
    info_path=None,             # path to info.json for persisting training state
    local_rank=-1,              # only rank 0 (or -1) writes files
    **kwargs,                   # extra fields forwarded to info.json (e.g. global_epoch)
):
    """Evaluate a ZeRO stage-3 model and save checkpoints according to policy.

    Returns the (possibly updated) *best_score*.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    results = infer(model, iter_dev)
    new_score = score_fn(outputs=results.outputs, targets=results.targets)
    snapshot_time = time.strftime("%Y%m%d_%H%M", time.localtime())

    if local_rank in [-1, 0]:
        monitor(results.outputs, results.targets, monitor_fns)

    if save_checkpoint in ("all", "default", "best") and new_score > best_score:
        best_score = new_score
        model.save_checkpoint(os.path.join(ckpt_dir, "pytorch_model"))
        if merge_checkpoint and local_rank in [-1, 0]:
            _strip_model_prefix(
                get_fp32_state_dict_from_zero_checkpoint(
                    os.path.join(ckpt_dir, "pytorch_model"),
                    exclude_frozen_parameters=exclude_freeze_parameters,
                )
            )

    if local_rank in [-1, 0] and info_path is not None:
        with open(info_path, "w") as f:
            json.dump({"best_score": best_score, **kwargs}, f, indent=4)

    if save_checkpoint in ("all", "default", "latest"):
        model.save_checkpoint(os.path.join(ckpt_dir, "pytorch_model_latest"))
        if merge_checkpoint and local_rank in [-1, 0]:
            state_dict = _strip_model_prefix(
                get_fp32_state_dict_from_zero_checkpoint(
                    os.path.join(ckpt_dir, "pytorch_model_latest"),
                    exclude_frozen_parameters=exclude_freeze_parameters,
                )
            )
            torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model_latest.bin"))

    if save_checkpoint in ("all", "every"):
        if merge_checkpoint and local_rank in [-1, 0]:
            state_dict = _strip_model_prefix(
                get_fp32_state_dict_from_zero_checkpoint(
                    os.path.join(ckpt_dir, "pytorch_model_latest"),
                    exclude_frozen_parameters=exclude_freeze_parameters,
                )
            )
            torch.save(state_dict, os.path.join(ckpt_dir, f"pytorch_model_{snapshot_time}.bin"))
        else:
            model.save_checkpoint(ckpt_dir, f"pytorch_model_{snapshot_time}")

    return best_score


@register_task("core/task/deepspeed/supervised")
class DeepspeedTask:
    """Supervised learning task backed by DeepSpeed."""

    def __init__(
        self,
        configure,
        model,
        datasets,
        local_rank: int = -1,      # GPU index for distributed training; -1 for single-GPU
        seed: int = 1123,           # global random seed for reproducibility
        cpu_offload: bool = False,  # keep model on CPU (e.g. for ZeRO-Infinity offload)
    ):
        set_seed(seed)
        self.n_gpu = 1 if torch.cuda.is_available() else 0
        if dist.is_initialized():
            self.n_gpu = dist.get_world_size()

        self.config = configure
        self.model = model
        self.datasets = datasets
        self.local_rank = local_rank

        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)

        if torch.cuda.is_available() and not cpu_offload:
            self.model = self.model.cuda()

        self.best_score = -np.inf

    @classmethod
    @config_defaults_init("core/task/deepspeed/supervised")
    def from_config(cls, config, **kwargs):
        try:
            deepspeed.init_distributed(dist_backend="nccl", init_method="env://")
        except Exception:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/deepspeed/supervised")

        model = config.getoption("model", None)
        dataset = config.getoption("dataset", None)

        if model is not None:
            model = init_registered_module(model, config, registered_model)
        if dataset is not None:
            dataset = init_registered_module(dataset, config, registered_dataset)

        return dict(
            configure=config,
            model=model,
            datasets=dataset,
            local_rank=config.getdefault("core/cli", "local_rank", get_local_rank()),
            cpu_offload=config.getdefault("core/task/deepspeed/supervised", "cpu_offload", False),
        )

    @config_defaults_method("core/task/deepspeed/supervised")
    def train(
        self,
        config_path: str,                               # path to DeepSpeed JSON config file
        optim: str,                                     # registered optimizer name
        loss_fn: str,                                   # registered loss function name
        score_fn: str,                                  # registered scoring function name
        monitor_fns: Optional[Union[str, List[str]]] = None,  # extra metrics logged at checkpoints
        from_ckpt_dir: str = "./from_ckpt",            # directory to load pretrained weights from
        to_ckpt_dir: str = "./to_ckpt",                # directory to write checkpoints to
        train_batch_size: int = 128,                   # per-GPU micro-batch size for training
        dev_batch_size: int = 128,                     # per-GPU batch size for validation
        pin_memory: bool = True,                       # pin DataLoader memory for faster GPU transfer
        num_workers: int = 4,                          # DataLoader worker processes
        save_optimizer: bool = False,                  # include optimizer state in non-ZeRO-3 checkpoints
        save_scheduler: bool = False,                  # include scheduler state in non-ZeRO-3 checkpoints
        save_checkpoint: str = "default",              # checkpoint policy: default/best/latest/every/all
        log_freq: int = 100,                           # log training loss every N steps
        ckpt_freq: int = 10000,                        # save checkpoint every N steps
        grad_acc_step: int = 1,                        # gradient accumulation steps before optimizer update
        learning_rate: Optional[float] = None,         # override optimizer lr in DeepSpeed config
        max_warmup_learning_rate: Optional[float] = None,  # WarmupLR warmup_max_lr override
        num_warmup_steps: Optional[int] = None,        # WarmupLR warmup_num_steps override
        epochs: int = 5,                               # total training epochs
        zero_stage: Optional[int] = None,              # ZeRO optimisation stage (1/2/3); inferred from config if None
        merge_zero3_checkpoint: bool = True,           # merge ZeRO-3 shards into a single fp32 bin
        exclude_freeze_parameters: bool = True,        # skip frozen parameters when merging ZeRO-3
        use_ema: bool = False,                         # maintain an EMA shadow model for evaluation
        ema_decay: float = 0.9999,                     # EMA decay factor
        ema_tau: int = 2000,                           # EMA warm-up steps
    ):
        if self.local_rank in [-1, 0]:
            os.makedirs(to_ckpt_dir, exist_ok=True)

        if loss_fn is not None:
            loss_fn = init_registered_module(loss_fn, self.config, registered_loss)
        if score_fn is not None:
            score_fn = init_registered_module(score_fn, self.config, registered_score)
        if monitor_fns is not None:
            monitor_fns = [
                init_registered_module(fn, self.config, registered_score)
                for fn in monitor_fns
                if fn in registered_score
            ]

        config_file = cached_path(config_path)
        with open(config_file) as f:
            config_dict = json.load(f)
        config_dict["train_micro_batch_size_per_gpu"] = train_batch_size

        if zero_stage is None:
            zero_stage = nested_dict_value(config_dict, "zero_optimization", "stage") or 2

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        # Resume from latest checkpoint for non-ZeRO-3 (ZeRO-3 uses load_checkpoint below)
        if os.path.exists(to_ckpt_dir) and zero_stage != 3:
            self.model.from_checkpoint(to_ckpt_dir, weight_name="pytorch_model_latest.bin")

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        assert "optimizer" in config_dict
        update_nested_dict(config_dict, "zero_optimization", "stage", zero_stage)

        if learning_rate is not None:
            update_nested_dict(config_dict, "optimizer", "params", "lr", learning_rate)

        scheduler_type = nested_dict_value(config_dict, "scheduler", "type")
        if scheduler_type == "WarmupLR":
            if learning_rate is not None:
                update_nested_dict(config_dict, "scheduler", "params", "warmup_max_lr", learning_rate)
            if max_warmup_learning_rate is not None:
                update_nested_dict(config_dict, "scheduler", "params", "warmup_max_lr", max_warmup_learning_rate)
            if num_warmup_steps is not None:
                update_nested_dict(config_dict, "scheduler", "params", "warmup_num_steps", num_warmup_steps)

        info_path = os.path.join(to_ckpt_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
        else:
            info = {}

        global_epoch = info.get("global_epoch", 0)
        global_step = info.get("global_step", 0)
        self.best_score = info.get("best_score", self.best_score)
        logging.info("best score so far: %s", self.best_score)

        self.ema_model = None
        if use_ema and zero_stage != 3:
            self.ema_model = ExponentialMovingAverage(
                self.model,
                decay=ema_decay,
                tau=ema_tau,
                num_steps=info.get("num_ema_steps", 0),
            )
            if os.path.exists(from_ckpt_dir):
                self.ema_model.from_checkpoint(from_ckpt_dir, weight_name="pytorch_ema_model.bin")
            if os.path.exists(to_ckpt_dir):
                self.ema_model.from_checkpoint(to_ckpt_dir, weight_name="pytorch_ema_model_latest.bin")

        for name, param in self.model.named_parameters():
            logging.debug(
                "%s: trainable=%s dtype=%s shape=%s device=%s",
                name, param.requires_grad, param.dtype, param.shape, param.device,
            )

        self.model, optim, _, scheduler = deepspeed.initialize(
            model=self.model,
            config=config_dict,
            model_parameters=params,
        )

        # ZeRO-3 checkpoints must be loaded after deepspeed.initialize
        if os.path.exists(os.path.join(to_ckpt_dir, "pytorch_model_latest")) and zero_stage == 3:
            self.model.load_checkpoint(os.path.join(to_ckpt_dir, "pytorch_model_latest"))

        global_rank = dist.get_rank() if self.n_gpu > 1 else -1
        train_sampler = DistributedSkipSampler if self.n_gpu > 1 else RandomSkipSampler
        dev_sampler = DistributedSampler if self.n_gpu > 1 else SequentialSampler

        dataset_train = self.datasets.get("train")
        dataset_dev = self.datasets.get("dev")

        iter_train = DataLoader(
            dataset_train,
            sampler=train_sampler(dataset_train) if not isinstance(dataset_train, Iterable) else None,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler(dataset_dev) if not isinstance(dataset_dev, Iterable) else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        snapshot_kwargs = dict(
            save_checkpoint=save_checkpoint,
            merge_checkpoint=merge_zero3_checkpoint,
            exclude_freeze_parameters=exclude_freeze_parameters,
            info_path=info_path,
            local_rank=self.local_rank,
        )

        def _snapshot(epoch, step):
            if zero_stage == 3:
                return save_snapshot_zero_3(
                    self.model,
                    to_ckpt_dir,
                    iter_dev,
                    score_fn,
                    monitor_fns,
                    best_score=self.best_score,
                    global_epoch=epoch,
                    global_step=step,
                    **snapshot_kwargs,
                )
            return save_snapshot(
                self.model.module,
                to_ckpt_dir,
                iter_dev,
                score_fn,
                monitor_fns,
                optim=optim if save_optimizer else None,
                scheduler=scheduler if save_scheduler else None,
                ema_model=self.ema_model if use_ema else None,
                best_score=self.best_score,
                global_epoch=epoch,
                global_step=step,
                **snapshot_kwargs,
            )

        log_loss = 0.0
        dev_epoch = 0

        for e in range(epochs):
            torch.cuda.empty_cache()
            if e < global_epoch:
                continue

            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(e)
            if hasattr(dataset_train, "set_skip_step"):
                dataset_train.set_skip_step(global_step * train_batch_size)
            if hasattr(iter_train.sampler, "set_epoch"):
                iter_train.sampler.set_epoch(e)
            if hasattr(iter_train.sampler, "set_skip_step"):
                iter_train.sampler.set_skip_step(global_step * train_batch_size)

            self.model.train()
            is_update_step = True

            for step, (inputs, targets) in enumerate(iter_train):
                step = step + global_step
                is_update_step = False

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = self.model(**inputs.dict())
                loss = (
                    outputs.loss if isinstance(outputs, LossOutputs)
                    else loss_fn(outputs=outputs, targets=targets)
                ) / grad_acc_step

                self.model.backward(loss)
                log_loss += loss.item() * grad_acc_step

                if (step + 1) % grad_acc_step == 0:
                    is_update_step = True
                    self.model.step()
                    if use_ema and self.ema_model is not None:
                        self.ema_model.step(self.model.module)

                if (step + 1) % log_freq == 0 and global_rank in [-1, 0]:
                    avg_loss = log_loss / log_freq
                    logging.info("epoch %d step %d: train/loss=%.6f", e, step, avg_loss)
                    if wandb.is_available():
                        wandb.log({"epoch": e, "step": step, "train/loss": avg_loss})
                    log_loss = 0.0

                if (step + 1) % ckpt_freq == 0:
                    if hasattr(dataset_dev, "set_epoch"):
                        dataset_dev.set_epoch(dev_epoch)
                    if hasattr(iter_dev.sampler, "set_epoch"):
                        iter_dev.sampler.set_epoch(dev_epoch)
                    dev_epoch += 1
                    self.best_score = _snapshot(e, step + 1)

            # Flush remaining accumulated gradients at epoch end
            if not is_update_step:
                self.model.step()
                if use_ema and self.ema_model is not None:
                    self.ema_model.step(self.model.module)

            log_loss = 0.0

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)
            if hasattr(iter_dev.sampler, "set_epoch"):
                iter_dev.sampler.set_epoch(dev_epoch)
            dev_epoch += 1

            global_step = 0
            self.best_score = _snapshot(e + 1, 0)

    @torch.no_grad()
    @config_defaults_method("core/task/deepspeed/supervised")
    def eval(
        self,
        monitor_fns: Union[str, List[str]],  # list of registered scoring function names
        from_ckpt_dir: str = "./from_ckpt",  # directory to load model weights from
        dev_batch_size: int = 128,           # per-GPU batch size for evaluation
        pin_memory: bool = True,
        num_workers: int = 4,
    ):
        monitor_fns = [
            init_registered_module(fn, self.config, registered_score)
            for fn in monitor_fns
            if fn in registered_score
        ]

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        global_rank = -1
        if self.n_gpu > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            global_rank = dist.get_rank()

        dev_sampler = DistributedSampler if self.n_gpu > 1 else SequentialSampler
        dataset_dev = self.datasets.get("dev")
        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler(dataset_dev) if not isinstance(dataset_dev, Iterable) else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        results = infer(self.model.module, iter_dev)
        if global_rank in [-1, 0]:
            monitor(outputs=results.outputs, targets=results.targets, monitor_fns=monitor_fns)

    @torch.no_grad()
    @config_defaults_method("core/task/deepspeed/supervised")
    def infer(
        self,
        postprocess_fn: str,                    # registered postprocessing function name
        writer: str,                            # registered writer name for output serialisation
        test_batch_size: int = 128,            # per-GPU batch size for inference
        pin_memory: bool = True,
        num_workers: int = 4,
        max_size: int = 10000,                 # maximum queue depth for async postprocessing
        from_ckpt_dir: str = "./from_ckpt",   # directory to load model weights from
        output_header: Optional[List] = None, # column names to copy from raw dataset into output
        output_path: str = "./output.txt",    # file path for inference results
        postprocess_workers: int = 2,         # number of parallel postprocessing workers
    ):
        assert self.n_gpu <= 1, "inference only supports single-GPU mode"
        assert writer is not None

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if postprocess_fn is not None:
            postprocess_fn = init_registered_process(postprocess_fn, self.config)

        writer = init_registered_module(
            writer, self.config, registered_writer, output_file=output_path
        )
        skip_step = writer.skip_n_samples

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        sampler = SequentialSkipSampler if skip_step > 0 else SequentialSampler
        dataset_test = self.datasets.get("test")

        iter_test = DataLoader(
            dataset_test,
            sampler=sampler(dataset_test) if not isinstance(dataset_test, Iterable) else None,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        if skip_step > 0:
            if hasattr(dataset_test, "set_skip_step"):
                dataset_test.set_skip_step(skip_step)
            if hasattr(iter_test.sampler, "set_skip_step"):
                iter_test.sampler.set_skip_step(skip_step)

        # Build a parallel loader for raw dataset metadata when available
        iter_data = None
        if hasattr(dataset_test, "dataset"):
            data_info = DatasetFeature(dataset_test.dataset)
            iter_data = DataLoader(
                deepcopy(data_info),
                sampler=sampler(data_info) if not isinstance(dataset_test, Iterable) else None,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=None,
            )
            if skip_step > 0 and hasattr(iter_data.sampler, "set_skip_step"):
                iter_data.sampler.set_skip_step(skip_step)

        self.model.eval()
        start = time.time()

        data_queue = Queue(maxsize=max_size)
        msg_queue = Queue(maxsize=max_size)
        postprocess_list = [
            PostProcess(postprocess_fn, data_queue, msg_queue)
            for _ in range(postprocess_workers)
        ]
        for p in postprocess_list:
            p.start()

        io_process = IOProcess(msg_queue, writer=writer)
        io_process.start()

        if iter_data is None:
            for step, (inputs, _) in enumerate(iter_test):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict()).cpu()
                data_queue.put((step, outputs))
        else:
            for step, ((inputs, _), raw_info) in enumerate(zip(iter_test, iter_data)):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict()).cpu()
                if output_header is not None:
                    raw_info = {k: raw_info[k] for k in output_header if k in raw_info}
                    outputs.from_pandas(pd.DataFrame(raw_info))
                data_queue.put((step, outputs))

        data_queue.put((-1, GENERATE_FINISHED))
        for p in postprocess_list:
            p.join()

        msg_queue.put((-1, GENERATE_FINISHED))
        io_process.join()

        elapsed_ms = (time.time() - start) * 1000
        throughput = (len(dataset_test) - skip_step) / elapsed_ms * 1000
        logging.info("%.2f ms | %.2f samples/s", elapsed_ms, throughput)
