# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import time
import json
import logging
import shutil
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from itertools import chain
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process, Queue
from megatron.core import mpu, dist_checkpointing, tensor_parallel
from megatron.core.pipeline_parallel.schedules import (
    get_forward_backward_func,
    forward_backward_no_pipelining,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    DistributedDataParallel,
)
from megatron.core.optimizer import DistributedOptimizer
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
from unitorch.models import GenericOutputs
from unitorch.cli import (
    cached_path,
    register_task,
    registered_model,
    registered_optim,
    registered_dataset,
    registered_loss,
    registered_score,
    registered_scheduler,
    registered_writer,
    init_registered_module,
    init_registered_process,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models import (
    ModelInputs,
    ModelOutputs,
    ModelTargets,
    LossOutputs,
    CombineTensorsInputs,
    CombineTensorsTargets,
)
from unitorch.cli.tasks.supervised import collate_fn
import unitorch.cli.wandb as wandb


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1  # if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)],
                    device="cpu",
                    pin_memory=True,
                ).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch


def process_batch_data(inputs, targets):
    data = [inputs, targets]
    dist.broadcast_object_list(
        data,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    inputs, targets = data[0], data[1]

    batch = {**inputs, **targets}
    batch = {k: v.cuda() if v is not None else v for k, v in batch.items()}
    batch = get_batch_on_this_cp_rank(batch)
    return batch


@register_task("core/task/megatron/supervised")
class MegatronTask:
    """Task class for deepspeed supervised learning."""

    def __init__(
        self,
        configure,
        model,
        datasets,
        local_rank: Optional[int] = -1,
        seed: Optional[int] = 1123,
    ):
        """
        Initialize the DeepspeedTask.

        Args:
            configure: The configuration object.
            model: The model for supervised learning.
            datasets: The datasets for training and evaluation.
            local_rank (optional): The local rank for distributed training. Defaults to -1.
            seed (optional): The random seed. Defaults to 1123.
        """
        set_seed(seed)
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
        self.config = configure
        self.model = model
        self.datasets = datasets
        self.local_rank = local_rank

        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.best_score = -np.inf
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self.cp_rank = mpu.get_context_parallel_rank()
        self.cp_group = mpu.get_context_parallel_group()
        self.cp_size = mpu.get_context_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.is_pp_last_rank = mpu.is_pipeline_last_stage(ignore_virtual=True)

    @classmethod
    @add_default_section_for_init("core/task/megatron/supervised")
    def from_core_configure(cls, config, **kwargs):
        """
        Create a MegatronTask instance from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A MegatronTask instance.
        """
        try:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        except:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/megatron/supervised")

        tensor_model_parallel_size = config.getoption(
            "tensor_model_parallel_size",
            1,
        )
        pipeline_model_parallel_size = config.getoption(
            "pipeline_model_parallel_size",
            1,
        )
        context_parallel_size = config.getoption(
            "context_parallel_size",
            1,
        )
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )

        model = config.getoption("model", None)
        dataset = config.getoption("dataset", None)

        if model is not None:
            model = init_registered_module(model, config, registered_model)

        if dataset is not None:
            dataset = init_registered_module(dataset, config, registered_dataset)

        local_rank = config.getdefault(
            "core/cli",
            "local_rank",
            get_local_rank(),
        )

        return dict(
            configure=config,
            model=model,
            datasets=dataset,
            local_rank=local_rank,
        )

    @add_default_section_for_function("core/task/megatron/supervised")
    def train(
        self,
        optim: str,
        loss_fn: str,
        score_fn: str,
        monitor_fns: Optional[Union[str, List[str]]] = None,
        scheduler: Optional[str] = None,
        from_ckpt_dir: Optional[str] = "./from_ckpt",
        to_ckpt_dir: Optional[str] = "./to_ckpt",
        train_batch_size: Optional[int] = 128,
        dev_batch_size: Optional[int] = 128,
        pin_memory: Optional[bool] = True,
        num_workers: Optional[int] = 4,
        log_freq: Optional[int] = 100,
        ckpt_freq: Optional[int] = 10000,
        grad_acc_step: Optional[int] = 1,
        max_grad_norm: Optional[float] = 1.0,
        num_training_samples: Optional[int] = 1000000000,
        num_dev_samples: Optional[int] = 10000,
        seq_length: Optional[int] = None,
        epochs: Optional[int] = 5,
    ):
        """
        Train the model using deepspeed.

        Args:
            config_path: The path to the deepspeed configuration file.
            optim: The optimizer used for training.
            loss_fn: The loss function used for training.
            score_fn: The score function used for evaluation.
            monitor_fns (optional): The monitoring functions for evaluation. Defaults to None.
            from_ckpt_dir (optional): The directory path to load checkpoints from. Defaults to "./from_ckpt".
            to_ckpt_dir (optional): The directory path to save checkpoints to. Defaults to "./to_ckpt".
            train_batch_size (optional): The batch size for training. Defaults to 128.
            dev_batch_size (optional): The batch size for evaluation. Defaults to 128.
            pin_memory (optional): Whether to pin memory during data loading. Defaults to True.
            num_workers (optional): The number of worker processes for data loading. Defaults to 4.
            save_optimizer (optional): Whether to save the optimizer state. Defaults to False.
            save_scheduler (optional): Whether to save the scheduler state. Defaults to False.
            log_freq (optional): The frequency of logging. Defaults to 100.
            ckpt_freq (optional): The frequency of saving checkpoints. Defaults to 10000.
            grad_acc_step (optional): The number of gradient accumulation steps. Defaults to 1.
            max_grad_norm (optional): The maximum gradient norm. Defaults to 1.0.
            learning_rate (optional): The learning rate for the optimizer. Defaults to None.
            max_warmup_learning_rate (optional): The maximum learning rate during warmup. Defaults to None.
            num_warmup_steps (optional): The number of warmup steps. Defaults to None.
            epochs (optional): The number of training epochs. Defaults to 5.
        """
        if not os.path.exists(to_ckpt_dir) and self.local_rank in [-1, 0]:
            os.makedirs(to_ckpt_dir, exist_ok=True)

        if loss_fn is not None:
            loss_fn = init_registered_module(loss_fn, self.config, registered_loss)

        if score_fn is not None:
            score_fn = init_registered_module(score_fn, self.config, registered_score)

        if monitor_fns is not None:
            monitor_fns = [
                init_registered_module(monitor_fn, self.config, registered_score)
                for monitor_fn in monitor_fns
                if monitor_fn in registered_score
            ]

        if optim is not None and self.model is not None:
            optim = init_registered_module(
                optim,
                self.config,
                registered_optim,
                params=filter(lambda x: x.requires_grad, self.model.parameters()),
            )

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        if os.path.exists(os.path.join(to_ckpt_dir, "pytorch_model_latest")):
            self.model.from_checkpoint(
                os.path.join(to_ckpt_dir, "pytorch_model_latest")
            )

        info_path = os.path.join(to_ckpt_dir, "info.json")
        if os.path.exists(info_path):
            info = json.load(open(os.path.join(to_ckpt_dir, "info.json")))
        else:
            info = dict()

        global_epoch = info.get("global_epoch", 0)
        global_step = info.get("global_step", 0)
        self.best_score = info.get("best_score", self.best_score)

        logging.info(f"the best score is {self.best_score}")

        for n, p in self.model.named_parameters():
            logging.debug(
                f"{n}: trainable - {p.requires_grad} | tensor dtype - {p.dtype} | tensor shape - {p.shape} | tensor device - {p.device}"
            )

        if self.dp_size > 1:
            self.model = DistributedDataParallel(
                config=self.model.config,
                module=self.model,
                ddp_config=DistributedDataParallelConfig(
                    use_distributed_optimizer=False,
                ),
            )

        dataset_train = self.datasets.get("train")
        dataset_dev = self.datasets.get("dev")

        iter_train = DataLoader(
            dataset_train,
            sampler=None,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        iter_dev = DataLoader(
            dataset_dev,
            sampler=None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        iter_train = iter(iter_train)
        iter_dev = iter(iter_dev)

        if scheduler is not None:
            num_training_steps = int(
                epochs
                * num_training_samples
                // train_batch_size
                // max(1, self.dp_size)
                // grad_acc_step
            )

            scheduler = init_registered_module(
                scheduler,
                self.config,
                registered_scheduler,
                optimizer=optim,
                num_training_steps=num_training_steps,
            )

        log_loss = 0
        dev_epoch = 0
        num_train_steps = num_training_samples // train_batch_size // self.dp_size
        num_dev_steps = num_dev_samples // dev_batch_size // self.dp_size

        if self.pp_size == 1:
            forward_backward_pipeline = forward_backward_no_pipelining
        elif self.pp_size > 1 and self.vp_size is not None:
            forward_backward_pipeline = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_pipeline = forward_backward_pipelining_without_interleaving

        def forward_step_func(_iter, _model):
            inputs, targets = next(_iter)
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = process_batch_data(inputs.dict(), targets.dict())

            outputs = _model(**inputs)
            if isinstance(outputs, LossOutputs):
                loss = outputs.loss
                if self.cp_size > 1:
                    dist.all_reduce(
                        loss,
                        group=self.cp_group,
                    )
                    loss = loss / self.cp_size
                return loss.unsqueeze(0), lambda x: (
                    x.squeeze(0),
                    x.detach().squeeze(0),
                )
            return outputs, None

        for e in range(0, epochs):
            torch.cuda.empty_cache()
            if e < global_epoch:
                continue

            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(e)

            if hasattr(dataset_train, "set_skip_step"):
                dataset_train.set_skip_step(global_step * train_batch_size)

            self.model.train()
            for step in range(num_train_steps):
                step = step + global_step

                loss = forward_backward_pipeline(
                    forward_step_func=forward_step_func,
                    data_iterator=iter_train,
                    model=self.model,
                    num_microbatches=grad_acc_step,
                    seq_length=seq_length,
                    micro_batch_size=train_batch_size,
                )

                for param in self.model.parameters():
                    if hasattr(param, "main_grad") and param.main_grad is not None:
                        param.grad = param.main_grad

                if self.is_pp_last_rank:
                    log_loss += loss[0].item()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm,
                )

                optim.step()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad()

                if (
                    (step + 1) % log_freq == 0
                    and self.dp_rank == 0
                    and self.is_pp_last_rank
                    and self.cp_rank == 0
                    and self.tp_rank == 0
                ):
                    logging.info(
                        f"epoch {e} step {step}: loss -- { log_loss / log_freq }"
                    )
                    if wandb.is_available():
                        wandb.log(
                            {
                                "epoch": e,
                                "step": step,
                                "train/loss": log_loss / log_freq,
                            },
                        )
                    log_loss = 0

                if (step + 1) % ckpt_freq == 0:
                    dist.barrier()

                    if hasattr(dataset_dev, "set_epoch"):
                        dataset_dev.set_epoch(dev_epoch)

                    self.model.eval()
                    _log_loss = 0
                    for _ in range(num_dev_steps):
                        _loss = forward_backward_pipeline(
                            forward_step_func=forward_step_func,
                            data_iterator=iter_dev,
                            model=self.model,
                            num_microbatches=grad_acc_step,
                            seq_length=seq_length,
                            micro_batch_size=dev_batch_size,
                            forward_only=True,
                        )

                        if self.is_pp_last_rank:
                            _log_loss += _loss[0].item() / num_dev_steps

                    self.model.train()
                    dev_epoch += 1

                    if (
                        self.dp_rank == 0
                        and self.is_pp_last_rank
                        and self.cp_rank == 0
                        and self.tp_rank == 0
                    ):
                        logging.info(f"loss is { _log_loss }")
                        if wandb.is_available():
                            wandb.log(
                                {
                                    "val/loss": _log_loss,
                                },
                            )

                    if -_log_loss > self.best_score:
                        getattr(self.model, "module", self.model).save_checkpoint(
                            ckpt_dir=os.path.join(to_ckpt_dir, "pytorch_model"),
                            weight_name="common.pt",
                        )
                        self.best_score = -_log_loss
                        info["best_score"] = self.best_score
                        info["global_epoch"] = e
                        info["global_step"] = step + 1
                        with open(info_path, "w") as f:
                            json.dump(info, f, indent=4)

                    getattr(self.model, "module", self.model).save_checkpoint(
                        ckpt_dir=os.path.join(to_ckpt_dir, "pytorch_model_latest"),
                        weight_name="common.pt",
                    )
                    info["best_score"] = self.best_score
                    info["global_epoch"] = e
                    info["global_step"] = step + 1
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=4)

                    dist.barrier()

            log_loss = 0

            dist.barrier()

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)

            self.model.eval()
            _log_loss = 0
            for _ in range(num_dev_steps):
                _loss = forward_backward_pipeline(
                    forward_step_func=forward_step_func,
                    data_iterator=iter_dev,
                    model=self.model,
                    num_microbatches=grad_acc_step,
                    seq_length=seq_length,
                    micro_batch_size=dev_batch_size,
                    forward_only=True,
                )

                if self.is_pp_last_rank:
                    _log_loss += _loss[0].item() / num_dev_steps

            self.model.train()
            dev_epoch += 1

            if (
                self.dp_rank == 0
                and self.is_pp_last_rank
                and self.cp_rank == 0
                and self.tp_rank == 0
            ):
                logging.info(f"loss is { _log_loss }")
                if wandb.is_available():
                    wandb.log(
                        {
                            "val/loss": _log_loss,
                        },
                    )

            if -_log_loss > self.best_score:
                getattr(self.model, "module", self.model).save_checkpoint(
                    ckpt_dir=os.path.join(to_ckpt_dir, "pytorch_model"),
                    weight_name="common.pt",
                )
                self.best_score = -_log_loss
                info["best_score"] = self.best_score
                info["global_epoch"] = e
                info["global_step"] = step + 1
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=4)

            getattr(self.model, "module", self.model).save_checkpoint(
                ckpt_dir=os.path.join(to_ckpt_dir, "pytorch_model_latest"),
                weight_name="common.pt",
            )
            info["best_score"] = self.best_score
            info["global_epoch"] = e
            info["global_step"] = step + 1
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)

            dist.barrier()

            global_step = 0
