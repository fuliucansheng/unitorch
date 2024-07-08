# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import time
import json
import logging
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from PIL import Image
from itertools import chain
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.multiprocessing import Process, Queue
from unitorch import set_seed, is_torch2_available
from unitorch.models import ExponentialMovingAverage
from unitorch.utils import get_local_rank
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


class DatasetFeature(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        row = self.dataset[idx]
        ret = {}
        for k, v in row.items():
            if isinstance(v, Image.Image):
                v = np.array(v).tolist()
            if not isinstance(v, str):
                v = json.dumps(v)
            ret[k] = v
        return ret

    def __len__(self):
        return len(self.dataset)


def collate_fn(bufs):
    combine_inputs, combine_targets = list(zip(*bufs))
    if isinstance(combine_inputs[0], ModelInputs):
        inputs = type(combine_inputs[0]).stack(*combine_inputs)
    else:
        combine_inputs = [
            type(_inputs[0]).stack(*_inputs) for _inputs in list(zip(*combine_inputs))
        ]
        inputs = CombineTensorsInputs()
        for _inputs in combine_inputs:
            inputs.add(_inputs)

    if isinstance(combine_targets[0], ModelTargets):
        targets = type(combine_targets[0]).stack(*combine_targets)
    else:
        combine_targets = [
            type(_targets[0]).stack(*_targets)
            for _targets in list(zip(*combine_targets))
        ]
        targets = CombineTensorsTargets()
        for _targets in combine_targets:
            targets.add(_targets)

    return inputs, targets


@torch.no_grad()
def infer(model, iter_data):
    model.eval()
    outputs, targets = [], []
    for _, (_inputs, _targets) in enumerate(iter_data):
        if torch.cuda.is_available():
            _inputs = _inputs.cuda()
            _targets = _targets.cuda()
        _outputs = model(**_inputs.dict())
        outputs.append(_outputs)
        targets.append(_targets)

    if isinstance(outputs[0], LossOutputs):
        outputs = LossOutputs(
            loss=torch.tensor([output.loss for output in outputs]).to(
                device=outputs[0].loss.device
            )
        )
        if dist.is_initialized():
            outputs = outputs.cuda().sync().cpu()
        else:
            outputs = outputs.cpu()
    else:
        outputs = type(outputs[0]).union(*outputs)
        targets = type(targets[0]).union(*targets)

        if dist.is_initialized():
            outputs = outputs.cuda().sync().cpu()
            targets = targets.cuda().sync().cpu()
        else:
            outputs = outputs.cpu()
            targets = targets.cpu()
    model.train()
    return GenericOutputs(outputs=outputs, targets=targets)


def monitor(outputs, targets, monitor_fns):
    if monitor_fns is None:
        return

    for monitor_fn in monitor_fns:
        score = monitor_fn(outputs=outputs, targets=targets)
        info = str(type(monitor_fn).__name__)
        logging.info(f"{info} is {score}")
    return


def save_checkpoint(
    model,
    ckpt_dir,
    iter_dev,
    score_fn,
    monitor_fns,
    optim,
    scheduler,
    ema_model=None,
    best_score=-np.inf,
    info_path=None,
    local_rank=-1,
    **kwargs,
):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if ema_model is not None:
        base_model = ema_model
    else:
        base_model = model

    results = infer(base_model, iter_dev)
    if local_rank in [-1, 0]:
        new_score = score_fn(outputs=results.outputs, targets=results.targets)
        monitor(results.outputs, results.targets, monitor_fns)
        if new_score > best_score:
            best_score = new_score
            if model:
                model.save_checkpoint(
                    ckpt_dir=ckpt_dir, weight_name="pytorch_model.bin"
                )
            if ema_model:
                ema_model.save_checkpoint(
                    ckpt_dir=ckpt_dir, weight_name="pytorch_ema_model.bin"
                )
            if optim:
                optim.save_checkpoint(
                    ckpt_dir=ckpt_dir, weight_name="pytorch_optim.bin"
                )
            if scheduler:
                scheduler.save_checkpoint(
                    ckpt_dir=ckpt_dir, weight_name="pytorch_scheduler.bin"
                )
        if model:
            model.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name="pytorch_model_latest.bin"
            )
        if ema_model:
            ema_model.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name="pytorch_ema_model_latest.bin"
            )
            kwargs["num_ema_steps"] = ema_model.num_steps
        if optim:
            optim.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name="pytorch_optim_latest.bin"
            )
        if scheduler:
            scheduler.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name="pytorch_scheduler_latest.bin"
            )
        if info_path is not None:
            json.dump({"best_score": best_score, **kwargs}, open(info_path, "w"))
    return best_score


@register_task("core/task/supervised")
class SupervisedTask:
    def __init__(
        self,
        configure,
        model,
        datasets,
        local_rank: Optional[int] = -1,
        seed: Optional[int] = 1123,
    ):
        """
        Initialize the SupervisedTask.

        Args:
            configure: The configuration object.
            model: The model for the task.
            datasets: The datasets for training and evaluation.
            local_rank (optional): The local rank for distributed training. Defaults to -1.
            seed (optional): The random seed. Defaults to 1123.
        """
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

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.best_score = -np.inf

    @classmethod
    @add_default_section_for_init("core/task/supervised")
    def from_core_configure(cls, config, **kwargs):
        """
        Create a SupervisedTask instance from the core configuration.

        Args:
            config: The core configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the configuration, model, datasets, and local rank.
        """
        try:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        except:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/supervised")

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

    @add_default_section_for_function("core/task/supervised")
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
        save_optimizer: Optional[bool] = True,
        save_scheduler: Optional[bool] = True,
        log_freq: Optional[int] = 100,
        ckpt_freq: Optional[int] = 10000,
        grad_acc_step: Optional[int] = 1,
        max_grad_norm: Optional[float] = 1.0,
        num_training_samples: Optional[int] = 1000000000,
        epochs: Optional[int] = 5,
        use_ema: Optional[bool] = False,
        ema_decay: Optional[float] = 0.9999,
        ema_tau: Optional[int] = 2000,
        use_amp: Optional[bool] = True,
        gpu_mode: Optional[bool] = False,
    ):
        """
        Train the model.

        Args:
            optim: The optimizer for training.
            loss_fn: The loss function for training.
            score_fn: The scoring function for evaluation.
            monitor_fns (optional): The monitoring functions for evaluation. Defaults to None.
            scheduler (optional): The scheduler for adjusting the learning rate. Defaults to None.
            from_ckpt_dir (optional): The directory path to load checkpoints from. Defaults to "./from_ckpt".
            to_ckpt_dir (optional): The directory path to save checkpoints to. Defaults to "./to_ckpt".
            train_batch_size (optional): The batch size for training. Defaults to 128.
            dev_batch_size (optional): The batch size for evaluation. Defaults to 128.
            pin_memory (optional): Whether to pin memory during data loading. Defaults to True.
            num_workers (optional): The number of worker processes for data loading. Defaults to 4.
            save_optimizer (optional): Whether to save the optimizer. Defaults to True.
            save_scheduler (optional): Whether to save the scheduler. Defaults to True.
            log_freq (optional): The frequency of logging training information. Defaults to 100.
            ckpt_freq (optional): The frequency of saving checkpoints. Defaults to 10000.
            grad_acc_step (optional): The number of gradient accumulation steps. Defaults to 1.
            max_grad_norm (optional): The maximum gradient norm for gradient clipping. Defaults to 1.0.
            num_training_samples (optional): The number of training samples. Defaults to 1000000000.
            epochs (optional): The number of training epochs. Defaults to 5.
            use_ema (optional): Whether to use exponential moving average. Defaults to False.
            ema_decay (optional): The decay rate for exponential moving average. Defaults to 0.9999.
            ema_tau (optional): The time constant for exponential moving average. Defaults to 2000.
            use_amp (optional): Whether to use automatic mixed precision. Defaults to True.
            gpu_mode (optional): Whether to make GPU active. Defaults to False.
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
            optim.from_checkpoint(
                from_ckpt_dir,
                weight_name="pytorch_optim.bin",
            )

        if os.path.exists(to_ckpt_dir):
            self.model.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_model_latest.bin",
            )
            optim.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_optim_latest.bin",
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

        self.ema_model = None
        if use_ema:
            num_ema_steps = info.get("num_ema_steps", 0)
            self.ema_model = ExponentialMovingAverage(
                self.model,
                decay=ema_decay,
                tau=ema_tau,
                num_steps=num_ema_steps,
            )
            if os.path.exists(from_ckpt_dir):
                self.ema_model.from_checkpoint(
                    from_ckpt_dir,
                    weight_name="pytorch_ema_model.bin",
                )
            if os.path.exists(to_ckpt_dir):
                self.ema_model.from_checkpoint(
                    to_ckpt_dir,
                    weight_name="pytorch_ema_model_latest.bin",
                )

        for n, p in self.model.named_parameters():
            logging.debug(
                f"{n}: trainable - {p.requires_grad} | tensor shape - {p.shape}"
            )

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

        train_sampler = DistributedSkipSampler if self.n_gpu > 1 else RandomSkipSampler
        dev_sampler = DistributedSampler if self.n_gpu > 1 else SequentialSampler

        dataset_train = self.datasets.get("train")
        dataset_dev = self.datasets.get("dev")

        iter_train = DataLoader(
            dataset_train,
            sampler=train_sampler(dataset_train)
            if not isinstance(dataset_train, Iterable)
            else None,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler(dataset_dev)
            if not isinstance(dataset_dev, Iterable)
            else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        if scheduler is not None:
            if not isinstance(dataset_train, Iterable):
                num_training_steps = int(
                    epochs
                    * len(dataset_train)
                    // train_batch_size
                    // max(1, self.n_gpu)
                    // grad_acc_step
                )
            else:
                num_training_steps = int(
                    epochs
                    * num_training_samples
                    // train_batch_size
                    // max(1, self.n_gpu)
                    // grad_acc_step
                )

            scheduler = init_registered_module(
                scheduler,
                self.config,
                registered_scheduler,
                optimizer=optim,
                num_training_steps=num_training_steps,
            )

        if scheduler and os.path.exists(to_ckpt_dir):
            scheduler.from_checkpoint(
                to_ckpt_dir,
                weight_name="pytorch_scheduler_latest.bin",
            )

        if use_amp:
            scaler = GradScaler()

        log_loss = 0
        dev_epoch = 0
        for e in range(0, epochs):
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

                if is_torch2_available():
                    with torch.cuda.amp.autocast(
                        enabled=use_amp
                    ) as autocast, torch.backends.cuda.sdp_kernel(
                        enable_flash=False
                    ) as disable:
                        outputs = self.model(**inputs.dict())
                        if isinstance(outputs, LossOutputs):
                            loss = outputs.loss / grad_acc_step
                        else:
                            loss = (
                                loss_fn(outputs=outputs, targets=targets)
                                / grad_acc_step
                            )
                else:
                    with torch.cuda.amp.autocast(enabled=use_amp) as autocast:
                        outputs = self.model(**inputs.dict())
                        if isinstance(outputs, LossOutputs):
                            loss = outputs.loss / grad_acc_step
                        else:
                            loss = (
                                loss_fn(outputs=outputs, targets=targets)
                                / grad_acc_step
                            )

                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                log_loss += loss.data * grad_acc_step
                if (step + 1) % grad_acc_step == 0:
                    is_update_step = True
                    if use_amp:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad()

                    if use_ema and self.ema_model is not None:
                        self.ema_model.step(
                            self.model.module if self.n_gpu > 1 else self.model
                        )

                if (step + 1) % log_freq == 0 and global_rank in [-1, 0]:
                    logging.info(
                        f"epoch {e} step {step}: loss -- { log_loss / log_freq }"
                    )
                    log_loss = 0

                if (step + 1) % ckpt_freq == 0:
                    if hasattr(dataset_dev, "set_epoch"):
                        dataset_dev.set_epoch(dev_epoch)

                    if hasattr(iter_dev.sampler, "set_epoch"):
                        iter_dev.sampler.set_epoch(dev_epoch)

                    dev_epoch += 1
                    self.best_score = save_checkpoint(
                        self.model.module if self.n_gpu > 1 else self.model,
                        to_ckpt_dir,
                        iter_dev,
                        score_fn,
                        monitor_fns,
                        optim=optim if save_optimizer else None,
                        scheduler=scheduler if save_scheduler else None,
                        ema_model=self.ema_model if use_ema else None,
                        best_score=self.best_score,
                        info_path=info_path,
                        local_rank=self.local_rank,
                        global_epoch=e,
                        global_step=step + 1,
                    )

            if not is_update_step:
                scaler.step(optim)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad()

                if use_ema and self.ema_model is not None:
                    self.ema_model.step(
                        self.model.module if self.n_gpu > 1 else self.model
                    )

            log_loss = 0

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)

            if hasattr(iter_dev.sampler, "set_epoch"):
                iter_dev.sampler.set_epoch(dev_epoch)

            dev_epoch += 1

            global_step = 0
            self.best_score = save_checkpoint(
                self.model.module if self.n_gpu > 1 else self.model,
                to_ckpt_dir,
                iter_dev,
                score_fn,
                monitor_fns,
                optim=optim if save_optimizer else None,
                scheduler=scheduler if save_scheduler else None,
                ema_model=self.ema_model if use_ema else None,
                best_score=self.best_score,
                info_path=info_path,
                local_rank=self.local_rank,
                global_epoch=e,
                global_step=0,
            )

    @torch.no_grad()
    @add_default_section_for_function("core/task/supervised")
    def eval(
        self,
        monitor_fns: Union[str, List[str]],
        from_ckpt_dir: Optional[str] = "./from_ckpt",
        dev_batch_size: Optional[int] = 128,
        pin_memory: Optional[bool] = True,
        num_workers: Optional[int] = 4,
    ):
        """
        Perform evaluation on the model.

        Args:
            monitor_fns: The monitoring functions for evaluation.
            from_ckpt_dir (optional): The directory path to load checkpoints from. Defaults to "./from_ckpt".
            dev_batch_size (optional): The batch size for evaluation. Defaults to 128.
            pin_memory (optional): Whether to pin memory during data loading. Defaults to True.
            num_workers (optional): The number of worker processes for data loading. Defaults to 4.
        """
        monitor_fns = [
            init_registered_module(monitor_fn, self.config, registered_score)
            for monitor_fn in monitor_fns
            if monitor_fn in registered_score
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
            sampler=dev_sampler(dataset_dev)
            if not isinstance(dataset_dev, Iterable)
            else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        results = infer(self.model.module if self.n_gpu > 1 else self.model, iter_dev)
        if global_rank in [-1, 0]:
            monitor(
                outputs=results.outputs,
                targets=results.targets,
                monitor_fns=monitor_fns,
            )

    @torch.no_grad()
    @add_default_section_for_function("core/task/supervised")
    def infer(
        self,
        postprocess_fn: str,
        writer: str,
        test_batch_size: Optional[int] = 128,
        pin_memory: Optional[bool] = True,
        num_workers: Optional[int] = 4,
        max_size: Optional[int] = 10000,
        from_ckpt_dir: Optional[str] = "./from_ckpt",
        output_header: Optional[List] = None,
        output_path: Optional[str] = "./cache/predict.txt",
        postprocess_workers: Optional[int] = 2,
    ):
        """
        Perform inference using the model.

        Args:
            postprocess_fn: The postprocessing function for inference.
            writer: The writer to save the inference results.
            test_batch_size (optional): The batch size for inference. Defaults to 128.
            pin_memory (optional): Whether to pin memory during data loading. Defaults to True.
            num_workers (optional): The number of worker processes for data loading. Defaults to 4.
            max_size (optional): The maximum number of samples to process. Defaults to 10000.
            from_ckpt_dir (optional): The directory path to load checkpoints from. Defaults to "./from_ckpt".
            output_header (optional): The header for the output file. Defaults to None.
            output_path (optional): The path to save the output file. Defaults to "./cache/predict.txt".
            postprocess_workers (optional): The number of worker processes for postprocessing. Defaults to 2.
        """
        assert self.n_gpu <= 1
        assert writer is not None

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if postprocess_fn is not None:
            postprocess_fn = init_registered_process(postprocess_fn, self.config)

        if writer is not None:
            writer = init_registered_module(
                writer,
                self.config,
                registered_writer,
                output_file=output_path,
            )

        skip_step = writer.skip_n_samples

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)

        if skip_step == 0:
            sampler = SequentialSampler
        else:
            sampler = SequentialSkipSampler

        dataset_test = self.datasets.get("test")

        iter_test = DataLoader(
            dataset_test,
            sampler=sampler(dataset_test)
            if not isinstance(dataset_test, Iterable)
            else None,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        if skip_step > 0 and hasattr(dataset_test, "set_skip_step"):
            dataset_test.set_skip_step(skip_step)

        if skip_step > 0 and hasattr(iter_test.sampler, "set_skip_step"):
            iter_test.sampler.set_skip_step(skip_step)

        if hasattr(dataset_test, "dataset"):
            data_info = dataset_test.dataset
            data_info = DatasetFeature(data_info)
            iter_data = DataLoader(
                deepcopy(data_info),
                sampler=sampler(data_info)
                if not isinstance(dataset_test, Iterable)
                else None,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=None,
            )
        else:
            iter_data = None

        if skip_step > 0 and hasattr(iter_data.sampler, "set_skip_step"):
            iter_data.sampler.set_skip_step(skip_step)

        self.model.eval()
        start = time.time()

        data_queue = Queue(maxsize=max_size)
        msg_queue = Queue(maxsize=max_size)
        postprocess_list = []
        for _ in range(postprocess_workers):
            p = PostProcess(
                postprocess_fn,
                data_queue,
                msg_queue,
            )
            postprocess_list.append(p)
            p.start()

        io_process = IOProcess(msg_queue, writer=writer)
        io_process.start()

        if iter_data is None:
            for step, (inputs, _) in enumerate(iter_test):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict())
                outputs = outputs.cpu()
                data_queue.put((step, outputs))
        else:
            for step, ((inputs, _), _infos) in enumerate(zip(iter_test, iter_data)):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict())
                outputs = outputs.cpu()
                if output_header is not None:
                    _infos = {k: _infos[k] for k in output_header if k in _infos}
                    outputs.from_pandas(pd.DataFrame(_infos))
                data_queue.put((step, outputs))

        data_queue.put((-1, GENERATE_FINISHED))
        for p in postprocess_list:
            p.join()

        msg_queue.put((-1, GENERATE_FINISHED))
        io_process.join()

        end = time.time()
        ms = (end - start) * 1000
        logging.info(
            "{:.2f} ms, {:.2f} sample/s".format(
                ms,
                ((len(dataset_test) - skip_step) / ms * 1000),
            )
        )
