# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from PIL import Image
from collections.abc import Iterable
from typing import List, Optional, Union
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Queue
from unitorch import set_seed
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
    ModelTargets,
    LossOutputs,
    TensorMixInputs,
    TensorMixTargets,
)
import unitorch.cli.wandb as wandb


class DatasetFeature(torch.utils.data.Dataset):
    """Wraps a dataset, serialising every field to a JSON string for collation."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        result = {}
        for k, v in row.items():
            if isinstance(v, Image.Image):
                v = np.array(v).tolist()
            if not isinstance(v, str):
                v = json.dumps(v)
            result[k] = v
        return result


def collate_fn(batch):
    combine_inputs, combine_targets = list(zip(*batch))

    if isinstance(combine_inputs[0], ModelInputs):
        inputs = type(combine_inputs[0]).stack(*combine_inputs)
    else:
        inputs = TensorMixInputs()
        for group in zip(*combine_inputs):
            inputs.add(type(group[0]).stack(*group))

    if isinstance(combine_targets[0], ModelTargets):
        targets = type(combine_targets[0]).stack(*combine_targets)
    else:
        targets = TensorMixTargets()
        for group in zip(*combine_targets):
            targets.add(type(group[0]).stack(*group))

    return inputs, targets


def _build_monitor_fns(names, config):
    """Resolve a string or list of score-function names into initialised callables."""
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    return [
        init_registered_module(name, config, registered_score)
        for name in names
        if name in registered_score
    ]


@torch.no_grad()
def run_inference(model, data_loader):
    """Run full-dataset inference and return (outputs, targets) on CPU."""
    model.eval()
    all_outputs, all_targets = [], []

    for inputs, targets in data_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        all_outputs.append(model(**inputs.dict()))
        all_targets.append(targets)

    if isinstance(all_outputs[0], LossOutputs):
        loss_tensor = torch.tensor(
            [o.loss for o in all_outputs],
            device=all_outputs[0].loss.device,
        )
        outputs = LossOutputs(loss=loss_tensor)
        outputs = outputs.cuda().sync().cpu() if dist.is_initialized() else outputs.cpu()
        targets = type(all_targets[0]).union(*all_targets)
    else:
        outputs = type(all_outputs[0]).union(*all_outputs)
        targets = type(all_targets[0]).union(*all_targets)
        if dist.is_initialized():
            outputs = outputs.cuda().sync().cpu()
            targets = targets.cuda().sync().cpu()
        else:
            outputs = outputs.cpu()
            targets = targets.cpu()

    model.train()
    return GenericOutputs(outputs=outputs, targets=targets)


def run_monitor(outputs, targets, monitor_fns):
    """Compute and log every score function in *monitor_fns*."""
    for fn in monitor_fns:
        score = fn(outputs=outputs, targets=targets)
        name = type(fn).__name__
        logging.info("%s: %s", name, score)
        if wandb.is_available():
            wandb.log({f"val/{name}": score})


def save_snapshot(
    model,
    ckpt_dir,
    data_loader,
    score_fn,
    monitor_fns,
    optim,
    scheduler,
    save_checkpoint="default",
    ema_model=None,
    best_score=-np.inf,
    info_path=None,
    local_rank=-1,
    **info_kwargs,
):
    """Evaluate the model, conditionally save checkpoints, and return the new best score."""
    os.makedirs(ckpt_dir, exist_ok=True)

    eval_model = ema_model if ema_model is not None else model
    snapshot_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    results = run_inference(eval_model, data_loader)

    if local_rank not in [-1, 0]:
        return best_score

    new_score = score_fn(outputs=results.outputs, targets=results.targets)
    run_monitor(results.outputs, results.targets, monitor_fns)

    if save_checkpoint in ("all", "default", "best") and new_score > best_score:
        best_score = new_score
        if model:
            model.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_model.bin")
        if ema_model:
            ema_model.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_ema_model.bin")
        if optim:
            optim.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_optim.bin")
        if scheduler:
            scheduler.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_scheduler.bin")

    if save_checkpoint in ("all", "default", "latest"):
        if model:
            model.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_model_latest.bin")
        if ema_model:
            ema_model.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_ema_model_latest.bin")
            info_kwargs["num_ema_steps"] = ema_model.num_steps
        if optim:
            optim.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_optim_latest.bin")
        if scheduler:
            scheduler.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_scheduler_latest.bin")

    if save_checkpoint in ("all", "every"):
        if model:
            model.save_checkpoint(
                ckpt_dir=ckpt_dir,
                weight_name=f"pytorch_model_{snapshot_time}.bin",
            )
        if ema_model:
            ema_model.save_checkpoint(
                ckpt_dir=ckpt_dir,
                weight_name=f"pytorch_ema_model_{snapshot_time}.bin",
            )

    if info_path is not None:
        with open(info_path, "w") as f:
            json.dump({"best_score": best_score, **info_kwargs}, f)

    return best_score


@register_task("core/task/supervised")
class SupervisedTask:
    """Supervised learning task supporting single-GPU, multi-GPU (DDP), and AMP training."""

    def __init__(
        self,
        configure,
        model,
        datasets,
        local_rank: int = -1,
        seed: int = 1123,
        cpu_offload: bool = False,
    ):
        set_seed(seed)

        self.n_gpu = 1 if torch.cuda.is_available() else 0
        if dist.is_initialized():
            self.n_gpu = dist.get_world_size()

        self.config = configure
        self.model = model
        self.datasets = datasets
        self.local_rank = local_rank
        self.best_score = -np.inf

        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)

        if torch.cuda.is_available() and not cpu_offload:
            self.model = self.model.cuda()

    @classmethod
    @add_default_section_for_init("core/task/supervised")
    def from_core_configure(cls, config, **kwargs):
        try:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        except Exception:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/supervised")
        model_name = config.getoption("model", None)
        dataset_name = config.getoption("dataset", None)

        model = (
            init_registered_module(model_name, config, registered_model)
            if model_name is not None
            else None
        )
        dataset = (
            init_registered_module(dataset_name, config, registered_dataset)
            if dataset_name is not None
            else None
        )

        return dict(
            configure=config,
            model=model,
            datasets=dataset,
            local_rank=config.getdefault("core/cli", "local_rank", get_local_rank()),
            cpu_offload=config.getoption("cpu_offload", False),
        )

    @add_default_section_for_function("core/task/supervised")
    def train(
        self,
        optim: str,
        loss_fn: Optional[str] = None,
        score_fn: Optional[str] = None,
        monitor_fns: Optional[Union[str, List[str]]] = None,
        scheduler: Optional[str] = None,
        from_ckpt_dir: str = "./from_ckpt",
        to_ckpt_dir: str = "./to_ckpt",
        train_batch_size: int = 128,
        dev_batch_size: int = 128,
        pin_memory: bool = True,
        num_workers: int = 4,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_checkpoint: str = "default",
        log_freq: int = 100,
        ckpt_freq: int = 10000,
        grad_acc_step: int = 1,
        max_grad_norm: float = 1.0,
        num_training_samples: int = 1_000_000_000,
        epochs: int = 5,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        ema_tau: int = 2000,
        use_amp: bool = True,
    ):
        if self.local_rank in (-1, 0):
            os.makedirs(to_ckpt_dir, exist_ok=True)

        if loss_fn is not None:
            loss_fn = init_registered_module(loss_fn, self.config, registered_loss)
        if score_fn is not None:
            score_fn = init_registered_module(score_fn, self.config, registered_score)

        resolved_monitor_fns = _build_monitor_fns(monitor_fns, self.config)

        if optim is not None and self.model is not None:
            optim = init_registered_module(
                optim,
                self.config,
                registered_optim,
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
            )

        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)
            optim.from_checkpoint(from_ckpt_dir, weight_name="pytorch_optim.bin")

        if os.path.exists(to_ckpt_dir):
            self.model.from_checkpoint(to_ckpt_dir, weight_name="pytorch_model_latest.bin")
            optim.from_checkpoint(to_ckpt_dir, weight_name="pytorch_optim_latest.bin")

        info_path = os.path.join(to_ckpt_dir, "info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)

        global_epoch = info.get("global_epoch", 0)
        global_step = info.get("global_step", 0)
        self.best_score = info.get("best_score", self.best_score)
        logging.info("best score so far: %s", self.best_score)

        self.ema_model = None
        if use_ema:
            self.ema_model = ExponentialMovingAverage(
                self.model,
                decay=ema_decay,
                tau=ema_tau,
                num_steps=info.get("num_ema_steps", 0),
            )
            if os.path.exists(from_ckpt_dir):
                self.ema_model.from_checkpoint(from_ckpt_dir, weight_name="pytorch_ema_model.bin")
            if os.path.exists(to_ckpt_dir):
                self.ema_model.from_checkpoint(
                    to_ckpt_dir, weight_name="pytorch_ema_model_latest.bin"
                )

        for name, param in self.model.named_parameters():
            logging.debug(
                "%s: trainable=%s dtype=%s shape=%s device=%s",
                name, param.requires_grad, param.dtype, param.shape, param.device,
            )

        global_rank = -1
        if self.n_gpu > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            global_rank = dist.get_rank()

        def get_base_model():
            return self.model.module if self.n_gpu > 1 else self.model

        train_sampler_cls = DistributedSkipSampler if self.n_gpu > 1 else RandomSkipSampler
        dev_sampler_cls = DistributedSampler if self.n_gpu > 1 else SequentialSampler

        dataset_train = self.datasets.get("train")
        dataset_dev = self.datasets.get("dev")

        iter_train = DataLoader(
            dataset_train,
            sampler=train_sampler_cls(dataset_train) if not isinstance(dataset_train, Iterable) else None,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler_cls(dataset_dev) if not isinstance(dataset_dev, Iterable) else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        if scheduler is not None:
            n_samples = (
                len(dataset_train) if not isinstance(dataset_train, Iterable)
                else num_training_samples
            )
            num_training_steps = int(
                epochs * n_samples // train_batch_size // max(1, self.n_gpu) // grad_acc_step
            )
            scheduler = init_registered_module(
                scheduler,
                self.config,
                registered_scheduler,
                optimizer=optim,
                num_training_steps=num_training_steps,
            )

        if scheduler is not None and os.path.exists(to_ckpt_dir):
            scheduler.from_checkpoint(to_ckpt_dir, weight_name="pytorch_scheduler_latest.bin")

        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        snapshot_kwargs = dict(
            save_checkpoint=save_checkpoint,
            info_path=info_path,
            local_rank=self.local_rank,
        )

        def take_snapshot(epoch, step):
            self.best_score = save_snapshot(
                get_base_model(),
                to_ckpt_dir,
                iter_dev,
                score_fn,
                resolved_monitor_fns,
                optim=optim if save_optimizer else None,
                scheduler=scheduler if save_scheduler else None,
                ema_model=self.ema_model if use_ema else None,
                best_score=self.best_score,
                global_epoch=epoch,
                global_step=step,
                **snapshot_kwargs,
            )

        def optimizer_step():
            if use_amp:
                scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if use_amp:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad()
            if use_ema and self.ema_model is not None:
                self.ema_model.step(get_base_model())

        log_loss = 0.0
        dev_epoch = 0

        for epoch in range(epochs):
            torch.cuda.empty_cache()
            if epoch < global_epoch:
                continue

            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            if hasattr(dataset_train, "set_skip_step"):
                dataset_train.set_skip_step(global_step * train_batch_size)
            if hasattr(iter_train.sampler, "set_epoch"):
                iter_train.sampler.set_epoch(epoch)
            if hasattr(iter_train.sampler, "set_skip_step"):
                iter_train.sampler.set_skip_step(global_step * train_batch_size)

            self.model.train()
            pending_grad = False

            for step, (inputs, targets) in enumerate(iter_train):
                step += global_step
                pending_grad = True

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                with torch.autocast(device_type=device_type, enabled=use_amp):
                    outputs = self.model(**inputs.dict())
                    if isinstance(outputs, LossOutputs):
                        loss = outputs.loss / grad_acc_step
                    else:
                        loss = loss_fn(outputs=outputs, targets=targets) / grad_acc_step

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                log_loss += loss.item() * grad_acc_step

                if (step + 1) % grad_acc_step == 0:
                    pending_grad = False
                    optimizer_step()

                if (step + 1) % log_freq == 0 and global_rank in (-1, 0):
                    avg_loss = log_loss / log_freq
                    logging.info("epoch %d step %d: loss=%.6f", epoch, step, avg_loss)
                    if wandb.is_available():
                        wandb.log({"epoch": epoch, "step": step, "train/loss": avg_loss})
                    log_loss = 0.0

                if (step + 1) % ckpt_freq == 0:
                    if hasattr(dataset_dev, "set_epoch"):
                        dataset_dev.set_epoch(dev_epoch)
                    if hasattr(iter_dev.sampler, "set_epoch"):
                        iter_dev.sampler.set_epoch(dev_epoch)
                    dev_epoch += 1
                    take_snapshot(epoch, step + 1)

            if pending_grad:
                optimizer_step()

            log_loss = 0.0

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)
            if hasattr(iter_dev.sampler, "set_epoch"):
                iter_dev.sampler.set_epoch(dev_epoch)
            dev_epoch += 1

            global_step = 0
            take_snapshot(epoch + 1, 0)

    @torch.no_grad()
    @add_default_section_for_function("core/task/supervised")
    def eval(
        self,
        monitor_fns: Union[str, List[str]],
        from_ckpt_dir: str = "./from_ckpt",
        dev_batch_size: int = 128,
        pin_memory: bool = True,
        num_workers: int = 4,
    ):
        resolved_monitor_fns = _build_monitor_fns(monitor_fns, self.config)

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

        dev_sampler_cls = DistributedSampler if self.n_gpu > 1 else SequentialSampler
        dataset_dev = self.datasets.get("dev")
        iter_dev = DataLoader(
            dataset_dev,
            sampler=dev_sampler_cls(dataset_dev) if not isinstance(dataset_dev, Iterable) else None,
            batch_size=dev_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        base_model = self.model.module if self.n_gpu > 1 else self.model
        results = run_inference(base_model, iter_dev)

        if global_rank in (-1, 0):
            run_monitor(results.outputs, results.targets, resolved_monitor_fns)

    @torch.no_grad()
    @add_default_section_for_function("core/task/supervised")
    def infer(
        self,
        postprocess_fn: str,
        writer: str,
        test_batch_size: int = 128,
        pin_memory: bool = True,
        num_workers: int = 4,
        max_size: int = 10000,
        from_ckpt_dir: str = "./from_ckpt",
        output_header: Optional[List] = None,
        output_path: str = "./output.txt",
        postprocess_workers: int = 2,
    ):
        assert self.n_gpu <= 1, "Inference does not support multi-GPU"
        assert writer is not None, "A writer must be specified for inference"

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

        sampler_cls = SequentialSkipSampler if skip_step > 0 else SequentialSampler
        dataset_test = self.datasets.get("test")

        iter_test = DataLoader(
            dataset_test,
            sampler=sampler_cls(dataset_test) if not isinstance(dataset_test, Iterable) else None,
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

        iter_info = None
        if hasattr(dataset_test, "dataset"):
            dataset_info = DatasetFeature(dataset_test.dataset)
            iter_info = DataLoader(
                deepcopy(dataset_info),
                sampler=sampler_cls(dataset_info) if not isinstance(dataset_test, Iterable) else None,
                batch_size=test_batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=None,
            )
            if skip_step > 0 and hasattr(iter_info.sampler, "set_skip_step"):
                iter_info.sampler.set_skip_step(skip_step)

        self.model.eval()
        start = time.time()

        data_queue = Queue(maxsize=max_size)
        msg_queue = Queue(maxsize=max_size)

        postprocess_workers_list = [
            PostProcess(postprocess_fn, data_queue, msg_queue)
            for _ in range(postprocess_workers)
        ]
        for worker in postprocess_workers_list:
            worker.start()

        io_process = IOProcess(msg_queue, writer=writer)
        io_process.start()

        if iter_info is None:
            for step, (inputs, _) in enumerate(iter_test):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict()).cpu()
                data_queue.put((step, outputs))
        else:
            for step, ((inputs, _), row_info) in enumerate(zip(iter_test, iter_info)):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                outputs = self.model(**inputs.dict()).cpu()
                if output_header is not None:
                    row_info = {k: row_info[k] for k in output_header if k in row_info}
                    outputs.from_pandas(pd.DataFrame(row_info))
                data_queue.put((step, outputs))

        data_queue.put((-1, GENERATE_FINISHED))
        for worker in postprocess_workers_list:
            worker.join()

        msg_queue.put((-1, GENERATE_FINISHED))
        io_process.join()

        elapsed_ms = (time.time() - start) * 1000
        throughput = (len(dataset_test) - skip_step) / elapsed_ms * 1000
        logging.info("%.2f ms | %.2f samples/s", elapsed_ms, throughput)
