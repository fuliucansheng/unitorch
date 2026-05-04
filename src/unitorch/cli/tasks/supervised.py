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
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Queue
from unitorch import set_seed
from unitorch.models import ExponentialMovingAverage, GenericOutputs
from unitorch.utils import get_local_rank
from unitorch.utils import (
    DistributedSkipSampler,
    RandomSkipSampler,
    SequentialSkipSampler,
    PostProcess,
    IOProcess,
    GENERATE_FINISHED,
)
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
    config_defaults_init,
    config_defaults_method,
)
from unitorch.cli.models import (
    ModelInputs,
    ModelTargets,
    LossOutputs,
    TensorMixInputs,
    TensorMixTargets,
)
import unitorch.cli.wandb as wandb


class DatasetFeature(Dataset):
    """Wraps a raw dataset to expose each row as a dict of JSON-serialisable strings."""

    def __init__(self, dataset):
        self.dataset = dataset

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

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    """Collate a list of (inputs, targets) pairs into stacked tensors.

    Supports both single ``ModelInputs``/``ModelTargets`` objects and lists of
    them (packed as ``TensorMixInputs``/``TensorMixTargets``).
    """
    raw_inputs, raw_targets = zip(*batch)

    if isinstance(raw_inputs[0], ModelInputs):
        inputs = type(raw_inputs[0]).stack(*raw_inputs)
    else:
        inputs = TensorMixInputs()
        for group in zip(*raw_inputs):
            inputs.add(type(group[0]).stack(*group))

    if isinstance(raw_targets[0], ModelTargets):
        targets = type(raw_targets[0]).stack(*raw_targets)
    else:
        targets = TensorMixTargets()
        for group in zip(*raw_targets):
            targets.add(type(group[0]).stack(*group))

    return inputs, targets


@torch.no_grad()
def infer(model, data_loader):
    """Run a full pass over *data_loader* and collect outputs and targets.

    Returns a ``GenericOutputs`` with ``.outputs`` and ``.targets`` fields,
    already reduced across distributed ranks when applicable.
    """
    model.eval()
    all_outputs, all_targets = [], []

    for inputs, targets in data_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(**inputs.dict())
        all_outputs.append(outputs)
        all_targets.append(targets)

    if isinstance(all_outputs[0], LossOutputs):
        # Aggregate scalar losses into a single tensor
        combined = LossOutputs(
            loss=torch.tensor(
                [o.loss for o in all_outputs], device=all_outputs[0].loss.device
            )
        )
        combined = combined.cuda().sync().cpu() if dist.is_initialized() else combined.cpu()
        return GenericOutputs(outputs=combined, targets=None)

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


def monitor(outputs, targets, monitor_fns):
    """Evaluate *outputs* against *targets* with each function in *monitor_fns* and log results."""
    if not monitor_fns:
        return
    for fn in monitor_fns:
        score = fn(outputs=outputs, targets=targets)
        name = type(fn).__name__
        logging.info("%s: %.6f", name, score)
        if wandb.is_available():
            wandb.log({f"val/{name}": score})


def save_snapshot(
    model,
    ckpt_dir,
    iter_dev,
    score_fn,
    monitor_fns,
    optim=None,            # optimizer instance, saved when ``save_optimizer`` is True
    scheduler=None,        # LR scheduler instance, saved when ``save_scheduler`` is True
    save_checkpoint="default",  # one of "default" | "best" | "latest" | "every" | "all"
    ema_model=None,        # EMA shadow model; replaces ``model`` for scoring when provided
    best_score=-np.inf,    # best validation score seen so far
    info_path=None,        # path to ``info.json`` for persisting training state
    local_rank=-1,         # only rank 0 (or -1 for single-GPU) writes checkpoints
    **kwargs,              # extra fields forwarded to ``info.json`` (e.g. global_epoch)
):
    """Evaluate, update best score, and save checkpoints according to *save_checkpoint* policy.

    Returns the (possibly updated) *best_score*.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    eval_model = ema_model if ema_model is not None else model
    snapshot_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    results = infer(eval_model, iter_dev)

    if local_rank not in [-1, 0]:
        return best_score

    new_score = score_fn(outputs=results.outputs, targets=results.targets)
    monitor(results.outputs, results.targets, monitor_fns)
    logging.info("val/score: %.6f  best: %.6f", new_score, best_score)

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
            kwargs["num_ema_steps"] = ema_model.num_steps
        if optim:
            optim.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_optim_latest.bin")
        if scheduler:
            scheduler.save_checkpoint(ckpt_dir=ckpt_dir, weight_name="pytorch_scheduler_latest.bin")

    if save_checkpoint in ("all", "every"):
        if model:
            model.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name=f"pytorch_model_{snapshot_time}.bin"
            )
        if ema_model:
            ema_model.save_checkpoint(
                ckpt_dir=ckpt_dir, weight_name=f"pytorch_ema_model_{snapshot_time}.bin"
            )

    if info_path is not None:
        with open(info_path, "w") as f:
            json.dump({"best_score": best_score, **kwargs}, f, indent=4)

    return best_score


@register_task("core/task/supervised")
class SupervisedTask:
    """Standard supervised learning task with optional DDP, AMP, and EMA support."""

    def __init__(
        self,
        configure,
        model,
        datasets,
        local_rank: int = -1,     # GPU index for distributed training; -1 for single-GPU
        seed: int = 1123,          # global random seed for reproducibility
        cpu_offload: bool = False, # keep model on CPU (e.g. for CPU-only environments)
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
    @config_defaults_init("core/task/supervised")
    def from_config(cls, config, **kwargs):
        try:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        except Exception:
            logging.info("PyTorch is not in distributed mode")

        config.set_default_section("core/task/supervised")

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
            cpu_offload=config.getoption("cpu_offload", False),
        )

    @config_defaults_method("core/task/supervised")
    def train(
        self,
        optim: str,                                     # registered optimizer name
        loss_fn: str,                                   # registered loss function name
        score_fn: str,                                  # registered scoring function name
        monitor_fns: Optional[Union[str, List[str]]] = None,  # extra metrics logged at checkpoints
        scheduler: Optional[str] = None,               # registered LR scheduler name
        from_ckpt_dir: str = "./from_ckpt",            # directory to load pretrained weights from
        to_ckpt_dir: str = "./to_ckpt",                # directory to write checkpoints to
        train_batch_size: int = 128,                   # per-GPU batch size for training
        dev_batch_size: int = 128,                     # per-GPU batch size for validation
        pin_memory: bool = True,                       # pin DataLoader memory for faster GPU transfer
        num_workers: int = 4,                          # DataLoader worker processes
        save_optimizer: bool = True,                   # include optimizer state in checkpoints
        save_scheduler: bool = True,                   # include scheduler state in checkpoints
        save_checkpoint: str = "default",              # checkpoint policy: default/best/latest/every/all
        log_freq: int = 100,                           # log training loss every N steps
        ckpt_freq: int = 10000,                        # save checkpoint every N steps
        grad_acc_step: int = 1,                        # gradient accumulation steps before optimizer update
        max_grad_norm: float = 1.0,                    # gradient clipping max norm
        num_training_samples: int = 1_000_000_000,     # fallback total samples for iterable datasets
        epochs: int = 5,                               # total training epochs
        use_ema: bool = False,                         # maintain an EMA shadow model for evaluation
        ema_decay: float = 0.9999,                     # EMA decay factor
        ema_tau: int = 2000,                           # EMA warm-up steps
        use_amp: bool = True,                          # enable automatic mixed precision (FP16)
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

        if optim is not None and self.model is not None:
            optim = init_registered_module(
                optim,
                self.config,
                registered_optim,
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
            )

        # Load pretrained weights, then resume from latest checkpoint if available
        if os.path.exists(from_ckpt_dir):
            self.model.from_checkpoint(from_ckpt_dir)
            optim.from_checkpoint(from_ckpt_dir, weight_name="pytorch_optim.bin")
        if os.path.exists(to_ckpt_dir):
            self.model.from_checkpoint(to_ckpt_dir, weight_name="pytorch_model_latest.bin")
            optim.from_checkpoint(to_ckpt_dir, weight_name="pytorch_optim_latest.bin")

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
                self.ema_model.from_checkpoint(to_ckpt_dir, weight_name="pytorch_ema_model_latest.bin")

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

        if scheduler is not None:
            if not isinstance(dataset_train, Iterable):
                num_training_steps = int(
                    epochs * len(dataset_train) // train_batch_size
                    // max(1, self.n_gpu) // grad_acc_step
                )
            else:
                num_training_steps = int(
                    epochs * num_training_samples // train_batch_size
                    // max(1, self.n_gpu) // grad_acc_step
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

        # AMP gradient scaler; only created when use_amp=True
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        def _optimizer_step():
            """Unscale gradients (if AMP), clip, then step the optimizer."""
            if scaler is not None:
                scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad()
            if use_ema and self.ema_model is not None:
                base = self.model.module if self.n_gpu > 1 else self.model
                self.ema_model.step(base)

        def _snapshot(epoch, step):
            return save_snapshot(
                self.model.module if self.n_gpu > 1 else self.model,
                to_ckpt_dir,
                iter_dev,
                score_fn,
                monitor_fns,
                optim=optim if save_optimizer else None,
                scheduler=scheduler if save_scheduler else None,
                save_checkpoint=save_checkpoint,
                ema_model=self.ema_model if use_ema else None,
                best_score=self.best_score,
                info_path=info_path,
                local_rank=self.local_rank,
                global_epoch=epoch,
                global_step=step,
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

                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=use_amp,
                ):
                    outputs = self.model(**inputs.dict())
                    loss = (
                        outputs.loss if isinstance(outputs, LossOutputs)
                        else loss_fn(outputs=outputs, targets=targets)
                    ) / grad_acc_step

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                log_loss += loss.item() * grad_acc_step

                if (step + 1) % grad_acc_step == 0:
                    is_update_step = True
                    _optimizer_step()

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

            # Flush any remaining accumulated gradients at epoch end
            if not is_update_step:
                _optimizer_step()

            log_loss = 0.0

            if hasattr(dataset_dev, "set_epoch"):
                dataset_dev.set_epoch(dev_epoch)
            if hasattr(iter_dev.sampler, "set_epoch"):
                iter_dev.sampler.set_epoch(dev_epoch)
            dev_epoch += 1

            global_step = 0
            self.best_score = _snapshot(e + 1, 0)

    @torch.no_grad()
    @config_defaults_method("core/task/supervised")
    def eval(
        self,
        monitor_fns: Union[str, List[str]],     # list of registered scoring function names
        from_ckpt_dir: str = "./from_ckpt",     # directory to load model weights from
        dev_batch_size: int = 128,              # per-GPU batch size for evaluation
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

        results = infer(self.model.module if self.n_gpu > 1 else self.model, iter_dev)
        if global_rank in [-1, 0]:
            monitor(outputs=results.outputs, targets=results.targets, monitor_fns=monitor_fns)

    @torch.no_grad()
    @config_defaults_method("core/task/supervised")
    def infer(
        self,
        postprocess_fn: str,                     # registered postprocessing function name
        writer: str,                             # registered writer name for output serialisation
        test_batch_size: int = 128,             # per-GPU batch size for inference
        pin_memory: bool = True,
        num_workers: int = 4,
        max_size: int = 10000,                  # maximum queue depth for async postprocessing
        from_ckpt_dir: str = "./from_ckpt",    # directory to load model weights from
        output_header: Optional[List] = None,  # column names to copy from raw dataset into output
        output_path: str = "./output.txt",     # file path for inference results
        postprocess_workers: int = 2,          # number of parallel postprocessing workers
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

        # Build a parallel loader for raw dataset metadata (images, text) when available
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
