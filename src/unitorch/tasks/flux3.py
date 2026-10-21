# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Resumable, bounded FLUX.3 assets LoRA runs with explicit admission gates."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from filelock import FileLock

from unitorch.datasets.flux3_assets import file_sha256, _publish, _rows


def _optimizer_boundary(step, cursor, epoch_size, save_every, pilot_steps,
                        stop_file, device, world):
    """Agree on both branches at every completed optimizer update."""
    stopping = bool((pilot_steps is not None and step >= pilot_steps)
                    or (stop_file and Path(stop_file).exists()))
    saving = stopping or step % save_every == 0 or cursor == epoch_size
    flags = torch.tensor([stopping, saving], dtype=torch.int32, device=device)
    if world > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return bool(flags[0].item()), bool(flags[1].item())


def _validate_training_checkpoint(path, identity, model, optim, epoch_size):
    """Validate native model, AdamW, cursor and RNG together, without mutation."""
    from unitorch.models.diffusers.checkpoint_flux3 import validate_state

    path = Path(path)
    try:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("checkpoint must be a local directory")
        metadata = json.loads((path / "trainer.json").read_text())
        if metadata["sha256"] != file_sha256(path / "trainer.pt"):
            raise ValueError("trainer checkpoint SHA256 mismatch")
        state = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity or state["base_fingerprint"] != model.base_fingerprint:
            raise ValueError("checkpoint base/run identity mismatch")
        weight_path = path / model.checkpoint_name
        sidecar = json.loads(Path(str(weight_path) + ".json").read_text())
        if any(sidecar.get(k) != v for k, v in model._checkpoint_metadata().items()):
            raise ValueError("model checkpoint base/config mismatch")
        if sidecar["sha256"] != file_sha256(weight_path):
            raise ValueError("model checkpoint SHA256 mismatch")
        weights = torch.load(weight_path, map_location="cpu", weights_only=True)
        expected = model.checkpoint_state_dict()
        validate_state(weights, expected, "model checkpoint")
        if any(weights[k].dtype != expected[k].dtype for k in weights):
            raise ValueError("model checkpoint dtype mismatch")
        epoch, cursor, step = state["epoch"], state["next_index"], state["step"]
        if any(type(v) is not int for v in (epoch, cursor, step)) or epoch < 0 or step < 1:
            raise ValueError("checkpoint optimizer step/epoch invalid")
        accumulation = identity["grad_acc_step"]
        if (not 0 < cursor <= epoch_size or (cursor % accumulation and cursor != epoch_size)
                or step != epoch * math.ceil(epoch_size / accumulation) + math.ceil(cursor / accumulation)):
            raise ValueError("checkpoint cursor does not match optimizer step")
        if (metadata["step"] != step or not math.isfinite(metadata["val_loss"])
                or not math.isfinite(state["best_loss"]) or state["best_loss"] > metadata["val_loss"]):
            raise ValueError("checkpoint step/loss metadata mismatch")
        saved_optim = state["optimizer"]
        if saved_optim["param_groups"] != optim.state_dict()["param_groups"]:
            raise ValueError("checkpoint optimizer parameter groups/config mismatch")
        parameters = {key: parameter for saved, live in zip(saved_optim["param_groups"], optim.param_groups)
                      for key, parameter in zip(saved["params"], live["params"])}
        if set(saved_optim["state"]) != set(parameters):
            raise ValueError("checkpoint optimizer state coverage mismatch")
        for key, parameter in parameters.items():
            slot = saved_optim["state"][key]
            if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("checkpoint AdamW optimizer state keys mismatch")
            counter = slot["step"]
            if not isinstance(counter, torch.Tensor) or counter.numel() != 1 or counter.item() != step:
                raise ValueError("checkpoint optimizer step mismatch")
            for name in ("exp_avg", "exp_avg_sq"):
                value = slot[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError("checkpoint optimizer tensor shape/dtype/finite mismatch")
            if (slot["exp_avg_sq"] < 0).any():
                raise ValueError("checkpoint optimizer variance is negative")
        if len(state["rng"]) != identity["world_size"]:
            raise ValueError("checkpoint RNG rank coverage mismatch")
        for rng in state["rng"]:
            if set(rng) != {"python", "numpy", "torch", "cuda"}:
                raise ValueError("checkpoint RNG keys mismatch")
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator().set_state(rng["torch"])
            if (not isinstance(rng["cuda"], torch.Tensor) or rng["cuda"].dtype != torch.uint8
                    or rng["cuda"].ndim != 1 or not rng["cuda"].numel()):
                raise ValueError("checkpoint CUDA RNG state invalid")
            if (torch.cuda.is_available() and rng["cuda"].numel()
                    != torch.cuda.get_rng_state(int(os.environ.get("LOCAL_RANK", "0"))).numel()):
                raise ValueError("checkpoint CUDA RNG state size mismatch")
        return state
    except Exception as error:
        raise ValueError(f"Invalid FLUX.3 checkpoint {path}: {error}") from error


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_latest(root, stage):
    """Keep the native layout; an interrupted rename is recovered under the run lock."""
    previous = root / "previous-latest"
    if previous.exists():
        raise ValueError("Unrecovered previous-latest checkpoint; preserving evidence")
    for path in stage.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    _sync_directory(stage)
    if (root / "latest").exists():
        os.replace(root / "latest", previous)
        _sync_directory(root)
    os.replace(stage, root / "latest")
    _sync_directory(root)
    if previous.exists():
        shutil.rmtree(previous)
        _sync_directory(root)


def _publish_best(root):
    stage = Path(tempfile.mkdtemp(dir=root, prefix="best-"))
    shutil.copytree(root / "latest", stage, dirs_exist_ok=True)
    if (root / "best-loss").exists():
        shutil.rmtree(root / "best-loss")
    os.replace(stage, root / "best-loss")
    _sync_directory(root)


def _recover_latest(root, identity, model, optim, epoch_size, lock):
    """Recover only this run's latest/previous-latest, never best or arbitrary stages.

    A valid published latest wins. In the rename gap, previous-latest is the
    committed fallback. Reject invalid/stale candidates before any mutation.
    Uncommitted stages are discarded only after their complete validation;
    incomplete/corrupt stages remain as evidence, never as resume candidates.
    """
    if not lock.is_locked or Path(lock.lock_file) != root / ".run.lock":
        raise RuntimeError("Checkpoint recovery requires the owned run lock")
    prior = json.loads((root / "run.json").read_text())
    if any(prior.get(k) != v for k, v in {**identity, "base_fingerprint": model.base_fingerprint}.items()):
        raise ValueError("Checkpoint run/base identity mismatch")
    latest, previous = root / "latest", root / "previous-latest"
    candidate = latest if latest.exists() else previous
    state = _validate_training_checkpoint(candidate, identity, model, optim, epoch_size)
    if candidate == latest and previous.exists():
        backup = _validate_training_checkpoint(previous, identity, model, optim, epoch_size)
        if backup["step"] >= state["step"]:
            raise ValueError("Checkpoint backup is not older than published latest")
    if (root / "progress.json").exists():
        progress = json.loads((root / "progress.json").read_text())
        if (progress["optimizer_step"] > state["step"]
                or (progress["optimizer_step"] == state["step"]
                    and (progress["epoch"], progress["next_index"]) != (state["epoch"], state["next_index"]))):
            raise ValueError("Checkpoint backup/latest is older than published progress or cursor differs")
    if candidate == previous:
        os.replace(previous, latest)
        _sync_directory(root)
    # A crash after latest publication can precede the corresponding best copy.
    metadata = json.loads((latest / "trainer.json").read_text())
    if metadata["val_loss"] == state["best_loss"]:
        _publish_best(root)
    for path in [previous, *root.glob("snapshot-*"), *root.glob("best-*")]:
        if path == root / "best-loss" or not path.exists():
            continue
        try:
            _validate_training_checkpoint(path, identity, model, optim, epoch_size)
        except ValueError:
            continue  # Preserve incomplete/corrupt evidence instead of deleting it.
        shutil.rmtree(path)
    _sync_directory(root)
    return state


def check_resources(output_root: str, min_disk_bytes: int = 20 * 1024**3,
                    min_ram_bytes: int = 8 * 1024**3, max_cuda_bytes=None):
    """Abort on exhausted resources; never change image resolution automatically."""
    root = Path(output_root)
    while not root.exists():
        root = root.parent
    if shutil.disk_usage(root).free < min_disk_bytes:
        raise RuntimeError("Insufficient disk reserve for FLUX.3 checkpoints")
    if psutil.virtual_memory().available < min_ram_bytes:
        raise RuntimeError("Insufficient host memory reserve for FLUX.3")
    if max_cuda_bytes and torch.cuda.is_available() and torch.cuda.memory_allocated() > max_cuda_bytes:
        raise RuntimeError("FLUX.3 CUDA memory admission limit exceeded")


def check_admission(data_validation: str, train_captions: str, val_captions: str,
                    official_proof=None, approval_file=None, pilot_steps=None):
    """Formal runs require full captions, official GPU proof and reviewed outputs."""
    validated = json.loads(Path(data_validation).read_text(encoding="utf-8"))
    if not validated.get("valid") or not validated.get("leakage_free") or validated["geometry"] != [1024, 1536]:
        raise ValueError("Complete 1024x1536 data validation is required")
    train, val = _rows(train_captions), _rows(val_captions)
    if not train or not val:
        raise ValueError("Nonempty training and validation scene captions are required")
    original_path = Path(validated["dataset_root"]) / "dataset/full.jsonl"
    if file_sha256(original_path) != validated["source_manifest_sha256"]:
        raise ValueError("Validated source manifest changed")
    original = {row["pair_id"]: row for row in _rows(original_path)}
    for rows, split in ((train, "train"), (val, "val")):
        if len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate captioned training/validation pair")
        if any(r["split"] != split or r.get("caption_provenance", {}).get("conditioning") != "source_image_and_instruction_only" for r in rows):
            raise ValueError("Caption split or source-only provenance gate failed")
        for row in rows:
            prior = original.get(row["pair_id"], {})
            fields = ("split", "group_id", "source_local", "result_local", "source_asset_sha256", "result_asset_sha256", "algo")
            if any(row.get(k) != prior.get(k) for k in fields):
                raise ValueError("Captioned pair differs from the inherited source/target/split")
    if {r["group_id"] for r in train} & {r["group_id"] for r in val}:
        raise ValueError("Captioned dataset has train/validation group leakage")
    identity = dict(train_sha256=file_sha256(train_captions), val_sha256=file_sha256(val_captions),
                    data_validation_sha256=file_sha256(data_validation))
    if pilot_steps is not None:
        if not 1 <= pilot_steps <= 1000:
            raise ValueError("A pilot must be bounded to 1..1000 optimizer updates")
        return identity
    if len(train) != 31640 or len(val) != 200:
        raise ValueError("Formal run requires every inherited 31640/200 caption")
    for path, rows in ((train_captions, train), (val_captions, val)):
        progress = json.loads(Path(path).with_suffix(".progress.json").read_text(encoding="utf-8"))
        if not progress.get("complete") or progress.get("output_sha256") != file_sha256(path):
            raise ValueError("Caption preparation is incomplete or changed")
    if official_proof is None or approval_file is None:
        raise ValueError("Formal training requires official GPU proof and parent quality-review approval")
    proof = json.loads(Path(official_proof).read_text(encoding="utf-8"))
    if not (proof.get("real_gpu_validation") and proof.get("checkpoint_roundtrip_exact") and proof.get("adapter_roundtrip_exact")):
        raise ValueError("Official train/save/resume/infer proof is incomplete")
    expected_proof = {"official_revision": "3f7fa5ee22ca1d5a5cc095ac7cc96e6a173b4c86",
                      "width": 1024, "height": 1536, "inference_steps": 50, "guidance_scale": 3.5}
    if any(proof.get(k) != v for k, v in expected_proof.items()):
        raise ValueError("Official GPU proof does not match the pinned 1K base recipe")
    approval = json.loads(Path(approval_file).read_text(encoding="utf-8"))
    expected = {**identity, "official_proof_sha256": file_sha256(official_proof), "quality_review_approved": True}
    if any(approval.get(k) != v for k, v in expected.items()):
        raise ValueError("Approval must refer to these exact captions and reviewed official proof")
    return identity


def train_assets(pretrained_model_path: str, train_captions: str, val_captions: str,
                 data_validation: str, output_root: str, epochs: int = 100,
                 grad_acc_step: int = 4, learning_rate: float = 1e-5,
                 lora_r: int = 16, lora_alpha: int = 32, save_every: int = 100,
                 pilot_steps=None, official_proof=None, approval_file=None,
                 resume: bool = False, seed: int = 1123, offload_conditioners: bool = True,
                 stop_file=None):
    """Batch one per GPU, FP32 adapters, BF16 DiT/VL and official FP32 VAE.

    torchrun selects DDP/NCCL. Writes transactional latest and best-loss only;
    best-loss is a numerical monitor, never a claim of visual quality. Resume
    requires matching base, split hashes, world size, optimizer and RNG states.
    A stop-file is honored at an optimizer boundary, after a complete snapshot.
    """
    from unitorch.models.diffusers.processing_flux3 import Flux3Processor
    from unitorch.models.peft.diffusers.modeling_flux3 import Flux3LoraForImageEditingGeneration

    identity = check_admission(data_validation, train_captions, val_captions,
                               official_proof, approval_file, pilot_steps)
    if epochs < 1 or grad_acc_step < 1 or save_every < 1 or learning_rate <= 0:
        raise ValueError("Epochs, accumulation, checkpoint interval and learning rate must be positive")
    # Keep the cheap single-process pilot guard. DDP checks the recovered state
    # below, after rank zero's decision has been broadcast to every rank.
    if (resume and pilot_steps is not None and int(os.environ.get("WORLD_SIZE", "1")) == 1
            and (Path(output_root) / "latest/trainer.json").is_file()):
        saved = json.loads((Path(output_root) / "latest/trainer.json").read_text(encoding="utf-8"))
        if saved["step"] >= pilot_steps:
            raise ValueError("Requested pilot limit already reached; explicitly increase pilot_steps to extend the run")
    if not torch.cuda.is_available():
        raise RuntimeError("Official 1K assets training requires an accessible CUDA device")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / ".run.lock"))
    if rank == 0:
        lock.acquire(timeout=0)
    try:
        check_resources(str(root))
        run_identity = {**identity, "world_size": world, "seed": seed, "grad_acc_step": grad_acc_step,
                        "learning_rate": learning_rate, "lora_r": lora_r, "lora_alpha": lora_alpha,
                        "width": 1024, "height": 1536}
        if not resume and ((root / "latest").exists() or (root / "run.json").exists()):
            raise FileExistsError("Existing FLUX.3 run; choose a fresh output_root or explicit resume=True")
        if resume:
            prior = json.loads((root / "run.json").read_text(encoding="utf-8"))
            if any(prior.get(k) != v for k, v in run_identity.items()):
                raise ValueError("FLUX.3 resume identity changed")
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        model = Flux3LoraForImageEditingGeneration(
            pretrained_model_path=pretrained_model_path, lora_r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0.0, caption_dropout=0.1, gradient_checkpointing=True,
            offload_conditioners=offload_conditioners, torch_dtype="bfloat16", vae_dtype="float32",
        ).to(f"cuda:{local_rank}")
        processor = Flux3Processor(str(Path(pretrained_model_path) / "text_encoder"), image_size=(1024, 1536))
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        rows, validation = _rows(train_captions), _rows(val_captions)
        start_epoch = next_index = step = 0
        best_loss = float("inf")
        if resume:
            latest = root / "latest"
            error = None
            if rank == 0:
                try:
                    state = _recover_latest(root, run_identity, model, optim,
                                            math.ceil(len(rows) / world), lock)
                except Exception as failure:
                    error = str(failure)
            if world > 1:
                errors = [error]
                dist.broadcast_object_list(errors, src=0)
                error = errors[0]
            if error is not None:
                raise ValueError(error)
            if rank != 0:
                state = _validate_training_checkpoint(latest, run_identity, model, optim,
                                                      math.ceil(len(rows) / world))
            if pilot_steps is not None and state["step"] >= pilot_steps:
                raise ValueError("Requested pilot limit already reached; explicitly increase pilot_steps to extend the run")
            model.from_checkpoint(str(latest))
            optim.load_state_dict(state["optimizer"])
            start_epoch, next_index, step, best_loss = state["epoch"], state["next_index"], state["step"], state["best_loss"]
            rng = state["rng"][rank]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], device=local_rank)
        if rank == 0:
            _publish(root / "run.json", {**run_identity, "base_fingerprint": model.base_fingerprint,
                                         "epochs": epochs, "pilot_steps": pilot_steps, "formal": pilot_steps is None})
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
        def batch(row):
            for side in ("source", "result"):
                if file_sha256(row[side + "_local"]) != row[side + "_asset_sha256"]:
                    raise ValueError("Dataset image changed after validation")
            return {k: v.unsqueeze(0).to(local_rank) for k, v in processor.editing(row["prompt"], row["source_local"], row["result_local"]).items()}
        def snapshot(epoch, cursor):
            nonlocal best_loss
            rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(local_rank)}
            model.eval()
            metric = torch.zeros(2, device=local_rank)
            with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(seed + 100000 + rank)
                for row in validation[rank::world]:
                    metric[0] += model(**batch(row)).float()
                    metric[1] += 1
            if world > 1:
                dist.all_reduce(metric)
            val_loss = (metric[0] / metric[1]).item()
            improved = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states[0] = rng
            if rank == 0:
                stage = Path(tempfile.mkdtemp(dir=root, prefix="snapshot-"))
                model.save_checkpoint(str(stage))
                torch.save({"identity": run_identity, "base_fingerprint": model.base_fingerprint,
                            "optimizer": optim.state_dict(), "epoch": epoch, "next_index": cursor,
                            "step": step, "best_loss": best_loss, "rng": states}, stage / "trainer.pt")
                _publish(stage / "trainer.json", {"sha256": file_sha256(stage / "trainer.pt"), "step": step, "val_loss": val_loss})
                _publish_latest(root, stage)
                if improved:
                    _publish_best(root)
                _publish(root / "progress.json", {"epoch": epoch, "next_index": cursor, "optimizer_step": step,
                                                   "val_loss": val_loss, "best_loss": best_loss, "quality_approved": False})
            if world > 1:
                dist.barrier()
            model.train()
        ddp.train()
        for epoch in range(start_epoch, epochs):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            per_rank = math.ceil(len(order) / world)
            order += order[:per_rank * world - len(order)]
            indices = order[rank::world]
            cursor = next_index if epoch == start_epoch else 0
            while cursor < len(indices):
                check_resources(str(root), max_cuda_bytes=int(torch.cuda.get_device_properties(local_rank).total_memory * 0.96))
                count = min(grad_acc_step, len(indices) - cursor)
                optim.zero_grad(set_to_none=True)
                losses = []
                for offset in range(count):
                    context = ddp.no_sync() if world > 1 and offset < count - 1 else nullcontext()
                    with context:
                        loss = ddp(**batch(rows[indices[cursor + offset]]))
                        losses.append(float(loss.detach()))
                        (loss / count).backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optim.step()
                cursor += count
                step += 1
                if rank == 0:
                    print(json.dumps({"epoch": epoch, "optimizer_step": step, "loss": sum(losses)/count,
                                      "cuda_peak_bytes": torch.cuda.max_memory_allocated()}), flush=True)
                stopping, saving = _optimizer_boundary(step, cursor, len(indices), save_every,
                                                       pilot_steps, stop_file, local_rank, world)
                if saving:
                    snapshot(epoch, cursor)
                if stopping:
                    return {"optimizer_steps": step, "stopped_at_checkpoint": True, "formal": pilot_steps is None}
        return {"optimizer_steps": step, "epochs_completed": epochs}
    finally:
        if rank == 0:
            lock.release()
        if world > 1:
            dist.destroy_process_group()
