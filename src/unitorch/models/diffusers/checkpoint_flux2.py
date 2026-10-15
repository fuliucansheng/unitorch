# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

"""Offline export of trusted native FLUX.2 ZeRO-3 training checkpoints."""

import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import psutil
import torch


def export_zero_checkpoint(
    checkpoint: str,
    output: str,
    mode: str = "e2e",
    dtype: str = "bfloat16",
    tag: Optional[str] = None,
    max_memory_bytes: Optional[int] = None,
    reserve_memory_bytes: int = 16 * 1024**3,
) -> dict:
    """Export trained components/adapters to ``output/pytorch_model.bin``.

    Run in a separate CPU process after saving a native ZeRO-3 checkpoint.
    DeepSpeed reconstructs FP32 master weights lazily from mmap optimizer
    shards; only selected tensors are materialized, one at a time. The final
    state still occupies RAM. Preflight includes output, the largest FP32
    gather/conversion, model-state files and memory headroom. ``max_memory_bytes``
    optionally lowers the allocation budget. Existing exports are never replaced.
    Native checkpoints contain Python metadata: only load trusted checkpoints.
    """
    if mode not in ("e2e", "lora"):
        raise ValueError("mode must be e2e or lora")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if reserve_memory_bytes < 0 or (max_memory_bytes is not None and max_memory_bytes <= 0):
        raise ValueError("memory limits must be positive and reserve nonnegative")
    checkpoint, output = Path(checkpoint), Path(output)
    destination = output / "pytorch_model.bin"
    if destination.exists():
        raise FileExistsError(destination)
    if tag is None:
        tag = (checkpoint / "latest").read_text().strip()
    directory = checkpoint / tag
    optimizer_files = sorted(directory.glob("*_optim_states.pt"))
    model_files = sorted(directory.glob("*_model_states.pt"))
    if not optimizer_files or not model_files:
        raise ValueError(f"Incomplete native ZeRO checkpoint: {directory}")
    # ZeRO-2 consolidation is eager even with lazy_mode=True. Refuse it before
    # calling DeepSpeed, so this exporter's memory preflight remains meaningful.
    metadata = torch.load(optimizer_files[0], map_location="cpu", mmap=True, weights_only=False)
    if metadata["optimizer_state_dict"]["zero_stage"] != 3:
        raise ValueError("FLUX.2 lazy export requires a ZeRO-3 checkpoint")
    del metadata
    model_bytes = sum(path.stat().st_size for path in model_files)

    def check_memory(required):
        budget = psutil.virtual_memory().available - reserve_memory_bytes
        if max_memory_bytes is not None:
            budget = min(budget, max_memory_bytes)
        if required > budget:
            raise RuntimeError(
                f"Insufficient memory for FLUX.2 export: need {required} bytes "
                f"plus {reserve_memory_bytes} bytes reserve; budget {budget} bytes"
            )

    # DeepSpeed mmap-loads optimizer shards, but reads model metadata eagerly.
    check_memory(model_bytes)
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    lazy = get_fp32_state_dict_from_zero_checkpoint(
        str(checkpoint), tag=tag, exclude_frozen_parameters=True, lazy_mode=True,
    )
    lazy = {(key[6:] if key.startswith("model.") else key): value for key, value in lazy.items()}
    # Buffers survive exclude_frozen_parameters. Include encoder buffers only
    # when that component has trainable parameters in the native checkpoint.
    prefixes = tuple(prefix for prefix in ("transformer.", "text.", "vae.encoder.", "vae.quant_conv.")
                     if any(key.startswith(prefix) and not isinstance(value, torch.Tensor)
                            for key, value in lazy.items()))
    selected = {
        key: value for key, value in lazy.items()
        if (".lora_" in key if mode == "lora" else key.startswith(prefixes))
    }
    if not selected or (mode == "e2e" and any(".lora_" in key for key in selected)):
        raise ValueError(f"Checkpoint contains no matching FLUX.2 {mode} trained components")
    target_dtype = getattr(torch, dtype)
    element_size = torch.empty((), dtype=target_dtype).element_size()
    output_bytes = sum(math.prod(value.shape) * (
        value.element_size() if isinstance(value, torch.Tensor) and not value.is_floating_point() else element_size
    ) for value in selected.values())
    largest_fp32 = max(math.prod(value.shape) * 4 for value in selected.values())
    # Two largest-tensor temporaries cover gather padding and dtype conversion.
    required_bytes = model_bytes + output_bytes + 2 * largest_fp32
    check_memory(required_bytes)
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < output_bytes + 1024**3:
        raise RuntimeError("Insufficient disk space for FLUX.2 export and 1 GiB headroom")
    state = {}
    for key, value in selected.items():
        tensor = value.contiguous()
        state[key] = tensor.to(dtype=target_dtype) if tensor.is_floating_point() else tensor
        del tensor
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".flux2-export-", suffix=".bin", dir=output)
    os.close(fd)
    try:
        torch.save(state, temporary)
        # Atomic publication without replacing an export created concurrently.
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return {"path": str(destination), "mode": mode, "dtype": dtype,
            "tensors": len(state), "output_bytes": output_bytes,
            "estimated_memory_bytes": required_bytes}
