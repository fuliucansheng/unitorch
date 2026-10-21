"""CPU fixtures for the real assets task; no official weights or CUDA jobs."""
from contextlib import contextmanager
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from unitorch.datasets.flux3_assets import file_sha256
from unitorch.models.diffusers.modeling_flux3 import GenericFlux3Model
from unitorch.tasks import flux3


class TinyAdapter(torch.nn.Module):
    """Real trainable tensor using the unchanged native checkpoint implementation."""
    checkpoint_name = "pytorch_model.bin"
    base_fingerprint = "cpu-fixture-base"
    text_encoder_out_layers = (1, 2, 3, 4)
    flow_shift = 6.93
    checkpoint_state_dict = GenericFlux3Model.checkpoint_state_dict
    _checkpoint_metadata = GenericFlux3Model._checkpoint_metadata
    save_checkpoint = GenericFlux3Model.save_checkpoint
    from_checkpoint = GenericFlux3Model.from_checkpoint

    def __init__(self, **kwargs):
        super().__init__()
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        self.lora_config = dict(r=kwargs["lora_r"], lora_alpha=kwargs["lora_alpha"])
        self.train_calls = 0
        self.trace = lambda event: None

    def to(self, *args, **kwargs):
        return self

    def train(self, mode=True):
        if not mode:
            self.trace("snapshot")
        return super().train(mode)

    def forward(self, input_ids, pixel_values):
        if self.training:
            self.trace("forward")
            self.train_calls += 1
        return ((self.transformer(input_ids) - pixel_values) ** 2).mean() + torch.rand(()) * 0.001


class TinyProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def editing(self, prompt, source, result):
        return dict(input_ids=torch.tensor([float(prompt)]), pixel_values=torch.tensor([0.25]))


def options(root):
    """Create tiny local input fixtures, isolated from all production artifacts."""
    root.mkdir(parents=True, exist_ok=True)
    image = root / "pixels.fixture"
    image.write_bytes(b"CPU orchestration fixture")
    row = dict(source_local=str(image), result_local=str(image),
               source_asset_sha256=file_sha256(image), result_asset_sha256=file_sha256(image))
    for name, size in (("train", 12), ("val", 4)):
        (root / (name + ".jsonl")).write_text("".join(
            json.dumps({**row, "prompt": str(1 + index / 12)}) + "\n" for index in range(size)))
    (root / "validation.json").write_text("{}")
    return dict(pretrained_model_path="unused-cpu-fixture", train_captions=str(root / "train.jsonl"),
                val_captions=str(root / "val.jsonl"), data_validation=str(root / "validation.json"),
                output_root=str(root / "run"), epochs=2, grad_acc_step=1, save_every=1, pilot_steps=2)


def patch_runtime(monkeypatch, trace=None):
    """Replace hardware/data admission only; keep task, AdamW, I/O and DDP real."""
    from unitorch.models.diffusers import processing_flux3
    from unitorch.models.peft.diffusers import modeling_flux3

    models = []
    def make_model(**kwargs):
        model = TinyAdapter(**kwargs)
        if trace is not None:
            model.trace = trace
        models.append(model)
        return model

    monkeypatch.setattr(modeling_flux3, "Flux3LoraForImageEditingGeneration", make_model)
    monkeypatch.setattr(processing_flux3, "Flux3Processor", TinyProcessor)
    monkeypatch.setattr(flux3, "check_admission", lambda validation, train, val, *args: dict(
        data_validation_sha256=file_sha256(validation), train_sha256=file_sha256(train),
        val_sha256=file_sha256(val)))
    monkeypatch.setattr(flux3, "check_resources", lambda *args, **kwargs: None)
    cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None,
                           get_rng_state=lambda device: torch.get_rng_state(),
                           set_rng_state=lambda state, device: None,
                           get_device_properties=lambda device: SimpleNamespace(total_memory=10**12),
                           max_memory_allocated=lambda: 0)
    class TorchCPU:
        def __getattr__(self, name):
            return getattr(torch, name)
    cpu = TorchCPU()
    cpu.cuda = cuda
    cpu.random = SimpleNamespace(fork_rng=lambda devices: torch.random.fork_rng(devices=[]))
    cpu.zeros = lambda *args, **kwargs: torch.zeros(*args, **{**kwargs, "device": "cpu"})
    # New boundary decisions use tensors on the task's local device too.
    cpu.tensor = lambda *args, **kwargs: torch.tensor(*args, **{**kwargs, "device": "cpu"})
    monkeypatch.setattr(flux3, "torch", cpu)
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], int):
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(flux3, "DistributedDataParallel", lambda model, **kwargs:
                        DistributedDataParallel(model, broadcast_buffers=kwargs["broadcast_buffers"]))
    original_init = dist.init_process_group
    monkeypatch.setattr(dist, "init_process_group", lambda backend:
                        original_init("gloo", timeout=timedelta(seconds=12)))
    return models


@contextmanager
def crash_at_publish(monkeypatch, root, point):
    """Interrupt actual filesystem mutations of optimizer update two."""
    replace, rmtree = flux3.os.replace, flux3.shutil.rmtree
    fired = []
    def crash():
        fired.append(point)
        raise RuntimeError("injected publication crash: " + point)

    def replacing(source, destination):
        source, destination = Path(source), Path(destination)
        old = source == root / "latest" and destination == root / "previous-latest"
        new = source.name.startswith("snapshot-") and destination == root / "latest"
        # The first publication is the durable fallback for the second.
        stage = source if new else next(iter(root.glob("snapshot-*")), root / "absent")
        meta = stage / "trainer.json"
        second = meta.exists() and json.loads(meta.read_text())["step"] == 2
        event = "old_rename" if old else "stage_rename" if new else None
        if second and point == "before_" + str(event):
            crash()
        result = replace(source, destination)
        if second and point == "after_" + str(event):
            crash()
        return result

    def removing(path, *args, **kwargs):
        latest = root / "latest/trainer.json"
        second = latest.exists() and json.loads(latest.read_text())["step"] == 2
        relevant = Path(path) == root / "previous-latest" and second
        if relevant and point == "before_backup_remove":
            crash()
        result = rmtree(path, *args, **kwargs)
        if relevant and point == "after_backup_remove":
            crash()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(flux3.os, "replace", replacing)
        patch.setattr(flux3.shutil, "rmtree", removing)
        yield fired
