"""Regression tests for STOP agreement and native-checkpoint publish recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytest.importorskip("diffusers", minversion="0.41.0.dev0")
pytest.importorskip("peft")

from unitorch.utils.import_utils import is_flux3_available

if not is_flux3_available():
    pytest.skip("FLUX.3 requires Diffusers >= 0.41.0.dev0 and PEFT", allow_module_level=True)

try:
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
except ImportError as exc:
    pytest.skip(f"FLUX.3 backend symbols unavailable: {exc}", allow_module_level=True)

from flux3_reliability_utils import crash_at_publish, options, patch_runtime
from unitorch.datasets.flux3_assets import file_sha256
from unitorch.tasks import flux3


@pytest.mark.parametrize("stop_rank,save_every,pilot_steps,stop_step,expected,accumulation,cursor", [
    (0, 3, 5, 2, 2, 1, 2), (1, 3, 5, 2, 2, 1, 2), (0, 2, 5, 2, 2, 1, 2),
    (1, 2, 5, 2, 2, 1, 2), (-1, 3, 2, 2, 2, 1, 2), (-1, 2, 5, 9, 5, 1, 5),
    (1, 3, 5, 3, 1, 4, 4), (-1, 9, 3, 99, 3, 4, 4),
])
def test_stop_agreement_real_gloo(tmp_path, stop_rank, save_every, pilot_steps, stop_step,
                                 expected, accumulation, cursor):
    config = tmp_path / "options.json"
    config.write_text(json.dumps(options(tmp_path)))
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(Path(__file__).with_name("flux3_reliability_worker.py")), f"--config={config}",
        f"--output={tmp_path}", f"--stop_rank={stop_rank}", f"--save_every={save_every}",
        f"--pilot_steps={pilot_steps}", f"--stop_step={stop_step}",
        f"--grad_acc_step={accumulation}",
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [json.loads((tmp_path / f"rank-{r}.json").read_text()) for r in range(2)]
    assert ranks[0]["result"] == ranks[1]["result"]
    assert ranks[0]["result"]["optimizer_steps"] == expected
    assert ranks[0]["events"] == ranks[1]["events"]
    assert ranks[0]["events"][-2:] == ["snapshot", "return"]
    if stop_rank >= 0:
        assert ranks[stop_rank]["observations"][-1] is True
        assert not any(ranks[1 - stop_rank]["observations"])
    state = torch.load(tmp_path / "run/latest/trainer.pt", weights_only=False)
    assert state["step"] == expected and state["next_index"] == cursor
    assert len(state["rng"]) == 2


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("point", [
    "before_old_rename", "after_old_rename", "before_stage_rename",
    "after_stage_rename", "before_backup_remove", "after_backup_remove",
])
def test_publish_crash_recovers_exact_resume(tmp_path, monkeypatch, point):
    patch_runtime(monkeypatch)
    kwargs = options(tmp_path)
    root = Path(kwargs["output_root"])
    with crash_at_publish(monkeypatch, root, point) as fired:
        with pytest.raises(RuntimeError, match="injected publication crash"):
            flux3.train_assets(**kwargs)
    assert fired == [point]
    result = flux3.train_assets(**{**kwargs, "resume": True, "pilot_steps": 3})
    assert result["optimizer_steps"] == 3
    restored = torch.load(root / "latest/trainer.pt", weights_only=False)
    weights = torch.load(root / "latest/pytorch_model.bin", weights_only=True)
    reference = tmp_path / "reference"
    flux3.train_assets(**{**kwargs, "output_root": str(reference), "pilot_steps": 3})
    expected = torch.load(reference / "latest/trainer.pt", weights_only=False)
    assert restored["step"] == expected["step"] == 3
    assert restored["epoch"] == expected["epoch"] == 0
    assert restored["next_index"] == expected["next_index"] == 3
    assert restored["best_loss"] == expected["best_loss"]
    assert_tree_equal(restored["optimizer"], expected["optimizer"])
    assert_tree_equal(weights, torch.load(reference / "latest/pytorch_model.bin", weights_only=True))
    assert_tree_equal(restored["rng"][0]["torch"], expected["rng"][0]["torch"])
    best = torch.load(root / "best-loss/trainer.pt", weights_only=False)
    assert best["next_index"] == best["step"]
    assert all(int(s["step"]) == best["step"] for s in best["optimizer"]["state"].values())
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["best-loss", "latest"]


def rehash(candidate):
    for payload, sidecar in (("trainer.pt", "trainer.json"), ("pytorch_model.bin", "pytorch_model.bin.json")):
        metadata = json.loads((candidate / sidecar).read_text())
        metadata["sha256"] = file_sha256(candidate / payload)
        (candidate / sidecar).write_text(json.dumps(metadata))


@pytest.mark.parametrize("damage", [
    "trainer_sha", "model_sha", "model_nan", "model_missing", "model_extra", "model_shape",
    "model_sidecar_base", "trainer_base", "identity", "optimizer_nan", "optimizer_shape",
    "optimizer_step", "cursor", "metadata_step", "rng", "rng_cuda", "stale_backup",
])
def test_corrupt_backup_refused_and_preserved(tmp_path, monkeypatch, damage):
    patch_runtime(monkeypatch)
    kwargs = options(tmp_path)
    root = Path(kwargs["output_root"])
    with crash_at_publish(monkeypatch, root, "after_old_rename"):
        with pytest.raises(RuntimeError, match="injected publication crash"):
            flux3.train_assets(**kwargs)
    backup = root / "previous-latest"
    state = torch.load(backup / "trainer.pt", weights_only=False)
    weights = torch.load(backup / "pytorch_model.bin", weights_only=True)
    key = next(iter(weights))
    slot = next(iter(state["optimizer"]["state"].values()))
    if damage == "trainer_sha":
        (backup / "trainer.pt").write_bytes(b"broken trainer")
    elif damage == "model_sha":
        (backup / "pytorch_model.bin").write_bytes(b"broken model")
    else:
        if damage == "model_nan":
            weights[key].fill_(float("nan"))
        elif damage == "model_missing":
            weights.pop(key)
        elif damage == "model_extra":
            weights["unexpected"] = torch.ones(1)
        elif damage == "model_shape":
            weights[key] = torch.ones(7)
        elif damage == "model_sidecar_base":
            meta = json.loads((backup / "pytorch_model.bin.json").read_text())
            meta["base_fingerprint"] = "other-base"
            (backup / "pytorch_model.bin.json").write_text(json.dumps(meta))
        elif damage == "trainer_base":
            state["base_fingerprint"] = "other-base"
        elif damage == "identity":
            state["identity"]["train_sha256"] = "other-dataset"
        elif damage == "optimizer_nan":
            slot["exp_avg"].fill_(float("nan"))
        elif damage == "optimizer_shape":
            slot["exp_avg_sq"] = torch.ones(7)
        elif damage == "optimizer_step":
            slot["step"].fill_(0)
        elif damage == "cursor":
            state["next_index"] = -1
        elif damage == "metadata_step":
            meta = json.loads((backup / "trainer.json").read_text())
            meta["step"] = 9
            (backup / "trainer.json").write_text(json.dumps(meta))
        elif damage == "rng":
            state["rng"] = []
        elif damage == "rng_cuda":
            state["rng"][0]["cuda"] = torch.zeros(1, dtype=torch.uint8)
        elif damage == "stale_backup":
            progress = json.loads((root / "progress.json").read_text())
            progress["optimizer_step"] = 10
            (root / "progress.json").write_text(json.dumps(progress))
        torch.save(state, backup / "trainer.pt")
        torch.save(weights, backup / "pytorch_model.bin")
        rehash(backup)
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="checkpoint|backup|identity|optimizer|cursor|RNG"):
        flux3.train_assets(**{**kwargs, "resume": True, "pilot_steps": 3})
    assert not (root / "latest").exists()
    assert before == {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}


def test_recovery_requires_owned_run_lock(tmp_path):
    from filelock import FileLock
    lock = FileLock(str(tmp_path / ".run.lock"))
    with pytest.raises(RuntimeError, match="owned run lock"):
        flux3._recover_latest(tmp_path, {}, None, None, 1, lock)
    with FileLock(str(tmp_path / "other.lock")) as other:
        with pytest.raises(RuntimeError, match="owned run lock"):
            flux3._recover_latest(tmp_path, {}, None, None, 1, other)


def test_corrupt_latest_does_not_roll_back_to_valid_backup(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    kwargs = options(tmp_path)
    root = Path(kwargs["output_root"])
    with crash_at_publish(monkeypatch, root, "after_stage_rename"):
        with pytest.raises(RuntimeError, match="injected publication crash"):
            flux3.train_assets(**kwargs)
    (root / "latest/pytorch_model.bin").write_bytes(b"corrupt latest")
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        flux3.train_assets(**{**kwargs, "resume": True, "pilot_steps": 3})
    assert before == {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}


def test_incomplete_stage_preserved_but_never_promoted(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    kwargs = options(tmp_path)
    flux3.train_assets(**kwargs)
    evidence = Path(kwargs["output_root"]) / "snapshot-incomplete/pytorch_model.bin"
    evidence.parent.mkdir()
    evidence.write_bytes(b"interrupted serialization")
    flux3.train_assets(**{**kwargs, "resume": True, "pilot_steps": 3})
    assert evidence.read_bytes() == b"interrupted serialization"
    state = torch.load(Path(kwargs["output_root"]) / "latest/trainer.pt", weights_only=False)
    assert state["step"] == 3


def test_resume_accumulation_tail_and_epoch(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    kwargs = {**options(tmp_path), "grad_acc_step": 5}
    root = Path(kwargs["output_root"])
    with crash_at_publish(monkeypatch, root, "after_old_rename"):
        with pytest.raises(RuntimeError, match="injected publication crash"):
            flux3.train_assets(**kwargs)
    flux3.train_assets(**{**kwargs, "resume": True, "pilot_steps": 4})
    state = torch.load(root / "latest/trainer.pt", weights_only=False)
    reference = tmp_path / "reference"
    flux3.train_assets(**{**kwargs, "output_root": str(reference), "pilot_steps": 4})
    expected = torch.load(reference / "latest/trainer.pt", weights_only=False)
    assert (state["step"], state["epoch"], state["next_index"]) == (4, 1, 5)
    assert_tree_equal(state["optimizer"], expected["optimizer"])
    assert_tree_equal(state["rng"][0]["torch"], expected["rng"][0]["torch"])
    assert_tree_equal(torch.load(root / "latest/pytorch_model.bin", weights_only=True),
                      torch.load(reference / "latest/pytorch_model.bin", weights_only=True))


# Reuse real, small FLUX.3 components as an additional native-format control.
from test_flux3_finetune import assets, inputs, kwargs as model_kwargs


def test_validates_real_flux3_native_snapshot(assets, tmp_path):
    import random
    import numpy as np
    from unitorch.models.peft.diffusers.modeling_flux3 import Flux3LoraForImageEditingGeneration

    model = Flux3LoraForImageEditingGeneration(**model_kwargs(assets), lora_r=2)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model(**inputs()).backward()
    optim.step()
    model.save_checkpoint(str(tmp_path))
    identity = dict(world_size=1, grad_acc_step=4)
    state = dict(identity=identity, base_fingerprint=model.base_fingerprint, optimizer=optim.state_dict(),
                 epoch=0, next_index=4, step=1, best_loss=0.1,
                 rng=[dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch=torch.get_rng_state(), cuda=torch.zeros(16, dtype=torch.uint8))])
    torch.save(state, tmp_path / "trainer.pt")
    (tmp_path / "trainer.json").write_text(json.dumps(dict(
        sha256=file_sha256(tmp_path / "trainer.pt"), step=1, val_loss=0.1)))
    restored = flux3._validate_training_checkpoint(tmp_path, identity, model, optim, 12)
    assert_tree_equal(restored["optimizer"], optim.state_dict())
