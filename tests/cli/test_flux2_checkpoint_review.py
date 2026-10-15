"""Checkpoint integrity regressions using concrete, tiny FLUX.2 components."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flux2_test_utils import make_assets, model_class, model_kwargs


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("flux2-checkpoint-review")
    return {variant: make_assets(root / variant, variant) for variant in ("dev", "klein-4b")}


@pytest.mark.parametrize("variant", ["dev", "klein-4b"])
@pytest.mark.parametrize("task", ["text2image", "editing"])
@pytest.mark.parametrize("component", ["transformer.", "text.", "vae.encoder.", "vae.quant_conv."])
@pytest.mark.parametrize("damage", ["missing", "shape"])
def test_e2e_rejects_invalid_checkpoint_before_loading(assets, variant, task, component, damage, tmp_path):
    model = model_class(task, "e2e")(
        **model_kwargs(assets[variant]), torch_dtype="float32",
        freeze_text_encoder=False, freeze_vae_encoder=False,
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() + 1 for key, value in model.checkpoint_state_dict().items()}
    key = next(key for key in reversed(state) if key.startswith(component))
    if damage == "missing":
        del state[key]
    else:
        state[key] = torch.zeros(1)
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="FLUX.2 checkpoint.*(missing|shape)"):
        model.from_checkpoint(str(tmp_path))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("task", ["text2image", "editing"])
@pytest.mark.parametrize("damage", ["missing", "shape"])
def test_lora_rejects_invalid_custom_adapter_checkpoint(assets, task, damage, tmp_path):
    model = model_class(task, "lora")(**model_kwargs(assets["klein-4b"]),
                                    target_modules=["to_q"], torch_dtype="float32")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() + 1 for key, value in model.checkpoint_state_dict().items()}
    key = list(state)[-1]
    if damage == "missing":
        del state[key]
    else:
        state[key] = torch.zeros(1)
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="FLUX.2 checkpoint.*(missing|shape)"):
        model.from_checkpoint(str(tmp_path))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


def run_worker(assets, tmp_path, *, task, mode="lora", workflow="initialize", variant="klein-4b", epochs=1):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(Path(__file__).with_name("flux2_checkpoint_review_worker.py")),
               f"--assets={assets[variant]}", f"--output={tmp_path}", f"--task={task}",
               f"--mode={mode}", f"--workflow={workflow}", f"--variant={variant}", f"--epochs={epochs}"]
    # Select CPU through DeepSpeed; CPU debug mode hides GPUs from optional imports.
    env = dict(os.environ, DS_ACCELERATOR="cpu", UNITORCH_DEBUG="INFO", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               UNITORCH_CACHE=str(tmp_path / "cache"), HF_DATASETS_CACHE=str(tmp_path / "datasets"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    (tmp_path / f"command-{epochs}.json").write_text(json.dumps(command))
    log_path = tmp_path / f"worker-{epochs}.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, log_path.read_text()[-16000:]


@pytest.mark.parametrize("visible_devices", [None, "0,1"])
def test_worker_env_preserves_gpu_visibility(monkeypatch, tmp_path, visible_devices):
    # Inspect the launch contract without starting a worker or touching GPUs.
    parent_env = dict(os.environ, UNITORCH_DEBUG="CPU")
    if visible_devices is None:
        parent_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        parent_env["CUDA_VISIBLE_DEVICES"] = visible_devices
    monkeypatch.setattr(os, "environ", parent_env)
    launches = []

    def check_launch(command, *, env, **kwargs):
        assert env.get("CUDA_VISIBLE_DEVICES") == visible_devices
        assert ("CUDA_VISIBLE_DEVICES" in env) == (visible_devices is not None)
        assert env["UNITORCH_DEBUG"] == "INFO"
        assert env["DS_ACCELERATOR"] == "cpu"
        assert "--nproc_per_node=2" in command
        launches.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", check_launch)
    run_worker({"klein-4b": tmp_path}, tmp_path, task="text2image")
    assert len(launches) == 1


@pytest.mark.parametrize("task", ["text2image", "editing"])
@pytest.mark.parametrize("variant", ["dev", "klein-4b"])
def test_zero_init_populated_from_ckpt_dir(assets, task, variant, tmp_path):
    run_worker(assets, tmp_path, task=task, variant=variant)


@pytest.mark.parametrize("mode", ["lora", "e2e"])
def test_native_two_rank_restore_and_export(assets, mode, tmp_path):
    for epochs in (1, 2):
        run_worker(assets, tmp_path, task="text2image", mode=mode, workflow="export", epochs=epochs)
    first = torch.load(tmp_path / "expected-1.bin", weights_only=True)
    second = torch.load(tmp_path / "expected-2.bin", weights_only=True)
    assert any(not torch.equal(first[key], value) for key, value in second.items())


@pytest.mark.parametrize("task", ["text2image", "editing"])
@pytest.mark.parametrize("damaged", [False, True])
def test_unfrozen_legacy_mistral_checkpoint_keys(assets, task, damaged, tmp_path):
    model = model_class(task, "e2e")(**model_kwargs(assets["dev"]),
                                   freeze_text_encoder=False, torch_dtype="float32")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() + 1 for key, value in model.checkpoint_state_dict().items()}
    aliases = (("text.model.language_model.", "text.language_model.model."),
               ("text.lm_head.", "text.language_model.lm_head."),
               ("text.model.vision_tower.", "text.vision_tower."),
               ("text.model.multi_modal_projector.", "text.multi_modal_projector."))
    legacy = {}
    for key, value in state.items():
        for current, old in aliases:
            if key.startswith(current):
                key = old + key[len(current):]
                break
        legacy[key] = value
    if damaged:
        key = next(key for key in legacy if key.startswith("text.language_model.model."))
        legacy[key] = torch.zeros(1)
    torch.save(legacy, tmp_path / "pytorch_model.bin")
    if damaged:
        with pytest.raises(RuntimeError, match="FLUX.2 checkpoint shape"):
            model.from_checkpoint(str(tmp_path))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    else:
        model.from_checkpoint(str(tmp_path))
        for key, value in model.checkpoint_state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_lora_legacy_serialization(assets, tmp_path):
    model = model_class("text2image", "lora")(**model_kwargs(assets["klein-4b"]), torch_dtype="float32")
    expected = {key: value.clone() + 1 for key, value in model.checkpoint_state_dict().items()}
    torch.save(expected, tmp_path / "pytorch_model.bin", _use_new_zipfile_serialization=False)
    model.from_checkpoint(str(tmp_path))
    for key, value in model.checkpoint_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["extra_non_tensor", "non_string_key"])
def test_corrupt_checkpoint_is_rejected_before_loading(assets, damage, tmp_path):
    model = model_class("text2image", "e2e")(**model_kwargs(assets["klein-4b"]), torch_dtype="float32")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: value.clone() + 1 for key, value in model.checkpoint_state_dict().items()}
    if damage == "extra_non_tensor":
        state["unexpected"] = None
    else:
        state[123] = torch.zeros(1)
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="FLUX.2 checkpoint"):
        model.from_checkpoint(str(tmp_path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_base_shards_remain_permissive(assets):
    model = model_class("text2image", "e2e")(**model_kwargs(assets["klein-4b"]), torch_dtype="float32")
    state = model.state_dict()
    first, second = list(model.checkpoint_state_dict())[:2]
    before = state[first].clone()
    expected = state[second].clone() + 1
    model.from_pretrained(state_dict={first: torch.zeros(1), second: expected})
    torch.testing.assert_close(model.state_dict()[first], before, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict()[second], expected, rtol=0, atol=0)
