"""Task placement must survive real FLUX.2 model/dataset config factories."""

import json
from functools import wraps
from unittest.mock import patch

import fire
import pytest
import torch
from PIL import Image

from flux2_test_utils import make_assets, write_cli_config
from unitorch.cli import Config, init_registered_module, registered_model, registered_task
from unitorch.cli.consoles.infer import infer as cli_infer
from unitorch.cli.tasks.supervised import SupervisedTask


TASK = "core/task/supervised"
MODEL = "core/model/diffusers/peft/lora/text2image/flux2"


@pytest.fixture(autouse=True)
def dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", tmp_path / "datasets")


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return make_assets(tmp_path_factory.mktemp("offload-config-assets"), "dev")


def component_options(value):
    if value is None:
        return []
    return [(section, "cpu_offload", value) for section in (
        MODEL, "core/dataset/ast", "core/dataset/ast/train",
        "core/dataset/ast/dev", "core/dataset/ast/test",
        "core/process/diffusion/flux2",
    )]


@pytest.mark.parametrize("task_offload,component_offload", [
    (True, None), (True, False), (False, True), (None, True),
])
def test_registry_task_placement(assets, tmp_path, task_offload, component_offload):
    path = write_cli_config(tmp_path, assets, "dev", "lora", "text2image")
    params = component_options(component_offload) + [("core/cli", "local_rank", -1)]
    if task_offload is not None:
        params.append((TASK, "cpu_offload", task_offload))
    config = Config(str(path), params=params)
    original_init = SupervisedTask.__init__
    placement = []

    @wraps(original_init)
    def observe_init(self, *args, **kwargs):
        # Simulate only constructor GPU visibility; never allocate CUDA storage.
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.nn.Module, "cuda", autospec=True, side_effect=lambda model: model
        ) as cuda:
            original_init(self, *args, **kwargs)
        placement.append((kwargs["cpu_offload"], cuda.call_count))

    with patch.object(SupervisedTask, "__init__", observe_init):
        task = init_registered_module(TASK, config, registered_task)
    assert len(task.datasets.get("test")) == 2
    assert task.n_gpu == 1
    assert all(parameter.device.type == "cpu" for parameter in task.model.parameters())
    assert placement == [(bool(task_offload), 0 if task_offload else 1)]


@pytest.mark.parametrize("component_offload", [None, False])
def test_cli_override_load_then_infer_offload(assets, tmp_path, component_offload):
    """Exercise Fire overrides, registry construction, checkpoint IO and real hooks."""
    path = write_cli_config(tmp_path, assets, "dev", "lora", "text2image")
    config = Config(str(path), params=component_options(component_offload))
    config.set(TASK, "cpu_offload", "False")
    config.set(MODEL, "enable_cpu_offload", "False")
    config.set("core/cli", "local_rank", "-1")
    with path.open("w") as handle:
        config.write(handle)

    source = init_registered_module(MODEL, config, registered_model)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if ".lora_B." in name:
                parameter.fill_(0.02)
    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    source.save_checkpoint(str(checkpoint))
    expected = torch.load(checkpoint / "pytorch_model.bin", weights_only=True, map_location="cpu")
    del source

    original_init = SupervisedTask.__init__
    tasks, loaded, generated = [], [], []

    @wraps(original_init)
    def observe_init(self, *args, **kwargs):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.nn.Module, "cuda", autospec=True, side_effect=lambda model: model
        ) as cuda:
            original_init(self, *args, **kwargs)
        assert kwargs["cpu_offload"] is True
        cuda.assert_not_called()
        assert all(parameter.device.type == "cpu" for parameter in self.model.parameters())
        tasks.append(self)
        load = self.model.from_checkpoint

        def observe_load(*args, **kwargs):
            result = load(*args, **kwargs)
            state = self.model.state_dict()
            for name, value in expected.items():
                torch.testing.assert_close(state[name], value, rtol=0, atol=0)
            loaded.append(args[0])
            return result

        self.model.from_checkpoint = observe_load

        def observe_forward(module, inputs, outputs):
            assert loaded == [str(checkpoint)]
            assert torch.isfinite(outputs.outputs).all()
            generated.append(outputs.outputs.shape[0])

        self.model.register_forward_hook(observe_forward)

    command = [str(path), f"--from_ckpt_dir={checkpoint}",
               f"--{TASK}@cpu_offload=True", f"--{MODEL}@enable_cpu_offload=True",
               f"--{MODEL}@offload_device=cpu"]
    # Use the console's Fire/Config/registry path without exiting the test process.
    # Replace only this module's os reference; forked workers need real os._exit.
    with patch.object(SupervisedTask, "__init__", observe_init), patch(
        "unitorch.cli.consoles.infer.os"
    ) as console_os, patch.object(torch.cuda, "is_available", return_value=False):
        fire.Fire(cli_infer, command=command)
    console_os._exit.assert_called_once_with(0)
    assert generated == [2]
    model = tasks[0].model.model
    assert model._inference_device == torch.device("cpu")
    for component in (model.text, model.transformer, model.vae):
        assert any(hasattr(module, "_hf_hook") for module in component.modules())
        assert any(parameter.device.type == "meta" for parameter in component.parameters())
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["text"] == "red square"
        with Image.open(tmp_path / "images" / row["diffusion"]) as output:
            assert output.size == (16, 16)
