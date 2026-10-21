"""Adapters must compose through the existing train/infer registries."""
import pytest

from unitorch.cli import Config, cached_path


def test_flux3_registry():
    pytest.importorskip("diffusers", minversion="0.41.0.dev0")
    pytest.importorskip("peft")

    from unitorch.cli import registered_model, registered_process
    for task in ("text2image", "editing"):
        assert f"core/model/diffusers/{task}/flux3" in registered_model
        assert f"core/model/diffusers/{task}/peft/lora/flux3" in registered_model
        assert f"core/process/diffusion/flux3/{task}" in registered_process
        assert f"core/process/diffusion/flux3/{task}/inputs" in registered_process


@pytest.mark.parametrize(
    "path",
    [
        "cli/configs/diffusion/text2image/flux3.ini",
        "cli/configs/diffusion/text2image/flux3.klein-base.e2e.ini",
        "cli/configs/diffusion/text2image/flux3.klein-base.lora.ini",
        "cli/configs/diffusion/editing/flux3.ini",
        "cli/configs/diffusion/editing/flux3.klein-base.e2e.ini",
        "cli/configs/diffusion/editing/flux3.klein-base.lora.ini",
    ],
)
def test_flux3_configs_parse(path):
    path = cached_path(path)
    Config(path)
