"""Adapters must compose through the existing train/infer registries."""
import pytest

pytest.importorskip("diffusers", minversion="0.41.0.dev0")
pytest.importorskip("peft")


def test_flux3_registry():
    from unitorch.cli import registered_model, registered_process
    for task in ("text2image", "editing"):
        assert f"core/model/diffusers/{task}/flux3" in registered_model
        assert f"core/model/diffusers/{task}/peft/lora/flux3" in registered_model
        assert f"core/process/diffusion/flux3/{task}" in registered_process
        assert f"core/process/diffusion/flux3/{task}/inputs" in registered_process
