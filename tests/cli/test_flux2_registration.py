# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pytest

from unitorch.cli import Config


def test_flux2_cli_registrations():
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")

    import unitorch.cli.models.diffusers  # noqa: F401
    import unitorch.cli.models.peft.diffusers  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    assert "flux2-dev" in pretrained_stable_infos
    assert "core/model/diffusers/text2image/flux2" in registered_model
    assert "core/model/diffusers/editing/flux2" in registered_model
    assert "core/model/diffusers/peft/lora/text2image/flux2" in registered_model
    assert "core/model/diffusers/peft/lora/editing/flux2" in registered_model
    assert "core/process/diffusion/flux2/text2image" in registered_process
    assert "core/process/diffusion/flux2/text2image/inputs" in registered_process
    assert "core/process/diffusion/flux2/editing" in registered_process
    assert "core/process/diffusion/flux2/editing/inputs" in registered_process


def test_flux2_fastapi_registrations():
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.flux2  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/flux2/text2image" in registered_fastapi
    assert "core/fastapi/flux2/editing" in registered_fastapi


@pytest.mark.parametrize(
    "path",
    [
        "examples/configs/diffusion/text2image/flux2.ini",
        "examples/configs/diffusion/text2image/flux2.lora.ini",
        "examples/configs/diffusion/editing/flux2.ini",
        "examples/configs/diffusion/editing/flux2.lora.ini",
        "examples/configs/fastapis/flux2.ini",
    ],
)
def test_flux2_configs_parse(path):
    Config(path)
