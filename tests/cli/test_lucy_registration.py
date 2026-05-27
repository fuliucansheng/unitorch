# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pytest

from unitorch.cli import Config


def _importorskip_lucy():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "LucyEditPipeline"):
        pytest.skip("LucyEditPipeline is not available in this diffusers version")


def test_lucy_cli_registrations():
    _importorskip_lucy()
    pytest.importorskip("cv2")

    import unitorch.cli.models.diffusers  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    assert "lucy-edit-v1.1-dev" in pretrained_stable_infos
    assert "core/model/diffusers/video_editing/lucy" in registered_model
    assert "core/process/diffusion/lucy/video_editing" in registered_process
    assert "core/process/diffusion/lucy/video_editing/inputs" in registered_process


def test_lucy_fastapi_registrations():
    _importorskip_lucy()
    pytest.importorskip("fastapi")
    pytest.importorskip("cv2")

    import unitorch.cli.fastapis.lucy  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/lucy/video_editing" in registered_fastapi


@pytest.mark.parametrize(
    "path",
    [
        "examples/configs/diffusion/video_editing/lucy.ini",
        "examples/configs/fastapis/lucy.ini",
        "examples/fastapis.ini",
    ],
)
def test_lucy_configs_parse(path):
    Config(path)
