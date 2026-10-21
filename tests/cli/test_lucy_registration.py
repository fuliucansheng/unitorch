# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pytest

from unitorch.cli import Config, cached_path


def _importorskip_lucy():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "LucyEditPipeline"):
        pytest.skip("LucyEditPipeline is not available in this diffusers version")


def test_lucy_cli_registrations():
    _importorskip_lucy()
    pytest.importorskip("cv2")

    import unitorch.cli.models.diffusers  # noqa: F401
    import unitorch.cli.models.peft.diffusers  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    assert "lucy-edit-v1.1-dev" in pretrained_stable_infos
    assert "core/model/diffusers/video_editing/lucy" in registered_model
    assert "core/model/diffusers/peft/lora/video_editing/lucy" in registered_model
    assert "core/process/diffusion/lucy/video_editing" in registered_process
    assert "core/process/diffusion/lucy/video_editing/inputs" in registered_process


def test_lucy_fastapi_registrations():
    _importorskip_lucy()
    pytest.importorskip("fastapi")
    pytest.importorskip("cv2")

    import unitorch.cli.fastapis.lucy  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/lucy/video_editing" in registered_fastapi


def test_lucy_state_dict_uses_wan_text_loader(monkeypatch):
    _importorskip_lucy()

    import unitorch.cli.models.diffusers.modeling_lucy as modeling_lucy

    def fake_load_weight(path, prefix_keys=None, use_auth_token=None, replace_keys=None):
        if "transformer" in path:
            return {"transformer.weight": 1}
        if "vae" in path:
            return {"vae.weight": 1}
        raise AssertionError(f"unexpected weight source {path}")

    def fake_load_wan_text_weight(path, use_auth_token=None, replace_keys=None):
        assert path == "text.safetensors"
        return {
            "text.shared.weight": 1,
            "text.encoder.embed_tokens.weight": 1,
        }

    monkeypatch.setattr(modeling_lucy, "load_weight", fake_load_weight)
    monkeypatch.setattr(modeling_lucy, "load_wan_text_weight", fake_load_wan_text_weight)

    state_dict = modeling_lucy._lucy_state_dict(
        {
            "transformer": {"weight": "transformer.safetensors"},
            "text": {"weight": "text.safetensors"},
            "vae": {"weight": "vae.safetensors"},
        },
        use_auth_token=False,
    )

    assert len(state_dict) == 3
    assert "text.encoder.embed_tokens.weight" in state_dict[1]


@pytest.mark.parametrize(
    "path",
    [
        "cli/configs/diffusion/video_editing/lucy.ini",
        "cli/configs/diffusion/editing/lucy.lora.ini",
        "cli/configs/fastapis/lucy.ini",
        "cli/configs/fastapis.ini",
    ],
)
def test_lucy_configs_parse(path):
    path = cached_path(path)
    Config(path)
