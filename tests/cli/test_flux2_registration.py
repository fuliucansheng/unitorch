# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pytest

from unitorch.cli import Config, cached_path


def test_flux2_cli_registrations():
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")

    import unitorch.cli.models.diffusers  # noqa: F401
    import unitorch.cli.models.peft.diffusers  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    assert "flux2-dev" in pretrained_stable_infos
    assert "flux2-klein-4b" in pretrained_stable_infos
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


def test_flux2_fastapi_start_uses_config_pretrained(monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.flux2  # noqa: F401
    from unitorch.cli.fastapis.flux2.text2image import (
        Flux2FastAPIPipeline,
        Flux2Text2ImageFastAPI,
    )

    captured = {}

    def fake_from_config(cls, config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        Flux2FastAPIPipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    fastapi = Flux2Text2ImageFastAPI(Config(cached_path("cli/configs/fastapis/flux2.ini")))
    assert fastapi.start() == "start success"
    assert captured["pretrained_name"] is None


def test_flux2_editing_fastapi_start_uses_config_pretrained(monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.flux2  # noqa: F401
    from unitorch.cli.fastapis.flux2.image_editing import (
        Flux2ImageEditingFastAPI,
        Flux2ImageEditingFastAPIPipeline,
    )

    captured = {}

    def fake_from_config(cls, config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        Flux2ImageEditingFastAPIPipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    fastapi = Flux2ImageEditingFastAPI(Config(cached_path("cli/configs/fastapis/flux2.ini")))
    assert fastapi.start() == "start success"
    assert captured["pretrained_name"] is None


def test_flux2_fastapi_health_check_timeout():
    pytest.importorskip("fastapi")

    from unitorch.cli.consoles.fastapi import _health_check_timeout

    config = Config(cached_path("cli/configs/fastapis/flux2.ini"))
    assert _health_check_timeout(config) == 300.0


def test_flux2_klein_pretrained_assets():
    pytest.importorskip("diffusers")

    import unitorch.cli.models.diffusers  # noqa: F401
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    info = pretrained_stable_infos["flux2-klein-4b"]
    assert info["text"]["tokenizer_class"] == "Qwen2Tokenizer"
    assert info["text"]["tokenizer"].endswith("/tokenizer/tokenizer.json")
    assert info["text"]["vocab"].endswith("/tokenizer/vocab.json")
    assert info["text"]["merge"].endswith("/tokenizer/merges.txt")
    assert info["text"]["tokenizer_config"].endswith("/tokenizer/tokenizer_config.json")
    assert info["text"]["special_tokens_map"].endswith(
        "/tokenizer/special_tokens_map.json"
    )
    assert info["text"]["chat_template"].endswith("/tokenizer/chat_template.jinja")
    assert info["text"]["added_tokens"].endswith("/tokenizer/added_tokens.json")


@pytest.mark.parametrize(
    "path",
    [
        "cli/configs/diffusion/text2image/flux2.ini",
        "cli/configs/diffusion/text2image/flux2.lora.ini",
        "cli/configs/diffusion/editing/flux2.ini",
        "cli/configs/diffusion/editing/flux2.lora.ini",
        "cli/configs/fastapis/flux2.ini",
    ],
)
def test_flux2_configs_parse(path):
    path = cached_path(path)
    Config(path)
