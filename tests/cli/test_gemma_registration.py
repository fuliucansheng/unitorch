# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pytest

from unitorch.cli import Config, cached_path


def test_gemma_cli_registrations():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import unitorch.cli.models.gemma  # noqa: F401
    import unitorch.cli.models.peft  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.gemma import pretrained_gemma_infos

    assert "gemma-4-12b" in pretrained_gemma_infos
    assert "core/model/generation/gemma" in registered_model
    assert "core/model/generation/gemma_vl" in registered_model
    assert "core/model/generation/peft/lora/gemma" in registered_model
    assert "core/process/gemma/generation" in registered_process
    assert "core/process/gemma/generation/inputs" in registered_process
    assert "core/process/gemma_vl/generation" in registered_process
    assert "core/process/gemma_vl/generation/inputs" in registered_process


def test_gemma_fastapi_registrations():
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.gemma  # noqa: F401
    import unitorch.cli.fastapis.gemma_vl  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/gemma" in registered_fastapi
    assert "core/fastapi/gemma_vl" in registered_fastapi


def test_gemma_fastapi_start_uses_config_pretrained(monkeypatch):
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.gemma as gemma_fastapi

    captured = {}

    def fake_from_config(cls, config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        gemma_fastapi.GemmaForGenerationPipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    fastapi = gemma_fastapi.GemmaFastAPI(Config(cached_path("cli/configs/fastapis/gemma.ini")))
    assert fastapi.start() == "start success"
    assert captured["pretrained_name"] == "gemma-4-12b"


def test_gemma_vl_fastapi_start_uses_config_pretrained(monkeypatch):
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.gemma_vl as gemma_vl_fastapi

    captured = {}

    def fake_from_config(cls, config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        gemma_vl_fastapi.GemmaVLForGenerationPipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    fastapi = gemma_vl_fastapi.GemmaVLFastAPI(
        Config(cached_path("cli/configs/fastapis/gemma.ini"))
    )
    assert fastapi.start() == "start success"
    assert captured["pretrained_name"] == "gemma-4-12b"


@pytest.mark.parametrize(
    "path",
    [
        "cli/configs/generation/gemma.ini",
        "cli/configs/generation/gemma.lora.ini",
        "cli/configs/generation/gemma_vl.ini",
        "cli/configs/fastapis/gemma.ini",
        "cli/configs/fastapis.ini",
    ],
)
def test_gemma_configs_parse(path):
    path = cached_path(path)
    Config(path)
