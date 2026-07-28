# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json

import pytest

from unitorch.cli import Config, cached_path


def test_clip_cli_registrations():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import unitorch.cli.models.clip  # noqa: F401
    import unitorch.cli.models.peft  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.clip import pretrained_clip_infos

    assert "clip-vit-base-patch16" in pretrained_clip_infos
    for name in [
        "core/model/pretrain/clip",
        "core/model/classification/clip",
        "core/model/classification/clip/text",
        "core/model/classification/clip/image",
        "core/model/classification/clip/image/v2",
        "core/model/matching/clip",
        "core/model/classification/peft/lora/clip/image/v2",
        "core/model/matching/peft/lora/clip",
    ]:
        assert name in registered_model

    for name in [
        "core/process/clip/classification",
        "core/process/clip/text_classification",
        "core/process/clip/image_classification",
    ]:
        assert name in registered_process


def test_clip_fastapi_registrations():
    pytest.importorskip("transformers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip  # noqa: F401
    from unitorch.cli import registered_fastapi

    for name in [
        "core/fastapi/clip",
        "core/fastapi/clip/text",
        "core/fastapi/clip/image",
        "core/fastapi/clip/image/v2",
        "core/fastapi/clip/matching",
    ]:
        assert name in registered_fastapi


def test_clip_image_v2_fastapi_start_builds_service_replicas(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip as clip_fastapi

    created = []
    moved = []
    pipeline_section = "core/fastapi/pipeline/clip/image/v2"
    service_section = "core/fastapi/clip/image/v2"

    class FakePipeline:
        def __init__(self, index):
            self.index = index

        def to(self, device):
            moved.append((self.index, device))
            return self

    def fake_from_config(cls, config, **kwargs):
        created.append(
            {
                "config_pretrained_name": config.getdefault(
                    pipeline_section, "pretrained_name", None
                ),
                "passed_pretrained_name": kwargs.get("pretrained_name"),
            }
        )
        return FakePipeline(len(created) - 1)

    monkeypatch.setattr(
        clip_fastapi.ClipForImageClassificationV2Pipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    service = clip_fastapi.ClipForImageClassificationV2FastAPI(
        Config(
            params=[
                (pipeline_section, "pretrained_name", "clip-vit-base-patch32"),
                (pipeline_section, "label_dict", "{'cat': 'a photo of a cat'}"),
                (service_section, "pipeline_num_replicas", "2"),
                (service_section, "pipeline_replica_devices", "['cpu', 'cuda:0']"),
                (service_section, "pipeline_replica_lock", "False"),
            ]
        )
    )

    assert service.start() == "start success"
    assert created == [
        {
            "config_pretrained_name": "clip-vit-base-patch32",
            "passed_pretrained_name": None,
        },
        {
            "config_pretrained_name": "clip-vit-base-patch32",
            "passed_pretrained_name": None,
        },
    ]
    assert moved == [(0, "cpu"), (1, "cuda:0")]
    assert service._pipes.num_replicas == 2
    assert service._pipes.status()["lock"] is False


def test_clip_image_v2_pipeline_from_config_supports_lora_path(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip as clip_fastapi

    captured = {}
    section = "core/fastapi/pipeline/clip/image/v2"

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(clip_fastapi, "cached_path", lambda path: path)
    monkeypatch.setattr(
        clip_fastapi.ClipForImageClassificationV2Pipeline,
        "__init__",
        fake_init,
    )

    clip_fastapi.ClipForImageClassificationV2Pipeline.from_config(
        Config(
            params=[
                (section, "label_dict", "{'cat': 'a photo of a cat'}"),
                (section, "pretrained_lora_weights_path", "/tmp/clip-lora.safetensors"),
                (section, "pretrained_lora_weights", "0.65"),
                (section, "pretrained_lora_alphas", "24.0"),
            ]
        )
    )

    assert captured["lora_checkpoints"] == "/tmp/clip-lora.safetensors"
    assert captured["lora_weights"] == 0.65
    assert captured["lora_alphas"] == 24.0


def test_clip_matching_pipeline_from_config_resolves_lora_name(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip as clip_fastapi

    captured = {}
    section = "core/fastapi/pipeline/matching/clip"

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(clip_fastapi, "cached_path", lambda path: path)
    monkeypatch.setattr(
        clip_fastapi,
        "pretrained_clip_extensions_infos",
        {"demo-lora": {"lora": {"weight": "/tmp/demo-lora.safetensors"}}},
    )
    monkeypatch.setattr(
        clip_fastapi.ClipForMatchingPipeline,
        "__init__",
        fake_init,
    )

    clip_fastapi.ClipForMatchingPipeline.from_config(
        Config(
            params=[
                (section, "pretrained_lora_names", "demo-lora"),
                (section, "pretrained_lora_weights", "0.35"),
                (section, "pretrained_lora_alphas", "16.0"),
            ]
        )
    )

    assert captured["lora_checkpoints"] == "/tmp/demo-lora.safetensors"
    assert captured["lora_weights"] == 0.35
    assert captured["lora_alphas"] == 16.0


@pytest.mark.parametrize(
    "path",
    [
        "cli/configs/classification/clip.ini",
        "cli/configs/classification/clip.lora.ini",
        "cli/configs/fastapis/clip.ini",
        "cli/configs/classification/text_clip.ini",
        "cli/configs/classification/image_clip.ini",
    ],
)
def test_clip_configs_parse(path):
    path = cached_path(path)
    Config(path)


def _write_minimal_clip_assets(tmp_path):
    from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

    config_path = tmp_path / "config.json"
    vocab_path = tmp_path / "vocab.json"
    merge_path = tmp_path / "merges.txt"
    vision_config_path = tmp_path / "preprocessor_config.json"

    config = CLIPConfig(
        text_config=CLIPTextConfig(
            vocab_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        ),
        projection_dim=16,
    )
    config_path.write_text(config.to_json_string())
    vocab_path.write_text(
        json.dumps(
            {
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "a</w>": 2,
                "b</w>": 3,
            }
        )
    )
    merge_path.write_text("#version: 0.2\n")
    vision_config_path.write_text(
        json.dumps(
            {
                "do_resize": True,
                "size": {"shortest_edge": 32},
                "do_center_crop": True,
                "crop_size": {"height": 32, "width": 32},
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "do_convert_rgb": True,
            }
        )
    )
    return config_path, vocab_path, merge_path, vision_config_path


def test_clip_image_classification_v2_supports_train_and_eval(tmp_path):
    pytest.importorskip("transformers")
    import torch

    from unitorch.models.clip.modeling import ClipForImageClassificationV2

    config_path, vocab_path, merge_path, vision_config_path = (
        _write_minimal_clip_assets(tmp_path)
    )
    model = ClipForImageClassificationV2(
        config_path=str(config_path),
        labels=["a", "b"],
        vocab_path=str(vocab_path),
        merge_path=str(merge_path),
        vision_config_path=str(vision_config_path),
        output_embed_dim=8,
        max_seq_length=8,
    )
    pixel_values = torch.randn(2, 3, 32, 32)

    model.train()
    train_outputs = model(pixel_values)
    assert train_outputs.shape == (2, 2)
    assert model.labels_embeds is None

    model.eval()
    eval_outputs = model(pixel_values)
    assert eval_outputs.shape == (2, 2)
    assert model.labels_embeds is not None
    assert model.labels_embeds.shape == (2, 8)

    cached_labels = model.labels_embeds
    eval_outputs_again = model(pixel_values)
    assert eval_outputs_again.shape == (2, 2)
    assert model.labels_embeds is cached_labels


def test_clip_lora_image_classification_v2_supports_forward(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch

    from unitorch.models.peft.modeling_clip import ClipLoraForImageClassificationV2

    config_path, vocab_path, merge_path, vision_config_path = (
        _write_minimal_clip_assets(tmp_path)
    )
    model = ClipLoraForImageClassificationV2(
        config_path=str(config_path),
        labels=["a", "b"],
        vocab_path=str(vocab_path),
        merge_path=str(merge_path),
        vision_config_path=str(vision_config_path),
        output_embed_dim=8,
        max_seq_length=8,
    )
    pixel_values = torch.randn(2, 3, 32, 32)

    model.train()
    train_outputs = model(pixel_values)
    assert train_outputs.shape == (2, 2)

    model.eval()
    eval_outputs = model(pixel_values)
    assert eval_outputs.shape == (2, 2)
