# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json

import pytest

from unitorch.cli import Config


def test_clip_cli_registrations():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import unitorch.cli.models.clip  # noqa: F401
    import unitorch.cli.models.peft  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.clip import pretrained_clip_infos

    assert "clip-vit-base-patch16" in pretrained_clip_infos
    assert "core/model/classification/clip" in registered_model
    assert "core/model/classification/clip/image/v2" in registered_model
    assert "core/model/matching/clip" in registered_model
    assert "core/model/classification/peft/lora/clip" in registered_model
    assert "core/process/clip/classification" in registered_process
    assert "core/process/clip/text_classification" in registered_process
    assert "core/process/clip/image_classification" in registered_process


def test_clip_fastapi_registrations():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/clip" in registered_fastapi
    assert "core/fastapi/clip/text" in registered_fastapi
    assert "core/fastapi/clip/image" in registered_fastapi
    assert "core/fastapi/clip/image/v2" in registered_fastapi
    assert "core/fastapi/clip/peft/lora" in registered_fastapi
    assert "core/fastapi/clip/matching" in registered_fastapi


def test_clip_lora_fastapi_start_uses_config_pretrained_name(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip as clip_fastapi

    captured = {}
    section = "core/fastapi/pipeline/clip/peft/lora"

    def fake_from_config(cls, config, **kwargs):
        captured["config_pretrained_name"] = config.getdefault(
            section, "pretrained_name", None
        )
        captured["passed_pretrained_name"] = kwargs.get("pretrained_name")
        return object()

    monkeypatch.setattr(
        clip_fastapi.ClipLoraForClassificationPipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    service = clip_fastapi.ClipLoraForClassificationFastAPI(
        Config(
            params=[
                (section, "pretrained_name", "clip-vit-base-patch32"),
                (section, "label_dict", "{'cat': 'cat', 'dog': 'dog'}"),
            ]
        )
    )

    assert service.start() == "start success"
    assert captured["config_pretrained_name"] == "clip-vit-base-patch32"
    assert captured["passed_pretrained_name"] is None


def test_clip_image_v2_fastapi_start_uses_config_pretrained_name(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("fastapi")

    import unitorch.cli.fastapis.clip as clip_fastapi

    captured = {}
    section = "core/fastapi/pipeline/clip/image/v2"

    def fake_from_config(cls, config, **kwargs):
        captured["config_pretrained_name"] = config.getdefault(
            section, "pretrained_name", None
        )
        captured["passed_pretrained_name"] = kwargs.get("pretrained_name")
        return object()

    monkeypatch.setattr(
        clip_fastapi.ClipForImageClassificationV2Pipeline,
        "from_config",
        classmethod(fake_from_config),
    )

    service = clip_fastapi.ClipForImageClassificationV2FastAPI(
        Config(
            params=[
                (section, "pretrained_name", "clip-vit-base-patch32"),
                (section, "label_dict", "{'cat': 'a photo of a cat'}"),
            ]
        )
    )

    assert service.start() == "start success"
    assert captured["config_pretrained_name"] == "clip-vit-base-patch32"
    assert captured["passed_pretrained_name"] is None


def test_clip_image_v2_pipeline_from_config_supports_init_lora(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
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

    assert captured["lora_weight_path"] == "/tmp/clip-lora.safetensors"
    assert captured["lora_weight"] == 0.65
    assert captured["lora_alpha"] == 24.0


def test_clip_matching_pipeline_from_config_resolves_init_lora_name(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
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
                (section, "pretrained_lora_name", "demo-lora"),
                (section, "pretrained_lora_weight", "0.35"),
                (section, "pretrained_lora_alpha", "16.0"),
            ]
        )
    )

    assert captured["lora_weight_path"] == "/tmp/demo-lora.safetensors"
    assert captured["lora_weight"] == 0.35
    assert captured["lora_alpha"] == 16.0


@pytest.mark.parametrize(
    "path",
    [
        "examples/configs/classification/clip.ini",
        "examples/configs/classification/clip.lora.ini",
        "examples/configs/fastapis/clip.ini",
        "examples/configs/classification/text_clip.ini",
        "examples/configs/classification/image_clip.ini",
    ],
)
def test_clip_configs_parse(path):
    Config(path)


def test_clip_lora_classification_supports_train_and_eval(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch
    from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

    from unitorch.models.peft.modeling_clip import ClipLoraForClassification

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

    model = ClipLoraForClassification(
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


def test_clip_image_classification_v2_supports_train_and_eval(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch
    from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

    from unitorch.models.clip.modeling import ClipForImageClassificationV2

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
