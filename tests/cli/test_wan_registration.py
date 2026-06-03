# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import pytest
from PIL import Image
import importlib

from unitorch.cli import Config
from unitorch.models import GenericOutputs


def test_wan_cli_registrations():
    pytest.importorskip("diffusers")
    pytest.importorskip("cv2")
    pytest.importorskip("peft")

    import unitorch.cli.models.diffusers  # noqa: F401
    import unitorch.cli.models.peft.diffusers  # noqa: F401
    from unitorch.cli import registered_model, registered_process
    from unitorch.cli.models.diffusers import pretrained_stable_infos

    assert "wan-v2.2-t2v-14b" in pretrained_stable_infos
    assert "wan-v2.2-i2v-14b" in pretrained_stable_infos
    assert "wan-v2.2-ti2v-5b" in pretrained_stable_infos
    assert "transformer2" not in pretrained_stable_infos["wan-v2.2-ti2v-5b"]
    assert "core/model/diffusers/text2video/wan" in registered_model
    assert "core/model/diffusers/image2video/wan" in registered_model
    assert "core/model/diffusers/peft/lora/text2video/wan" in registered_model
    assert "core/model/diffusers/peft/lora/image2video/wan" in registered_model
    assert "core/process/diffusion/wan/text2video" in registered_process
    assert "core/process/diffusion/wan/text2video/inputs" in registered_process
    assert "core/process/diffusion/wan/image2video" in registered_process
    assert "core/process/diffusion/wan/image2video/inputs" in registered_process


def test_wan_fastapi_registrations():
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")
    pytest.importorskip("cv2")

    import unitorch.cli.fastapis.wan  # noqa: F401
    from unitorch.cli import registered_fastapi

    assert "core/fastapi/wan/text2video" in registered_fastapi
    assert "core/fastapi/wan/image2video" in registered_fastapi


def test_add_adapter_compat_falls_back_to_inject(monkeypatch):
    pytest.importorskip("peft")

    import unitorch.models.peft as peft_module

    injected = {}

    class DummyModule:
        def add_adapter(self, config):
            raise ValueError(
                "The version of PEFT you are using is not compatible, please use a version >= 0.18.2"
            )

    def fake_inject_adapter_in_model(config, module, adapter_name="default"):
        injected["config"] = config
        injected["module"] = module
        injected["adapter_name"] = adapter_name
        return module

    monkeypatch.setattr(
        peft_module,
        "inject_adapter_in_model",
        fake_inject_adapter_in_model,
    )

    module = DummyModule()
    config = object()

    result = peft_module.add_adapter_compat(module, config)

    assert result is module
    assert injected["config"] is config
    assert injected["module"] is module
    assert injected["adapter_name"] == "default"


def test_wan_lora_text_target_modules_match_umt5_attention():
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")

    import unitorch.models.peft.diffusers.modeling_wan as modeling_wan

    assert modeling_wan.GenericWanLoraModel.text_target_modules == [
        "q",
        "k",
        "v",
        "o",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "examples/configs/diffusion/text2video/wan.ini",
        "examples/configs/diffusion/text2video/wan.lora.ini",
        "examples/configs/diffusion/image2video/wan.ini",
        "examples/configs/diffusion/image2video/wan.lora.ini",
        "examples/configs/fastapis/wan.ini",
    ],
)
def test_wan_configs_parse(path):
    Config(path)


def test_wan_image2video_resize_uses_width_height_order():
    pytest.importorskip("diffusers")
    pytest.importorskip("cv2")

    from unitorch.models.diffusers.processing_wan import WanProcessor

    class CaptureCrop:
        def __call__(self, image):
            self.size = image.size
            return image

    class CapturePreprocessor:
        def preprocess(self, image):
            return [torch.tensor(image.size)]

    class CaptureImageProcessor:
        def __call__(self, image):
            return torch.tensor(image.size)

    def text2video_inputs(self, prompt, negative_prompt="", max_seq_length=None):
        return GenericOutputs(
            input_ids=torch.tensor([1]),
            attention_mask=torch.tensor([1]),
            negative_input_ids=torch.tensor([0]),
            negative_attention_mask=torch.tensor([1]),
        )

    processor = WanProcessor.__new__(WanProcessor)
    processor.video_size = (832, 480)
    processor.divisor = 16
    processor.center_crop_processor = CaptureCrop()
    processor.vae_image_processor = CapturePreprocessor()
    processor.image_processor = CaptureImageProcessor()
    processor.text2video_inputs = text2video_inputs.__get__(processor, WanProcessor)

    image = Image.new("RGB", (400, 300))
    outputs = WanProcessor.image2video_inputs(processor, "prompt", image)

    assert processor.center_crop_processor.size == (832, 624)
    assert outputs.image_pixel_values.tolist() == [832, 624]
    assert outputs.vae_pixel_values.tolist() == [832, 624]


def test_wan_text2video_fastapi_keeps_config2_path(monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")
    pytest.importorskip("cv2")

    import unitorch.cli.fastapis.wan.text2video as text2video

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(text2video, "cached_path", lambda path: path)
    monkeypatch.setattr(
        text2video.WanForText2VideoFastAPIPipeline,
        "__init__",
        fake_init,
    )

    text2video.WanForText2VideoFastAPIPipeline.from_config(
        Config(params=[]),
        config_path="transformer.json",
        config2_path="transformer2.json",
        text_config_path="text.json",
        vae_config_path="vae.json",
        scheduler_config_path="scheduler.json",
        vocab_path="spiece.model",
        pretrained_weight_path="weights.safetensors",
    )

    assert captured["config2_path"] == "transformer2.json"


@pytest.mark.parametrize(
    ("module_name", "class_name", "section"),
    [
        (
            "unitorch.cli.models.diffusers.modeling_wan",
            "WanForText2VideoGeneration",
            "core/model/diffusers/text2video/wan",
        ),
        (
            "unitorch.cli.models.diffusers.modeling_wan",
            "WanForImage2VideoGeneration",
            "core/model/diffusers/image2video/wan",
        ),
    ],
)
def test_wan_ti2v_from_config_allows_missing_transformer2(
    monkeypatch, module_name, class_name, section
):
    pytest.importorskip("diffusers")
    pytest.importorskip("cv2")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, "cached_path", lambda path: path)
    monkeypatch.setattr(module, "load_weight", lambda *args, **kwargs: {})
    monkeypatch.setattr(cls, "__init__", fake_init)
    monkeypatch.setattr(cls, "from_pretrained", lambda self, *args, **kwargs: None)

    cls.from_config(
        Config(params=[(section, "pretrained_name", "wan-v2.2-ti2v-5b")])
    )

    assert captured["config2_path"] is None
    if class_name == "WanForImage2VideoGeneration":
        assert captured["expand_timesteps"] is True


def test_load_wan_text_weight_copies_shared_embedding(monkeypatch):
    pytest.importorskip("diffusers")

    from unitorch.cli.models import diffusers as diffusers_module

    shared = torch.randn(2, 3)

    monkeypatch.setattr(
        diffusers_module,
        "load_weight",
        lambda *args, **kwargs: {"text.shared.weight": shared},
    )

    weights = diffusers_module.load_wan_text_weight("unused.safetensors")

    assert torch.equal(weights["text.shared.weight"], shared)
    assert torch.equal(weights["text.encoder.embed_tokens.weight"], shared)


@pytest.mark.parametrize(
    ("factory_name", "pipeline_name"),
    [
        ("WanForText2VideoGeneration", "WanPipeline"),
        ("WanForImage2VideoGeneration", "WanImageToVideoPipeline"),
    ],
)
def test_wan_ti2v_pipeline_disables_boundary_ratio_without_transformer2(
    monkeypatch, factory_name, pipeline_name
):
    pytest.importorskip("diffusers")

    import unitorch.models.diffusers.modeling_wan as modeling_wan

    captured = {}

    class DummyTransformer:
        def enable_gradient_checkpointing(self):
            return None

    class DummyPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def set_progress_bar_config(self, **kwargs):
            return None

    def fake_generic_init(self, *args, **kwargs):
        self.boundary_ratio = 1.0
        self.transformer = DummyTransformer()
        self.vae = object()
        self.text = object()
        self.scheduler = object()

    monkeypatch.setattr(modeling_wan.GenericWanModel, "__init__", fake_generic_init)
    monkeypatch.setattr(modeling_wan, pipeline_name, DummyPipeline)

    factory = getattr(modeling_wan, factory_name)
    factory(
        config_path="unused.json",
        text_config_path="unused.json",
        vae_config_path="unused.json",
        scheduler_config_path="unused.json",
        config2_path=None,
        gradient_checkpointing=False,
    )

    assert captured["transformer_2"] is None
    assert captured["boundary_ratio"] is None


@pytest.mark.parametrize(
    ("module_name", "fastapi_class_name", "section"),
    [
        (
            "unitorch.cli.fastapis.wan.text2video",
            "WanForText2VideoFastAPI",
            "core/fastapi/pipeline/wan/text2video",
        ),
        (
            "unitorch.cli.fastapis.wan.image2video",
            "WanForImage2VideoFastAPI",
            "core/fastapi/pipeline/wan/image2video",
        ),
    ],
)
def test_wan_fastapi_start_uses_config_pretrained_name(
    monkeypatch, module_name, fastapi_class_name, section
):
    pytest.importorskip("diffusers")
    pytest.importorskip("fastapi")
    pytest.importorskip("cv2")

    module = importlib.import_module(module_name)
    fastapi_cls = getattr(module, fastapi_class_name)
    captured = {}

    def fake_from_config(config, **kwargs):
        captured["config_pretrained_name"] = config.getdefault(
            section, "pretrained_name", None
        )
        captured["passed_pretrained_name"] = kwargs.get("pretrained_name")
        return object()

    if "text2video" in module_name:
        monkeypatch.setattr(
            module.WanForText2VideoFastAPIPipeline,
            "from_config",
            fake_from_config,
        )
    else:
        monkeypatch.setattr(
            module.WanForImage2VideoFastAPIPipeline,
            "from_config",
            fake_from_config,
        )

    service = fastapi_cls(
        Config(params=[(section, "pretrained_name", "wan-v2.2-ti2v-5b")])
    )
    service.start()

    assert captured["config_pretrained_name"] == "wan-v2.2-ti2v-5b"
    assert captured["passed_pretrained_name"] is None
