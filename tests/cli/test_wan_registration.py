# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import pytest
from PIL import Image

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
    processor.text2video_inputs = text2video_inputs.__get__(processor, WanProcessor)

    image = Image.new("RGB", (400, 300))
    outputs = WanProcessor.image2video_inputs(processor, "prompt", image)

    assert processor.center_crop_processor.size == (832, 624)
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
