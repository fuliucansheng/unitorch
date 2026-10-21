"""FLUX.3 captions are tokenized without ever passing a target image to Qwen."""
import json

import pytest
import torch

pytest.importorskip("diffusers", minversion="0.41.0.dev0")
pytest.importorskip("peft")
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer
from unitorch.models.diffusers.processing_flux3 import Flux3Processor


@pytest.fixture
def processor(tmp_path):
    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    vocab.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(vocab=str(tmp_path / "vocab.json"), merges=str(tmp_path / "merges.txt"),
                              pad_token="<|endoftext|>", eos_token="<|im_end|>")
    tokenizer.chat_template = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer.save_pretrained(tmp_path)
    return Flux3Processor(str(tmp_path), max_seq_length=320, image_size=(16, 16), image_size_multiple=4)


def test_processor_separates_source_target(processor):
    source, target = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
    a = processor.editing("A detailed product photograph on white.", source, target)
    b = processor.editing("A detailed product photograph on white.", source, source)
    for key in ("input_ids", "attention_mask", "refer_pixel_values", "negative_input_ids"):
        torch.testing.assert_close(a[key], b[key])
    assert not torch.equal(a.pixel_values, b.pixel_values)
    assert a.input_ids.shape == (320,)
    assert a.attention_mask.sum() < 160
    assert a.negative_attention_mask.sum() < a.attention_mask.sum()


def test_processor_refuses_silent_geometry_change(processor):
    with pytest.raises(ValueError, match="size|geometry"):
        processor.editing_inputs("caption", Image.new("RGB", (8, 16)))


def test_long_caption_is_not_silently_truncated(processor):
    with pytest.raises(ValueError, match="caption|tokens"):
        processor.text2image_inputs("long caption " * 200)
