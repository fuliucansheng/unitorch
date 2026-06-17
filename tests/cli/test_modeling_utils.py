# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from unitorch.cli.models.modeling_utils import (
    ObjectInputs,
    TensorInputs,
    TensorTargets,
)
from unitorch.cli.tasks.supervised import collate_fn


@dataclass
class Labels(TensorTargets):
    label: torch.Tensor


def test_object_inputs_stack_keeps_non_tensor_values():
    first = ObjectInputs(text="hello", meta={"id": 1})
    second = ObjectInputs(text="world", meta={"id": 2})

    batch = ObjectInputs.stack(first, second)

    assert batch.dict() == {
        "text": ["hello", "world"],
        "meta": [{"id": 1}, {"id": 2}],
    }
    assert batch.cuda().dict() == batch.dict()


def test_collate_fn_keeps_mixed_tensor_and_object_inputs():
    batch = [
        (
            [TensorInputs(input_ids=torch.tensor([1, 2])), ObjectInputs(text="a")],
            Labels(label=torch.tensor(0)),
        ),
        (
            [TensorInputs(input_ids=torch.tensor([3, 4])), ObjectInputs(text="b")],
            Labels(label=torch.tensor(1)),
        ),
    ]

    inputs, targets = collate_fn(batch)

    assert torch.equal(inputs.dict()["input_ids"], torch.tensor([[1, 2], [3, 4]]))
    assert inputs.dict()["text"] == ["a", "b"]
    assert torch.equal(targets.dict()["label"], torch.tensor([0, 1]))


def test_object_inputs_process_keeps_arbitrary_fields():
    from unitorch.cli.models.processing_utils import PreProcessor

    text_inputs = PreProcessor._object_inputs(None, prompt="hello")
    image_inputs = PreProcessor._object_inputs(
        None, prompt="describe", images="image.png"
    )

    assert isinstance(text_inputs, ObjectInputs)
    assert text_inputs.dict() == {"prompt": "hello"}
    assert isinstance(image_inputs, ObjectInputs)
    assert image_inputs.dict() == {"prompt": "describe", "images": "image.png"}


class FakeLLM:
    def __init__(self):
        self.prompts = None
        self.sampling_params = None

    def generate(self, prompts, sampling_params):
        self.prompts = prompts
        self.sampling_params = sampling_params
        
        return [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[index + 1])])
            for index, _ in enumerate(prompts)
        ]


def test_base_vllm_model_accepts_object_prompt_inputs():
    pytest.importorskip("vllm")
    from unitorch.models.vllm.modeling import VLLMForGeneration

    model = object.__new__(VLLMForGeneration)
    model.llm = FakeLLM()
    
    outputs = model.generate(prompt=["hello", "world"])

    assert model.llm.prompts == [{"prompt": "hello"}, {"prompt": "world"}]
    assert outputs == [[[1]], [[2]]]


def test_base_vllm_vl_model_accepts_object_prompt_and_image_inputs():
    pytest.importorskip("vllm")
    from unitorch.models.vllm.modeling_vl import VLLMVLForGeneration

    image = Image.new("RGB", (2, 2), (255, 0, 0))
    model = object.__new__(VLLMVLForGeneration)
    model.llm = FakeLLM()

    outputs = model.generate(prompt=["describe"], images=[image])

    assert model.llm.prompts[0]["prompt"] == "describe"
    assert model.llm.prompts[0]["multi_modal_data"]["image"] is image
    assert outputs == [[[1]]]
