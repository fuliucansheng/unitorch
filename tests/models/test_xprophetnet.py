# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import unitorch
import pkg_resources
from absl.testing import absltest, parameterized
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import set_seed
from unitorch.models.xprophetnet import XProphetNetForGeneration, XProphetNetProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.xprophetnet.modeling import (
    XProphetNetForGeneration as CoreXProphetNetForGeneration,
)
from unitorch.cli.models.xprophetnet.processing import (
    XProphetNetProcessor as CoreXProphetNetProcessor,
)


class XProphetNetTest(parameterized.TestCase):
    def setUp(self):
        # Set the random seed for reproducibility
        set_seed(42)

        # Load the configuration file
        config_path = cached_path("examples/configs/generation/xprophetnet.ini")
        self.config = CoreConfigureParser(config_path)

        # Define the paths for cached files
        self.config_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
        )
        self.vocab_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer"
        )
        self.weight_path = cached_path(
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/pytorch_model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init xprophetnet from core configure",
            "encode": "test text for xprophetnet model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_config_init(
        self,
        encode: str,
        decode: str,
        max_gen_seq_length: Optional[int] = 10,
    ):
        # Initialize XProphetNet model from core configure
        model = CoreXProphetNetForGeneration.from_core_configure(self.config)
        process = CoreXProphetNetProcessor.from_core_configure(self.config)

        # Load the pretrained weights
        model.from_pretrained(self.weight_path)
        model.eval()

        # Tokenize and encode the input text
        inputs = process._generation_inputs(encode).input_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate output sequences
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated sequences
        results = process._detokenize(outputs).to_pandas()["decoded"].tolist()

        # Check if the decoded result matches the expected output
        assert results[0] == decode

    @parameterized.named_parameters(
        {
            "testcase_name": "init xprophetnet from package",
            "encode": "test text for xprophetnet model",
            "max_gen_seq_length": 5,
            "decode": "",
        }
    )
    def test_package_init(
        self,
        encode: str,
        decode: str,
        max_gen_seq_length: Optional[int] = 10,
    ):
        # Initialize XProphetNet model from package
        model = XProphetNetForGeneration(self.config_path)
        process = XProphetNetProcessor(self.vocab_path)

        # Load the pretrained weights
        model.from_pretrained(self.weight_path)
        model.eval()

        # Tokenize and encode the input text
        inputs = process.generation_inputs(encode).input_ids.unsqueeze(0)
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate output sequences
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated sequences
        results = process.detokenize(outputs.sequences)

        # Check if the decoded result matches the expected output
        assert results[0] == decode
