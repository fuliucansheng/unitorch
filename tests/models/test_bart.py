# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import unitorch
import pkg_resources
from absl.testing import absltest, parameterized
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import set_seed
from unitorch.models.bart import BartForGeneration, BartProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.bart.modeling import BartForGeneration as CoreBartForGeneration
from unitorch.cli.models.bart.processing import BartProcessor as CoreBartProcessor


class BartTest(parameterized.TestCase):
    def setUp(self):
        set_seed(42)

        # Load the configuration file
        config_path = cached_path("examples/configs/generation/bart.ini")
        self.config = CoreConfigureParser(config_path)

        # Download the required files if they are not already cached
        self.config_path = cached_path(
            "https://huggingface.co/facebook/bart-base/resolve/main/config.json"
        )
        self.vocab_path = cached_path(
            "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json"
        )
        self.merge_path = cached_path(
            "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt"
        )
        self.weight_path = cached_path(
            "https://huggingface.co/facebook/bart-base/resolve/main/pytorch_model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init bart from core configure",
            "encode": "test text for bart model",
            "max_gen_seq_length": 5,
            "decode": "test text",
        }
    )
    def test_config_init(
        self,
        encode: str,
        decode: str,
        max_gen_seq_length: Optional[int] = 10,
    ):
        # Initialize the BART model from the core configuration
        model = CoreBartForGeneration.from_core_configure(self.config)

        # Initialize the BART processor from the core configuration
        process = CoreBartProcessor.from_core_configure(self.config)

        # Load the pre-trained weights for the model
        model.from_pretrained(self.weight_path)
        model.eval()

        # Process the input text for inference
        inputs = process._generation_inputs(encode).input_ids.unsqueeze(0)

        # Move the model and inputs to GPU if available
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate output sequences using the BART model
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated sequences
        results = process._detokenize(outputs).to_pandas()["decoded"].tolist()

        # Assert that the decoded result matches the expected output
        assert results[0] == decode

    @parameterized.named_parameters(
        {
            "testcase_name": "init bart from package",
            "encode": "test text for bart model",
            "max_gen_seq_length": 5,
            "decode": "test text",
        }
    )
    def test_package_init(
        self,
        encode: str,
        decode: str,
        max_gen_seq_length: Optional[int] = 10,
    ):
        # Initialize the BART model from the package
        model = BartForGeneration(self.config_path)

        # Initialize the BART processor from the package
        process = BartProcessor(self.vocab_path, self.merge_path)

        # Load the pre-trained weights for the model
        model.from_pretrained(self.weight_path)
        model.eval()

        # Process the input text for inference
        inputs = process.generation_inputs(encode).input_ids.unsqueeze(0)

        # Move the model and inputs to GPU if available
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate output sequences using the BART model
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated sequences
        results = process.detokenize(outputs.sequences)

        # Assert that the decoded result matches the expected output
        assert results[0] == decode
