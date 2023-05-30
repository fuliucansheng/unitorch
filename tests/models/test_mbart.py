# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import unitorch
import pkg_resources
from absl.testing import absltest, parameterized
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import set_seed
from unitorch.models.mbart import MBartForGeneration, MBartProcessor
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.mbart.modeling import (
    MBartForGeneration as CoreMBartForGeneration,
)
from unitorch.cli.models.mbart.processing import MBartProcessor as CoreMBartProcessor


class MBartTest(parameterized.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        set_seed(42)

        # Load the configuration from the core package
        config_path = cached_path("examples/configs/generation/mbart.ini")
        self.config = CoreConfigureParser(config_path)

        # Define paths to the required files
        self.config_path = cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json"
        )
        self.vocab_path = cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentence.bpe.model"
        )
        self.weight_path = cached_path(
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/pytorch_model.bin"
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "init mbart from core configure",
            "encode": "test text for mbart model",
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
        # Initialize the MBartForGeneration model using the configuration from the core package
        model = CoreMBartForGeneration.from_core_configure(self.config)

        # Initialize the MBartProcessor using the configuration from the core package
        process = CoreMBartProcessor.from_core_configure(self.config)

        # Load the pretrained weights into the model
        model.from_pretrained(self.weight_path)

        # Set the model to evaluation mode
        model.eval()

        # Process the input text for inference
        inputs = process._generation_inputs(encode).input_ids.unsqueeze(0)

        # Move the model and inputs to the GPU if available
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate outputs using the model
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated outputs
        results = process._detokenize(outputs).to_pandas()["decoded"].tolist()

        # Assert that the decoded result matches the expected decode value
        assert results[0] == decode

    @parameterized.named_parameters(
        {
            "testcase_name": "init mbart from package",
            "encode": "test text for mbart model",
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
        # Initialize the MBartForGeneration model using the configuration file path
        model = MBartForGeneration(self.config_path)

        # Initialize the MBartProcessor using the vocabulary file path
        process = MBartProcessor(self.vocab_path)

        # Load the pretrained weights into the model
        model.from_pretrained(self.weight_path)

        # Set the model to evaluation mode
        model.eval()

        # Process the input text for inference
        inputs = process.generation_inputs(encode).input_ids.unsqueeze(0)

        # Move the model and inputs to the GPU if available
        if torch.cuda.is_available():
            model, inputs = model.cuda(), inputs.cuda()

        # Generate outputs using the model
        outputs = model.generate(inputs, max_gen_seq_length=max_gen_seq_length)

        # Decode the generated outputs
        results = process.detokenize(outputs.sequences)

        # Assert that the decoded result matches the expected decode value
        assert results[0] == decode
