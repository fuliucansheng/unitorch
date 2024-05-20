# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import unitorch
import importlib_resources
from absl.testing import absltest, parameterized
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
)


class SimpleClass:
    def __init__(
        self,
        param1=1,
        param2=2,
        param3=3,
    ):
        """
        Initialize SimpleClass with three parameters.

        Args:
            param1 (int): Parameter 1 (default: 1).
            param2 (int): Parameter 2 (default: 2).
            param3 (int): Parameter 3 (default: 3).
        """
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    @classmethod
    @add_default_section_for_init("simple_class")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of SimpleClass using configuration.

        Args:
            config (CoreConfigureParser): Configuration parser object.
            **kwargs: Additional keyword arguments.

        Returns:
            SimpleClass: An instance of SimpleClass.
        """
        pass

    @add_default_section_for_function("simple_class")
    def inst_function(
        self,
        param4=4,
        param5="param5",
    ):
        """
        Perform an instance function with optional parameters.

        Args:
            param4 (int): Parameter 4 (default: 4).
            param5 (str): Parameter 5 (default: "param5").
        """
        self.param4 = param4
        self.param5 = param5


class DecoratorsTest(parameterized.TestCase):
    def setUp(self):
        self.config = CoreConfigureParser(
            params=[
                ["simple_class", "param1", 2],
                ["simple_class", "param2", 3],
                ["simple_class", "param5", "param6"],
            ]
        )

    def test_default_for_init(self):
        """
        Test if default values are correctly set during initialization.
        """
        inst = SimpleClass.from_core_configure(self.config)

        assert inst.param1 == 2 and inst.param2 == 3 and inst.param3 == 3

    def test_default_for_function(self):
        """
        Test if default values are correctly set during function call.
        """
        inst = SimpleClass.from_core_configure(self.config)
        inst.inst_function()

        assert inst.param4 == 4 and inst.param5 == "param6"
