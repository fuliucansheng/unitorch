# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import ast
import torch
import datasets
import torch.distributed as dist
from copy import deepcopy
from datasets import Dataset
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.datasets.hf import HFDatasets, HFIterableDatasets
from unitorch.cli import CoreConfigureParser
from unitorch.cli import cached_path, registered_process, register_dataset
from unitorch.cli import add_default_section_for_init, add_default_section_for_function
from unitorch.cli import init_registered_process
from unitorch.cli.models import (
    ModelInputs,
    ModelTargets,
    CombineTensorsInputs,
    CombineTensorsTargets,
)


class ASTFunction:
    def __init__(self, func: str):
        for name in registered_process.keys():
            _name = f"{name}("
            func = func.replace(_name, _name.replace("/", "_"))

        registered_process_mapping = {
            k.replace("/", "_"): k for k, v in registered_process.items()
        }
        self.func = func
        self.__ast_func__ = ast.parse(func, "", mode="eval")
        self.__ast_keys__ = []
        self.__ast_process__ = []
        for node in ast.walk(self.__ast_func__):
            if isinstance(node, ast.Name):
                if node.id in registered_process_mapping:
                    self.__ast_process__.append(node.id)
                else:
                    self.__ast_keys__.append(node.id)
        self.__ast_func__ = compile(self.__ast_func__, "", "eval")

    def process(self, row: Dict):
        for key in self.__ast_keys__:
            if key not in row:
                continue
            locals()[key] = row.get(key, None)

        return eval(deepcopy(self.__ast_func__))


class ASTHFDatasets(HFDatasets):
    def __init__(
        self,
        dataset: Dataset,
        process_functions: List[ASTFunction],
    ):
        super().__init__(dataset=dataset)
        self.process_functions = deepcopy(process_functions)

    def __getitem__(self, idx):
        model_inputs_list, model_targets_list = [], []
        row = self.dataset[idx]
        for pfunc in self.process_functions:
            results = pfunc.process(row)
            if isinstance(results, ModelInputs) or isinstance(results, ModelTargets):
                results = [results]

            for result in results:
                if isinstance(result, ModelInputs):
                    is_new = True
                    for _inputs in model_inputs_list:
                        if type(_inputs) == type(result):
                            _inputs.add(result)
                            is_new = False
                            break

                    if is_new:
                        model_inputs_list.append(result)

                if isinstance(result, ModelTargets):
                    is_new = True
                    for _targets in model_targets_list:
                        if type(_targets) == type(result):
                            _targets.add(result)
                            is_new = False
                            break

                    if is_new:
                        model_targets_list.append(result)
        if len(model_inputs_list) == 0:
            model_inputs_list = CombineTensorsInputs()
        elif len(model_inputs_list) == 1:
            model_inputs_list = model_inputs_list[0]

        if len(model_targets_list) == 0:
            model_targets_list = CombineTensorsTargets()
        elif len(model_targets_list) == 1:
            model_targets_list = model_targets_list[0]

        return model_inputs_list, model_targets_list

    def __len__(
        self,
    ):
        return len(self.dataset)


class ASTHFIterableDatasets(HFIterableDatasets):
    def __init__(
        self,
        dataset: Dataset,
        process_functions: List[ASTFunction],
        enable_ddp_partition: Optional[bool] = True,
    ):
        super().__init__(dataset=dataset)
        self.process_functions = deepcopy(process_functions)
        self.dataset.shuffle(10000)
        if enable_ddp_partition and dist.is_initialized():
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.global_rank = 0
            self.world_size = 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        mod = self.world_size
        shift = self.global_rank
        if worker_info:
            mod *= worker_info.num_workers
            shift = self.global_rank * worker_info.num_workers + worker_info.id
        for i, row in enumerate(self.dataset):
            if (i + shift) % mod != 0:
                continue
            inputs, targets = None, None
            for pfunc in self.process_functions:
                results = pfunc.process(row)

                if isinstance(results, ModelInputs) or isinstance(
                    results, ModelTargets
                ):
                    results = [results]

                for result in results:
                    if isinstance(result, ModelInputs):
                        if inputs is None:
                            inputs = result
                        else:
                            inputs.add(result)

                    if isinstance(result, ModelTargets):
                        if targets is None:
                            targets = result
                        else:
                            targets.add(result)
            if inputs is None:
                inputs = CombineTensorsInputs()

            if targets is None:
                targets = CombineTensorsTargets()

            yield inputs, targets

    def set_skip_step(self, step):
        self.__dataset__ = self.__dataset__.skip(step * self.world_size)


@register_dataset("core/dataset/ast")
class ASTDatasets:
    """Class for managing AST datasets."""

    splits = ["train", "dev", "test"]
    templates = ["csv", "json", "parquet", "hub"]
    __ASTDatasets__ = dict()

    def __init__(self, configure: CoreConfigureParser):
        """
        Initialize ASTDatasets.

        Args:
            configure (CoreConfigureParser): The configuration parser for AST datasets.
        """
        self.config = configure

    def __getdataset__(self, split):
        """
        Get the dataset for the specified split.

        Args:
            split (str): The split to get the dataset for.

        Returns:
            dataset: The dataset for the specified split.
        """
        config = self.config

        registered_process_mapping = {
            k.replace("/", "_"): k for k, v in registered_process.items()
        }

        config.set_default_section(f"core/dataset/ast")
        _iterable = config.getoption("iterable", False)
        _template = config.getoption("template", "csv")
        _data_name = config.getoption("data_name", None)
        _config_name = config.getoption("config_name", None)
        _data_dir = config.getoption("data_dir", None)
        _data_files = config.getoption("data_files", None)
        _names = config.getoption("names", None)
        _features = config.getoption("features", None)
        _sep = config.getoption("sep", "\t")
        _quoting = config.getoption("quoting", 3)
        _escapechar = config.getoption("escapechar", None)
        _field = config.getoption("field", None)
        _process_functions = config.getoption("preprocess_functions", None)
        _enable_ddp_partition = config.getoption("enable_ddp_partition", True)

        _HFDatasets = HFIterableDatasets if _iterable else HFDatasets
        _ASTDatasets = ASTHFIterableDatasets if _iterable else ASTHFDatasets

        config.set_default_section(f"core/dataset/ast/{split}")

        template = config.getoption("template", _template)
        if config.getoption("data_name", _data_name) is not None:
            template = "hub"

        assert template in self.templates

        new_split = "validation" if split == "dev" else split
        new_split = config.getoption("split", new_split)

        # get dataset
        dataset = None
        if template == "csv":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            names = config.getoption("names", _names)
            sep = config.getoption("sep", _sep)
            quoting = config.getoption("quoting", _quoting)
            escapechar = config.getoption("escapechar", _escapechar)
            dataset = _HFDatasets.from_csv(
                data_dir=data_dir,
                data_files=data_files,
                names=names,
                sep=sep,
                quoting=quoting,
                escapechar=escapechar,
                split=new_split,
            )

        if template == "json":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            field = config.getoption("field", _field)

            dataset = _HFDatasets.from_json(
                data_dir=data_dir,
                data_files=data_files,
                field=field,
                split=new_split,
            )

        if template == "parquet":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            features = config.getoption("features", _features)
            if isinstance(features, str):
                features = eval(features)
            dataset = _HFDatasets.from_parquet(
                data_dir=data_dir,
                data_files=data_files,
                split=new_split,
                features=features,
            )

        if template == "hub":
            data_name = config.getoption("data_name", _data_name)
            config_name = config.getoption("config_name", _config_name)
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            data_name = (
                cached_path(data_name) if data_name.endswith(".py") else data_name
            )
            dataset = _HFDatasets.from_hub(
                data_name=data_name,
                config_name=config_name,
                data_dir=data_dir,
                data_files=data_files,
                split=new_split,
            )

        assert dataset is not None

        # get process functions
        process_functions = config.getoption("preprocess_functions", _process_functions)
        if process_functions is None:
            process_functions = []
        else:
            process_functions = [ASTFunction(func) for func in process_functions]

        for pfunc in process_functions:
            for name in pfunc.__ast_process__:
                globals()[name] = init_registered_process(
                    registered_process_mapping[name],
                    config,
                )

        enable_ddp_partition = config.getoption(
            "enable_ddp_partition", _enable_ddp_partition
        )

        if isinstance(_ASTDatasets, HFIterableDatasets):
            self.__ASTDatasets__[split] = _ASTDatasets(
                dataset=dataset.dataset,
                process_functions=process_functions,
                enable_ddp_partition=enable_ddp_partition,
            )
        else:
            self.__ASTDatasets__[split] = _ASTDatasets(
                dataset=dataset.dataset,
                process_functions=process_functions,
            )

        return self.__ASTDatasets__.get(split)

    @classmethod
    @add_default_section_for_init("core/dataset/ast")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ASTDatasets from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ASTDatasets: An instance of ASTDatasets.
        """
        return cls(configure=config)

    def get(self, split: Optional[str] = "train"):
        """
        Get the dataset for the specified split.

        Args:
            split (str, optional): The split to get the dataset for. Defaults to "train".

        Returns:
            dataset: The dataset for the specified split.
        """
        return self.__getdataset__(split)
