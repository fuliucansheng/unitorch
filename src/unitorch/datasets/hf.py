# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset, IterableDataset
from datasets.features import Features
from itertools import cycle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class HFDatasets(TorchDataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: Optional[str] = "\t",
        quoting: Optional[int] = 3,
        escapechar: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "csv",
            data_dir=data_dir,
            data_files=data_files,
            delimiter=sep,
            column_names=names,
            quoting=quoting,
            escapechar=escapechar,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
        features: Optional[Features] = None,
    ) -> Optional["HFDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
            features=features,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_hub(
        cls,
        data_name: str,
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ) -> Optional["HFDatasets"]:
        dataset = load_dataset(
            data_name,
            name=config_name,
            data_dir=data_dir,
            data_files=data_files,
        )

        if split in dataset:
            dataset = dataset.get(split)
            return cls(dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class HFIterableDatasets(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
    ):
        self.dataset = dataset

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: Optional[str] = "\t",
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "csv",
            data_dir=data_dir,
            data_files=data_files,
            delimiter=sep,
            column_names=names,
            quoting=3,
            streaming=True,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
            streaming=True,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if data_files is None:
            return None

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return None

        dataset = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
            streaming=True,
        )

        if split not in dataset:
            dataset = dataset.get("train")
        else:
            dataset = dataset.get(split)

        return cls(dataset)

    @classmethod
    def from_hub(
        cls,
        data_name: str,
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        dataset = load_dataset(
            data_name,
            name=config_name,
            data_dir=data_dir,
            data_files=data_files,
            streaming=True,
        )

        if split in dataset:
            dataset = dataset.get(split)
            return cls(dataset)

    def __iter__(self) -> Iterable:
        return iter(cycle(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)
