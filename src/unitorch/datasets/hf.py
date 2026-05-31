# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from itertools import cycle
from typing import Iterator, List, Optional, Union

from datasets import Dataset, load_dataset
from datasets.features import Features
from torch.utils.data import Dataset as TorchDataset, IterableDataset


def _select_split(dataset, split: Optional[str]):
    """Return the requested split, falling back to 'train' if not found."""
    if split in dataset:
        return dataset[split]
    return dataset["train"]


def _data_files_exist(data_files: Optional[Union[str, List[str]]]) -> bool:
    """Return False if data_files is None or a single non-existent path."""
    if data_files is None:
        return False
    if isinstance(data_files, str) and not os.path.exists(data_files):
        return False
    return True


class HFDatasets(TorchDataset):
    """Map-style wrapper around a HuggingFace :class:`datasets.Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: str = "\t",
        quoting: int = 3,
        escapechar: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFDatasets"]:
        if not _data_files_exist(data_files):
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
        return cls(_select_split(dataset, split))

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFDatasets"]:
        if not _data_files_exist(data_files):
            return None
        dataset = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
        )
        return cls(_select_split(dataset, split))

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
        features: Optional[Features] = None,
    ) -> Optional["HFDatasets"]:
        if not _data_files_exist(data_files):
            return None
        dataset = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
            features=features,
        )
        return cls(_select_split(dataset, split))

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
        if split not in dataset:
            return None
        return cls(dataset[split])

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class HFIterableDatasets(IterableDataset):
    """Iterable wrapper around a HuggingFace streaming dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: str = "\t",
        quoting: int = 3,
        escapechar: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if not _data_files_exist(data_files):
            return None
        dataset = load_dataset(
            "csv",
            data_dir=data_dir,
            data_files=data_files,
            delimiter=sep,
            column_names=names,
            quoting=quoting,
            escapechar=escapechar,
            streaming=True,
        )
        return cls(_select_split(dataset, split))

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if not _data_files_exist(data_files):
            return None
        dataset = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
            streaming=True,
        )
        return cls(_select_split(dataset, split))

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ) -> Optional["HFIterableDatasets"]:
        if not _data_files_exist(data_files):
            return None
        dataset = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
            streaming=True,
        )
        return cls(_select_split(dataset, split))

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
        if split not in dataset:
            return None
        return cls(dataset[split])

    def __iter__(self) -> Iterator:
        return iter(cycle(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)
