# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import math
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler, T_co


def get_local_rank() -> int:
    """
    Get the local rank from the environment variable 'LOCAL_RANK'.

    Returns:
        int: Local rank or -1 if not found.
    """
    return int(os.environ.get("LOCAL_RANK", -1))


class DistributedSkipSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        skip_step: Optional[int] = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.skip_step = skip_step

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[self.skip_step :])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


class RandomSkipSampler(RandomSampler):
    def __init__(
        self,
        data_source: Dataset,
        replacement: Optional[bool] = False,
        num_samples: Optional[int] = None,
        skip_step: Optional[int] = 0,
    ):
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )
        self.skip_step = skip_step

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n,
                    size=(self.num_samples,),
                    dtype=torch.int64,
                ).tolist()[self.skip_step :]
            )
        return iter(torch.randperm(n).tolist()[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


class SequentialSkipSampler(SequentialSampler):
    def __init__(
        self,
        data_source: Dataset,
        skip_step: Optional[int] = 0,
    ):
        super().__init__(
            data_source=data_source,
        )
        self.skip_step = skip_step

    def __iter__(self):
        return iter(range(len(self.data_source))[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step
