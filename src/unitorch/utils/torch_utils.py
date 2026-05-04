# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import math
import os
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def get_local_rank() -> int:
    """Return the local rank from ``LOCAL_RANK``, or ``-1`` if not set."""
    return int(os.environ.get("LOCAL_RANK", -1))


class DistributedSkipSampler(DistributedSampler):
    """A :class:`DistributedSampler` that skips the first *skip_step* indices.

    Useful for resuming distributed training mid-epoch without re-processing
    already-seen samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        skip_step: int = 0,
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

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

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
    """A :class:`RandomSampler` that skips the first *skip_step* indices.

    Useful for resuming single-process training mid-epoch.
    """

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        skip_step: int = 0,
    ) -> None:
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )
        self.skip_step = skip_step

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            indices = torch.randint(
                high=n, size=(self.num_samples,), dtype=torch.int64
            ).tolist()
        else:
            indices = torch.randperm(n).tolist()
        return iter(indices[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step


class SequentialSkipSampler(SequentialSampler):
    """A :class:`SequentialSampler` that skips the first *skip_step* indices.

    Useful for resuming inference or evaluation mid-dataset.
    """

    def __init__(
        self,
        data_source: Dataset,
        skip_step: int = 0,
    ) -> None:
        super().__init__(data_source=data_source)
        self.skip_step = skip_step

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source))[self.skip_step :])

    def set_skip_step(self, step: int) -> None:
        self.skip_step = step
