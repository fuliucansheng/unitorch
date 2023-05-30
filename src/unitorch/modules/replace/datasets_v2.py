# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import datasets

datasets.logging.set_verbosity(datasets.logging.ERROR)
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from datasets.iterable_dataset import (
    _BaseExamplesIterable,
    SkipExamplesIterable,
    ExamplesIterable,
)
from unitorch.utils.decorators import replace


@replace(datasets.iterable_dataset.ExamplesIterable)
class ExamplesIterableV2(datasets.iterable_dataset.ExamplesIterable):
    def __init__(
        self,
        generate_examples_fn: Callable,
        kwargs: dict,
        ex_dataset_builder: Optional[datasets.GeneratorBasedBuilder] = None,
    ):
        super().__init__(generate_examples_fn, kwargs)
        self.ex_dataset_builder = ex_dataset_builder
        self.support_skip_step = hasattr(self.ex_dataset_builder, "skip")

    def __iter__(self):
        for key, example in self.generate_examples_fn(**self.kwargs):
            yield key, example

    def skip(self, n):
        self.ex_dataset_builder.skip(n)


@replace(datasets.iterable_dataset.SkipExamplesIterable)
class SkipExamplesIterableV2(datasets.iterable_dataset.SkipExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, n: int):
        self.ex_iterable = ex_iterable
        self.n = n

    def __iter__(self):
        if (
            isinstance(self.ex_iterable, ExamplesIterable)
            and self.ex_iterable.support_skip_step
        ):
            self.ex_iterable.skip(self.n)
            self.n = 0
            ex_iterator = iter(self.ex_iterable)
        else:
            ex_iterator = iter(self.ex_iterable)
            for _ in islice(ex_iterator, self.n):
                pass
        yield from ex_iterator

    def shuffle_data_sources(self, seed: Optional[int]) -> "SkipExamplesIterable":
        """Doesn't shuffle the wrapped examples iterable since it would skip exampels from other shards instead."""
        return self

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards
