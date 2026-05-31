# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from itertools import islice
from typing import Callable, Optional

import datasets

datasets.logging.set_verbosity(datasets.logging.ERROR)

from datasets.iterable_dataset import _BaseExamplesIterable
from unitorch.utils.decorators import replace


@replace(datasets.iterable_dataset.ExamplesIterable)
class ExamplesIterableV2(datasets.iterable_dataset.ExamplesIterable):
    """Extended :class:`ExamplesIterable` that supports builder-level skip.

    When the underlying :class:`~datasets.GeneratorBasedBuilder` exposes a
    ``skip`` method, cheap seek-based skipping is used instead of iterating
    through discarded examples.
    """

    def __init__(
        self,
        generate_examples_fn: Callable,
        kwargs: dict,
        ex_dataset_builder: Optional[datasets.GeneratorBasedBuilder] = None,
    ) -> None:
        super().__init__(generate_examples_fn, kwargs)
        self.ex_dataset_builder = ex_dataset_builder
        self.support_skip_step = hasattr(ex_dataset_builder, "skip")

    def skip(self, n: int) -> None:
        """Delegate skip to the underlying dataset builder."""
        self.ex_dataset_builder.skip(n)


@replace(datasets.iterable_dataset.SkipExamplesIterable)
class SkipExamplesIterableV2(datasets.iterable_dataset.SkipExamplesIterable):
    """Extended :class:`SkipExamplesIterable` with fast builder-level skipping.

    When the wrapped iterable supports ``skip``, the first *n* examples are
    bypassed efficiently via the builder rather than consumed one by one.
    """

    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        n: int,
        block_sources_order_when_shuffling: bool = True,
        split_when_sharding: bool = True,
    ) -> None:
        self.ex_iterable = ex_iterable
        self.n = n
        self.block_sources_order_when_shuffling = block_sources_order_when_shuffling
        self.split_when_sharding = split_when_sharding

    def __iter__(self):
        ex_iterator = iter(self.ex_iterable)
        if (
            isinstance(self.ex_iterable, datasets.iterable_dataset.ExamplesIterable)
            and self.ex_iterable.support_skip_step
        ):
            self.ex_iterable.skip(self.n)
            self.n = 0
        else:
            # Consume and discard the first n examples.
            for _ in islice(ex_iterator, self.n):
                pass
        yield from ex_iterator

    def shuffle_data_sources(self, seed: Optional[int]) -> "SkipExamplesIterableV2":
        """Return self unchanged — shuffling would skip examples from wrong shards."""
        return self
