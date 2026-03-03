"""BatchLoader: iterator-based batching over any Dataset."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

from src.dataset import Dataset
from src.utils import check_type


class BatchLoader:
    """Wraps a dataset and yields batches of data via iteration.

    Batches are built from indices only; the actual data is loaded from the
    dataset (which may itself be lazy) only when each batch is consumed.

    Args:
        dataset: Any ``Dataset`` instance (labeled or unlabeled).
        batch_size: Number of samples per batch.
        shuffle: If ``True``, indices are randomly shuffled before batching
            at the start of every iteration. If ``False``, data is returned
            in its original order.
        drop_last: If ``True``, the last batch is discarded when it contains
            fewer than ``batch_size`` samples. If ``False`` (default), the
            last batch is kept even if it is smaller.

    Raises:
        TypeError: If any argument has an unexpected type.
        ValueError: If ``batch_size`` is less than 1.

    Example::

        loader = BatchLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            # batch is a list of (data, label) tuples or data items
            ...
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        check_type(dataset, Dataset, "dataset")
        check_type(batch_size, int, "batch_size")
        check_type(shuffle, bool, "shuffle")
        check_type(drop_last, bool, "drop_last")
        if batch_size < 1:
            raise ValueError(f"'batch_size' must be at least 1, got {batch_size}.")

        self._dataset: Dataset = dataset
        self._batch_size: int = batch_size
        self._shuffle: bool = shuffle
        self._drop_last: bool = drop_last

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[Any]]:
        """Yield batches of data, loading each item only when reached.

        Yields:
            A list of items returned by ``dataset[i]`` — either raw data
            (unlabeled) or ``(data, label)`` tuples (labeled).
        """
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self._batch_size):
            batch_indices = indices[start : start + self._batch_size]
            if self._drop_last and len(batch_indices) < self._batch_size:
                return
            yield [self._dataset[i] for i in batch_indices]

    # ------------------------------------------------------------------
    # Sized protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of batches produced per iteration.

        Accounts for ``drop_last``: if the last batch is incomplete and
        ``drop_last`` is ``True``, it is not counted.
        """
        n = len(self._dataset)
        if self._drop_last:
            return n // self._batch_size
        return math.ceil(n / self._batch_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> Dataset:
        """The underlying dataset."""
        return self._dataset

    @property
    def batch_size(self) -> int:
        """Number of samples per batch."""
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """Whether indices are shuffled at the start of each iteration."""
        return self._shuffle

    @property
    def drop_last(self) -> bool:
        """Whether the last incomplete batch is dropped."""
        return self._drop_last
