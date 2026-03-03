"""Abstract base classes for the dataset hierarchy.

Hierarchy
---------
Dataset (ABC)
├── LabeledDataset (ABC)
└── UnlabeledDataset (ABC)

Concrete subclasses live in image_dataset.py and audio_dataset.py.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from src.utils import check_range, check_type


class Dataset(ABC):
    """Abstract base class for all datasets.

    Subclasses must implement :meth:`_scan_files`, :meth:`_load_file`,
    and :meth:`__getitem__`.

    Attributes (private):
        _root: Root folder path where data files are stored.
        _lazy: Whether to load data lazily (on access) or eagerly (at init).
        _file_paths: List of absolute file paths discovered by :meth:`_scan_files`.
        _data: List of pre-loaded data objects when *eager*; ``None`` when lazy.
    """

    def __init__(self, root: str, lazy: bool = True) -> None:
        """Initialise the dataset.

        Args:
            root: Path to the root folder containing the data files.
            lazy: If ``True`` (default), data is loaded on demand.
                  If ``False``, all data is loaded into memory immediately.

        Raises:
            TypeError: If *root* is not a ``str`` or *lazy* is not a ``bool``.
        """
        check_type(root, str, "root")
        check_type(lazy, bool, "lazy")
        self._root: str = root
        self._lazy: bool = lazy
        self._file_paths: list[str] = []
        self._data: list[Any] | None = None

        self._scan_files()
        if not self._lazy:
            self._data = [self._load_file(p) for p in self._file_paths]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _scan_files(self) -> None:
        """Populate ``self._file_paths`` with the paths to every data file."""

    @abstractmethod
    def _load_file(self, path: str) -> Any:
        """Load a single data point from *path* and return it."""

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return the data point (and label, if applicable) at *index*."""

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self._file_paths)

    def split(self, train_ratio: float) -> tuple[Dataset, Dataset]:
        """Split the dataset into training and test subsets.

        The dataset is shuffled randomly before splitting so that the
        distribution of examples is approximately balanced in both subsets.

        Args:
            train_ratio: Fraction of data points to include in the training
                set. Must be in the open interval (0, 1).

        Returns:
            A ``(train_dataset, test_dataset)`` tuple, each of the same
            concrete class as ``self``.

        Raises:
            TypeError: If *train_ratio* is not a ``float``.
            ValueError: If *train_ratio* is not strictly between 0 and 1.
        """
        check_type(train_ratio, float, "train_ratio")
        check_range(train_ratio, 0.0, 1.0, "train_ratio")

        indices = list(range(len(self)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        return self._create_subset(train_indices), self._create_subset(test_indices)

    def _create_subset(self, indices: list[int]) -> Dataset:
        """Create a new dataset instance containing only the specified indices.

        Uses :meth:`object.__new__` to bypass ``__init__`` and then calls
        :meth:`_init_subset` so that subclasses can extend the copy logic.

        Args:
            indices: Positions of the data points to include.

        Returns:
            A new dataset of the same concrete class.
        """
        new_ds: Dataset = object.__new__(type(self))
        self._init_subset(new_ds, indices)
        return new_ds

    def _init_subset(self, new_ds: Dataset, indices: list[int]) -> None:
        """Populate *new_ds* with the data corresponding to *indices*.

        Subclasses that add extra per-item attributes (e.g. ``_labels``)
        should call ``super()._init_subset(new_ds, indices)`` and then
        slice their own attributes accordingly.

        Args:
            new_ds: The freshly-created (uninitialised) dataset object.
            indices: Positions of the data points to copy.
        """
        new_ds._root = self._root
        new_ds._lazy = self._lazy
        new_ds._file_paths = [self._file_paths[i] for i in indices]
        new_ds._data = (
            [self._data[i] for i in indices] if self._data is not None else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> str:
        """Root folder path."""
        return self._root

    @property
    def lazy(self) -> bool:
        """Whether the dataset uses lazy loading."""
        return self._lazy


class LabeledDataset(Dataset, ABC):
    """Abstract base class for datasets that carry per-sample labels.

    Adds a ``_labels`` list that is populated by :meth:`_load_labels` and
    kept parallel to ``_file_paths``.

    Concrete subclasses must still implement :meth:`_scan_files` and
    :meth:`_load_file`.  They should call ``_load_labels()`` **after**
    ``super().__init__()`` (which calls ``_scan_files``).
    """

    def __init__(self, root: str, lazy: bool = True) -> None:
        self._labels: list[Any] = []
        super().__init__(root, lazy)
        # _scan_files() has been called; now populate labels.
        self._load_labels()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_labels(self) -> None:
        """Populate ``self._labels`` with one label per file path."""

    # ------------------------------------------------------------------
    # Concrete implementation of __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return ``(data, label)`` for the data point at *index*.

        Args:
            index: Zero-based index of the data point.

        Returns:
            A ``(data, label)`` tuple.

        Raises:
            IndexError: If *index* is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} is out of range for dataset of size {len(self)}."
            )
        data = (
            self._load_file(self._file_paths[index])
            if self._lazy
            else self._data[index]  # type: ignore[index]
        )
        return data, self._labels[index]

    def _init_subset(self, new_ds: Dataset, indices: list[int]) -> None:
        """Extend the base copy logic to also slice ``_labels``."""
        super()._init_subset(new_ds, indices)
        # new_ds is guaranteed to be a LabeledDataset subclass.
        new_ds._labels = [self._labels[i] for i in indices]  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[Any]:
        """Per-sample labels, parallel to the file-path list."""
        return self._labels


class UnlabeledDataset(Dataset, ABC):
    """Abstract base class for datasets without labels.

    Provides a concrete :meth:`__getitem__` that returns only the data.
    """

    def __getitem__(self, index: int) -> Any:
        """Return the data point at *index* (no label).

        Args:
            index: Zero-based index of the data point.

        Returns:
            The loaded data object.

        Raises:
            IndexError: If *index* is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} is out of range for dataset of size {len(self)}."
            )
        if self._lazy:
            return self._load_file(self._file_paths[index])
        return self._data[index]  # type: ignore[index]
