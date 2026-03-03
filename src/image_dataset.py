"""Concrete image dataset classes.

Classes
-------
ImageDataset          – labeled images (classification or regression)
UnlabeledImageDataset – images without labels
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.dataset import LabeledDataset, UnlabeledDataset
from src.utils import check_type, load_image, parse_labels_csv

# Supported image file extensions (lowercase).
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def _is_image(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_EXTENSIONS


class ImageDataset(LabeledDataset):
    """Labeled image dataset supporting two storage layouts.

    **CSV mode** (``labels_file`` provided):
        Images are stored flat inside *root*. A CSV file maps each filename
        to its label (string for classification, numeric string for regression).

    **Folder mode** (``labels_file`` is ``None``):
        Images are stored inside per-class subdirectories of *root*. The label
        of each image is the name of its parent subdirectory.

    Args:
        root: Path to the directory containing the image files (or
            subdirectories in folder mode).
        lazy: If ``True`` (default), images are loaded on demand; if
            ``False``, all images are loaded into memory at construction time.
        labels_file: Path to the CSV file (filename → label). Pass ``None``
            to use the folder-hierarchy mode.

    Raises:
        TypeError: If any argument has an unexpected type.
        FileNotFoundError: If *root* or *labels_file* does not exist.
    """

    def __init__(
        self,
        root: str,
        lazy: bool = True,
        labels_file: str | None = None,
    ) -> None:
        check_type(root, str, "root")
        check_type(lazy, bool, "lazy")
        if labels_file is not None:
            check_type(labels_file, str, "labels_file")

        # Store before super().__init__ calls _scan_files / _load_labels.
        self._labels_file: str | None = labels_file
        # _labels is initialised to [] by LabeledDataset.__init__ before
        # super().__init__ is called, so no extra initialisation needed here.
        super().__init__(root, lazy)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _scan_files(self) -> None:
        """Populate ``_file_paths`` with the paths to every image file."""
        if self._labels_file is not None:
            # CSV mode: flat scan of the root directory.
            self._file_paths = [
                os.path.join(self._root, fname)
                for fname in sorted(os.listdir(self._root))
                if _is_image(fname)
            ]
        else:
            # Folder mode: one subdirectory per class.
            paths: list[str] = []
            for class_dir in sorted(os.listdir(self._root)):
                class_path = os.path.join(self._root, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for fname in sorted(os.listdir(class_path)):
                    if _is_image(fname):
                        paths.append(os.path.join(class_path, fname))
            self._file_paths = paths

    def _load_labels(self) -> None:
        """Populate ``_labels`` from the CSV file or from directory names."""
        if self._labels_file is not None:
            mapping = parse_labels_csv(self._labels_file)
            labels: list[Any] = []
            for path in self._file_paths:
                fname = os.path.basename(path)
                raw = mapping[fname]
                # Try to cast to numeric; fall back to string for classification.
                try:
                    label: Any = int(raw)
                except ValueError:
                    try:
                        label = float(raw)
                    except ValueError:
                        label = raw
                labels.append(label)
            self._labels = labels
        else:
            # Folder mode: label = parent directory name.
            self._labels = [
                os.path.basename(os.path.dirname(p)) for p in self._file_paths
            ]

    def _load_file(self, path: str) -> np.ndarray:
        """Load an image from disk and return an RGB numpy array.

        Args:
            path: Path to the image file.

        Returns:
            Numpy array of shape (H, W, 3), dtype uint8.
        """
        return load_image(path)


class UnlabeledImageDataset(UnlabeledDataset):
    """Image dataset without labels (flat folder, no CSV required).

    Args:
        root: Path to the directory containing image files.
        lazy: If ``True`` (default), images are loaded on demand.

    Raises:
        TypeError: If any argument has an unexpected type.
    """

    def __init__(self, root: str, lazy: bool = True) -> None:
        check_type(root, str, "root")
        check_type(lazy, bool, "lazy")
        super().__init__(root, lazy)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _scan_files(self) -> None:
        """Populate ``_file_paths`` with paths to images in *root*."""
        self._file_paths = [
            os.path.join(self._root, fname)
            for fname in sorted(os.listdir(self._root))
            if _is_image(fname)
        ]

    def _load_file(self, path: str) -> np.ndarray:
        """Load an image and return an RGB numpy array.

        Args:
            path: Path to the image file.

        Returns:
            Numpy array of shape (H, W, 3), dtype uint8.
        """
        return load_image(path)
