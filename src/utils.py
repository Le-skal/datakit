"""Utility functions shared across dataset classes."""

import csv
import numpy as np
from PIL import Image


def check_type(value: object, expected_type: type | tuple, name: str) -> None:
    """Raise TypeError if value is not an instance of expected_type.

    Args:
        value: The value to check.
        expected_type: The expected type or tuple of types.
        name: The parameter name (used in the error message).

    Raises:
        TypeError: If value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        expected = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " | ".join(t.__name__ for t in expected_type)
        )
        raise TypeError(
            f"'{name}' must be of type {expected}, got {type(value).__name__}."
        )


def check_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Raise ValueError if value is not within [min_val, max_val].

    Args:
        value: The numeric value to check.
        min_val: The minimum allowed value (inclusive).
        max_val: The maximum allowed value (inclusive).
        name: The parameter name (used in the error message).

    Raises:
        ValueError: If value is outside the allowed range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"'{name}' must be between {min_val} and {max_val}, got {value}."
        )


def parse_labels_csv(labels_file: str) -> dict[str, str]:
    """Parse a CSV file mapping filenames to labels.

    The CSV is expected to have no header row. Each row contains:
      - column 0: filename (basename only, e.g. ``image.jpg``)
      - column 1: label (string; callers may cast to int/float as needed)

    Args:
        labels_file: Absolute or relative path to the CSV file.

    Returns:
        A dict mapping each filename to its label string.

    Raises:
        FileNotFoundError: If labels_file does not exist.
        ValueError: If the CSV contains a row with fewer than two columns.
    """
    check_type(labels_file, str, "labels_file")
    mapping: dict[str, str] = {}
    with open(labels_file, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(
                    f"Row {row_num} in '{labels_file}' has fewer than 2 columns."
                )
            filename, label = row[0].strip(), row[1].strip()
            mapping[filename] = label
    return mapping


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an RGB numpy array.

    Args:
        path: Path to the image file (.jpg, .jpeg, or .png).

    Returns:
        A numpy array of shape (H, W, 3) with dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If Pillow cannot open the file.
    """
    check_type(path, str, "path")
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
    return np.array(img_rgb)
