"""Concrete audio dataset classes.

Classes
-------
AudioDataset          – labeled audio files (classification or regression)
UnlabeledAudioDataset – audio files without labels
"""

from __future__ import annotations

import os
from typing import Any

import librosa
import numpy as np

from src.dataset import LabeledDataset, UnlabeledDataset
from src.utils import check_type, parse_labels_csv

# Supported audio file extensions (lowercase).
_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3"})


def _is_audio(filename: str) -> bool:
    """Return True if *filename* has a recognised audio extension."""
    return os.path.splitext(filename)[1].lower() in _AUDIO_EXTENSIONS


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return ``(waveform, sample_rate)``.

    Args:
        path: Path to the audio file (.wav or .mp3).

    Returns:
        A tuple ``(y, sr)`` where *y* is a 1-D float32 numpy array and
        *sr* is the original sampling rate of the file.
    """
    y, sr = librosa.load(path, sr=None)
    return y, int(sr)


class AudioDataset(LabeledDataset):
    """Labeled audio dataset supporting two storage layouts.

    **CSV mode** (``labels_file`` provided):
        Audio files are stored flat inside *root*. A CSV file maps each
        filename to its label (string for classification, numeric string for
        regression, e.g. BPM values).

    **Folder mode** (``labels_file`` is ``None``):
        Audio files are stored inside per-class subdirectories of *root*. The
        label of each file is the name of its parent subdirectory (e.g. the
        BallroomData genre folders).

    Args:
        root: Path to the directory containing audio files (or
            subdirectories in folder mode).
        lazy: If ``True`` (default), audio is loaded on demand; if
            ``False``, all files are loaded into memory at construction time.
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

        self._labels_file: str | None = labels_file
        super().__init__(root, lazy)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _scan_files(self) -> None:
        """Populate ``_file_paths`` with the paths to every audio file."""
        if self._labels_file is not None:
            # CSV mode: flat scan of the root directory.
            self._file_paths = [
                os.path.join(self._root, fname)
                for fname in sorted(os.listdir(self._root))
                if _is_audio(fname)
            ]
        else:
            # Folder mode: one subdirectory per class.
            paths: list[str] = []
            for class_dir in sorted(os.listdir(self._root)):
                class_path = os.path.join(self._root, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for fname in sorted(os.listdir(class_path)):
                    if _is_audio(fname):
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

    def _load_file(self, path: str) -> tuple[np.ndarray, int]:
        """Load an audio file and return ``(waveform, sample_rate)``.

        Args:
            path: Path to the audio file.

        Returns:
            A ``(y, sr)`` tuple compatible with librosa processing functions.
        """
        return _load_audio(path)


class UnlabeledAudioDataset(UnlabeledDataset):
    """Audio dataset without labels (flat folder, no CSV required).

    Args:
        root: Path to the directory containing audio files.
        lazy: If ``True`` (default), audio is loaded on demand.

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
        """Populate ``_file_paths`` with paths to audio files in *root*."""
        self._file_paths = [
            os.path.join(self._root, fname)
            for fname in sorted(os.listdir(self._root))
            if _is_audio(fname)
        ]

    def _load_file(self, path: str) -> tuple[np.ndarray, int]:
        """Load an audio file and return ``(waveform, sample_rate)``.

        Args:
            path: Path to the audio file.

        Returns:
            A ``(y, sr)`` tuple.
        """
        return _load_audio(path)
