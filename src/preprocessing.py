"""Preprocessing transforms and pipeline.

All transforms are callable classes: hyperparameters at construction time,
data-only at call time.

Class hierarchy
---------------
Transform (ABC)
├── CenterCrop          image
├── RandomCrop          image
├── RandomFlip          image
├── Padding             image
├── MelSpectrogram      audio  (y, sr) → np.ndarray
├── AudioRandomCrop     audio  (y, sr) → (y, sr)
├── Resample            audio  (y, sr) → (y, sr)
├── PitchShift          audio  (y, sr) → (y, sr)
└── Pipeline            any    chains transforms sequentially
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import librosa
import numpy as np

from src.utils import check_range, check_type


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Transform(ABC):
    """Abstract base class for all preprocessing transforms.

    Every transform is callable: hyperparameters are fixed at construction,
    and :meth:`__call__` receives only the data to transform.
    """

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply the transform.

        Args:
            data: Input data.

        Returns:
            Transformed data.
        """


# ---------------------------------------------------------------------------
# Image transforms  (input/output: np.ndarray of shape (H, W, 3))
# ---------------------------------------------------------------------------


class CenterCrop(Transform):
    """Crop an image around its center to at most ``height`` × ``width``.

    A dimension is only cropped when the image is strictly larger than the
    target in that dimension; smaller dimensions are left unchanged.

    Args:
        height: Maximum output height in pixels.
        width: Maximum output width in pixels.

    Raises:
        TypeError: If ``height`` or ``width`` are not ``int``.
        ValueError: If either dimension is less than 1.
    """

    def __init__(self, height: int, width: int) -> None:
        check_type(height, int, "height")
        check_type(width, int, "width")
        if height < 1:
            raise ValueError(f"'height' must be ≥ 1, got {height}.")
        if width < 1:
            raise ValueError(f"'width' must be ≥ 1, got {width}.")
        self._height = height
        self._width = width

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply center crop.

        Args:
            data: Image array of shape (H, W, 3).

        Returns:
            Cropped image array.
        """
        check_type(data, np.ndarray, "data")
        h, w = data.shape[:2]
        if h > self._height:
            top = (h - self._height) // 2
            data = data[top : top + self._height, :]
        if w > self._width:
            left = (w - self._width) // 2
            data = data[:, left : left + self._width]
        return data

    @property
    def height(self) -> int:
        """Target height."""
        return self._height

    @property
    def width(self) -> int:
        """Target width."""
        return self._width


class RandomCrop(Transform):
    """Crop an image at a random position to at most ``height`` × ``width``.

    A dimension is only cropped when the image is strictly larger than the
    target in that dimension; smaller dimensions are left unchanged.

    Args:
        height: Maximum output height in pixels.
        width: Maximum output width in pixels.

    Raises:
        TypeError: If ``height`` or ``width`` are not ``int``.
        ValueError: If either dimension is less than 1.
    """

    def __init__(self, height: int, width: int) -> None:
        check_type(height, int, "height")
        check_type(width, int, "width")
        if height < 1:
            raise ValueError(f"'height' must be ≥ 1, got {height}.")
        if width < 1:
            raise ValueError(f"'width' must be ≥ 1, got {width}.")
        self._height = height
        self._width = width

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random crop.

        Args:
            data: Image array of shape (H, W, 3).

        Returns:
            Cropped image array.
        """
        check_type(data, np.ndarray, "data")
        h, w = data.shape[:2]
        if h > self._height:
            top = random.randint(0, h - self._height)
            data = data[top : top + self._height, :]
        if w > self._width:
            left = random.randint(0, w - self._width)
            data = data[:, left : left + self._width]
        return data

    @property
    def height(self) -> int:
        """Target height."""
        return self._height

    @property
    def width(self) -> int:
        """Target width."""
        return self._width


class RandomFlip(Transform):
    """Randomly flip an image along its horizontal and/or vertical axis.

    Each axis is flipped independently with probability ``p``.

    Args:
        p: Probability of flipping along each axis. Must be in ``[0, 1]``.

    Raises:
        TypeError: If ``p`` is not a ``float``.
        ValueError: If ``p`` is outside ``[0, 1]``.
    """

    def __init__(self, p: float = 0.5) -> None:
        check_type(p, float, "p")
        check_range(p, 0.0, 1.0, "p")
        self._p = p

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply random flip.

        Args:
            data: Image array of shape (H, W, 3).

        Returns:
            Flipped image array (a copy).
        """
        check_type(data, np.ndarray, "data")
        if random.random() < self._p:
            data = np.flip(data, axis=0)  # vertical flip
        if random.random() < self._p:
            data = np.flip(data, axis=1)  # horizontal flip
        return np.ascontiguousarray(data)

    @property
    def p(self) -> float:
        """Flip probability per axis."""
        return self._p


class Padding(Transform):
    """Pad an image to at least ``height`` × ``width`` using a solid colour.

    Padding is added symmetrically. Dimensions that already meet or exceed
    the target are not modified.

    Args:
        height: Minimum output height in pixels.
        width: Minimum output width in pixels.
        color: RGB fill colour as an ``(R, G, B)`` int tuple. Default black.

    Raises:
        TypeError: If ``height`` or ``width`` are not ``int``, or ``color``
            is not a ``tuple``.
        ValueError: If either dimension is less than 1.
    """

    def __init__(
        self,
        height: int,
        width: int,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        check_type(height, int, "height")
        check_type(width, int, "width")
        check_type(color, tuple, "color")
        if height < 1:
            raise ValueError(f"'height' must be ≥ 1, got {height}.")
        if width < 1:
            raise ValueError(f"'width' must be ≥ 1, got {width}.")
        self._height = height
        self._width = width
        self._color = color

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply padding.

        Args:
            data: Image array of shape (H, W, 3).

        Returns:
            Padded image array of shape (max(H, height), max(W, width), 3).
        """
        check_type(data, np.ndarray, "data")
        h, w = data.shape[:2]
        out_h = max(h, self._height)
        out_w = max(w, self._width)

        if out_h == h and out_w == w:
            return data

        # Create a canvas pre-filled with the target colour, then paste image.
        canvas = np.full((out_h, out_w, 3), self._color, dtype=data.dtype)
        top = (out_h - h) // 2
        left = (out_w - w) // 2
        canvas[top : top + h, left : left + w] = data
        return canvas

    @property
    def height(self) -> int:
        """Minimum output height."""
        return self._height

    @property
    def width(self) -> int:
        """Minimum output width."""
        return self._width

    @property
    def color(self) -> tuple[int, int, int]:
        """RGB fill colour."""
        return self._color


# ---------------------------------------------------------------------------
# Audio transforms  (input: (np.ndarray, int) = (waveform, sample_rate))
# ---------------------------------------------------------------------------


class MelSpectrogram(Transform):
    """Convert a waveform to a Mel spectrogram.

    This transform changes the data type: input is ``(y, sr)``, output is a
    2-D ``np.ndarray``. Audio-specific transforms (e.g. :class:`Resample`)
    cannot be chained after this one.

    Args:
        n_mels: Number of Mel frequency bands.
        n_fft: FFT window size in samples.
        hop_length: Hop length in samples between successive frames.

    Raises:
        TypeError: If any argument is not an ``int``.
    """

    def __init__(
        self, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512
    ) -> None:
        check_type(n_mels, int, "n_mels")
        check_type(n_fft, int, "n_fft")
        check_type(hop_length, int, "hop_length")
        self._n_mels = n_mels
        self._n_fft = n_fft
        self._hop_length = hop_length

    def __call__(self, data: tuple[np.ndarray, int]) -> np.ndarray:
        """Compute the Mel spectrogram.

        Args:
            data: ``(waveform, sample_rate)`` tuple.

        Returns:
            Mel spectrogram array of shape ``(n_mels, T)``.
        """
        check_type(data, tuple, "data")
        y, sr = data
        return librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self._n_mels, n_fft=self._n_fft, hop_length=self._hop_length
        )

    @property
    def n_mels(self) -> int:
        return self._n_mels

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def hop_length(self) -> int:
        return self._hop_length


class AudioRandomCrop(Transform):
    """Randomly crop an audio track to a fixed duration.

    If the track is shorter than or equal to ``duration`` seconds, the
    original track is returned unchanged.

    Args:
        duration: Target duration in seconds (must be positive).

    Raises:
        TypeError: If ``duration`` is not numeric.
        ValueError: If ``duration`` is not positive.
    """

    def __init__(self, duration: float) -> None:
        check_type(duration, (int, float), "duration")
        if duration <= 0:
            raise ValueError(f"'duration' must be positive, got {duration}.")
        self._duration = float(duration)

    def __call__(
        self, data: tuple[np.ndarray, int]
    ) -> tuple[np.ndarray, int]:
        """Apply random crop.

        Args:
            data: ``(waveform, sample_rate)`` tuple.

        Returns:
            ``(cropped_waveform, sample_rate)`` tuple.
        """
        check_type(data, tuple, "data")
        y, sr = data
        total = librosa.get_duration(y=y, sr=sr)
        if total <= self._duration:
            return data
        start_sec = random.uniform(0.0, total - self._duration)
        start = int(start_sec * sr)
        n = int(self._duration * sr)
        return y[start : start + n], sr

    @property
    def duration(self) -> float:
        """Target duration in seconds."""
        return self._duration


class Resample(Transform):
    """Resample an audio track to a new sampling rate.

    Args:
        target_sr: Target sampling rate in Hz.

    Raises:
        TypeError: If ``target_sr`` is not an ``int``.
        ValueError: If ``target_sr`` is less than 1.
    """

    def __init__(self, target_sr: int) -> None:
        check_type(target_sr, int, "target_sr")
        if target_sr < 1:
            raise ValueError(f"'target_sr' must be ≥ 1, got {target_sr}.")
        self._target_sr = target_sr

    def __call__(
        self, data: tuple[np.ndarray, int]
    ) -> tuple[np.ndarray, int]:
        """Resample the waveform.

        Args:
            data: ``(waveform, sample_rate)`` tuple.

        Returns:
            ``(resampled_waveform, target_sr)`` tuple.
        """
        check_type(data, tuple, "data")
        y, sr = data
        if sr == self._target_sr:
            return data
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=self._target_sr)
        return y_resampled, self._target_sr

    @property
    def target_sr(self) -> int:
        """Target sampling rate in Hz."""
        return self._target_sr


class PitchShift(Transform):
    """Shift the pitch of an audio track by a fixed number of semitones.

    Args:
        n_steps: Semitones to shift (positive = up, negative = down).

    Raises:
        TypeError: If ``n_steps`` is not numeric.
    """

    def __init__(self, n_steps: float) -> None:
        check_type(n_steps, (int, float), "n_steps")
        self._n_steps = float(n_steps)

    def __call__(
        self, data: tuple[np.ndarray, int]
    ) -> tuple[np.ndarray, int]:
        """Apply pitch shift.

        Args:
            data: ``(waveform, sample_rate)`` tuple.

        Returns:
            ``(shifted_waveform, sample_rate)`` tuple.
        """
        check_type(data, tuple, "data")
        y, sr = data
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=self._n_steps)
        return y_shifted, sr

    @property
    def n_steps(self) -> float:
        """Semitone shift applied."""
        return self._n_steps


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline(Transform):
    """Chain transforms and apply them sequentially.

    Takes a variable number of :class:`Transform` instances and applies them
    left to right. Order matters: some transforms change the data type (e.g.
    :class:`MelSpectrogram`) and cannot be followed by transforms that expect
    the original type.

    Args:
        *transforms: :class:`Transform` instances to apply in order.

    Raises:
        TypeError: If any positional argument is not a :class:`Transform`.

    Example::

        pipeline = Pipeline(
            AudioRandomCrop(duration=5.0),
            Resample(target_sr=22050),
            MelSpectrogram(n_mels=128),
        )
        spectrogram = pipeline((y, sr))
    """

    def __init__(self, *transforms: Transform) -> None:
        for i, t in enumerate(transforms):
            check_type(t, Transform, f"transforms[{i}]")
        self._transforms: tuple[Transform, ...] = transforms

    def __call__(self, data: Any) -> Any:
        """Apply all transforms in sequence.

        Args:
            data: Input data compatible with the first transform.

        Returns:
            Output after all transforms have been applied.
        """
        for transform in self._transforms:
            data = transform(data)
        return data

    @property
    def transforms(self) -> tuple[Transform, ...]:
        """The ordered tuple of transforms in this pipeline."""
        return self._transforms
