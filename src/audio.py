from dataclasses import dataclass
from pathlib import Path

import numpy as np


def hz_to_midi(pitches_hz: np.ndarray) -> np.ndarray:
    """Convert pitch values from Hertz to MIDI note numbers.

    Args:
    pitches_hz: Array of pitch frequencies in Hertz.

    Returns:
    An array containing the corresponding MIDI note numbers.

    Raises:
    ValueError: If any frequency value is zero or negative.
    """

    if np.any(pitches_hz <= 0):
        raise ValueError("Hz values cant be zero or negative")

    return 69 + 12 * np.log2(pitches_hz / 440.0)


@dataclass
class Audio:
    """Represent an audio input together with its musical metadata.

    Attributes:
        path: Path to the audio file.
        bpm: Tempo of the audio in beats per minute.
        time_signature: Time signature represented as a tuple containing the
            numerator and denominator.
    """

    path: str
    bpm: float
    time_signature: tuple[int, int]

    def __post_init__(self):
        """Validate the audio instance after initialization."""
        self._validate()

    def _validate(self):
        """Validate the audio file path and musical parameters.

        Raises:
            FileNotFoundError: If the specified audio file does not exist.
            ValueError: If BPM is not positive or if the time signature contains
                non-positive values.
        """

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if self.bpm <= 0:
            raise ValueError("Bpm must be > 0")

        if self.time_signature[0] <= 0 or self.time_signature[1] <= 0:
            raise ValueError("Time signature must be at least (1/1) or more")
