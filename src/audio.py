from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


def hz_to_midi(pitches_hz: np.ndarray) -> np.ndarray:
    if np.any(pitches_hz <= 0):
        raise ValueError("Hz values cant be zero or negative")

    return 69 + 12 * np.log2(pitches_hz / 440.0)


@dataclass
class Audio:
    path: str
    bpm: float
    time_signature: tuple[int, int]
    msr_cnt: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """
        Helper function for validation of parameters
        """
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if self.bpm <= 0:
            raise ValueError("Bpm must be > 0")

        if self.time_signature[0] <= 0 or self.time_signature[1] <= 0:
            raise ValueError("Time signature must be at least (1/1) or more")

        if self.msr_cnt <= 0:
            raise ValueError("Msr_cnt must be > 0")

        if type(self.msr_cnt) is not int:
            raise ValueError("Msr_cnt must be an int")
