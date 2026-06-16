import math
from dataclasses import dataclass

import librosa
import numpy as np

@dataclass
class Audio:
    path: str
    bpm: float
    time_signature: tuple[int, int]
    msr_cnt: int

    @property
    def _seconds_per_bar(self) -> float:
        """
        Calculates how many seconds per bar
        
        Args:
            audio: The audio dataclass to calculate

        Returns:
            How many seconds per bar
        """
        beats_per_bar = self.time_signature[0]
        return beats_per_bar * 60 / self.bpm

    def _bpm_to_secs(self, measure: int, msr_offset: int) -> float:
        """
        Converts bpm to seconds taking to consideration measure number and measure offset

        Args:
            audio: The audio dataclass to calculate
            measure: The measure number index (start at 0)
            msr_offset: The offset inside the measure (start at 0)

        Returns:
            The time in seconds
        """
        offset = self._seconds_per_bar / self.time_signature[0] * msr_offset
        return self._seconds_per_bar * measure + offset

    def get_pitches_and_times(self, start_msr: int, msr_offset: int, end_msr: int) -> tuple[np.ndarray, np.ndarray]:
        start = self._bpm_to_secs(start_msr, msr_offset)
        end = self._bpm_to_secs(end_msr, msr_offset=0)
        dur = end - start
        # Load audio
        # audio signal, sample rate
        y, sr = librosa.load(self.path, sr=None, offset=start, duration=dur)

        # Estimate pitch / fundamental frequency
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("G3"),
            fmax=librosa.note_to_hz("E7"),
            sr=sr
        )

        # Time axis for each pitch estimate
        times = librosa.times_like(f0, sr=sr)

        # Keep only voiced frames
        pitches = f0[voiced_flag]
        pitch_times = times[voiced_flag]

        return pitches, pitch_times # Hz values, times in seconds