from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.audio import Audio


@dataclass
class PitchExtractor:
    audio: Audio

    FRAME_LENGTH = 2048
    HOP_LENGTH = 256
    # May need tweaking
    VOICED_PROB_THRESHOLD = 0.8
    RMS_DB_THRESHOLD = -45.0

    def extract_pitches_and_times(
        self,
        start_msr: int = 0,
        msr_offset: float = 0,
        end_msr: int | None = None,
        get_hz=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts audio format into pitches and times and removes unvoiced / low confidence frames.

        Args:
            start_msr: The start measure number index (start at 0)
            msr_offset: The shift in beats in measures (start at 0)
            end_msr: The end measure number index
            get_hz: If true, returns hz values instead of notes (391.9 instead of G4)
                Defaults to False.
        """
        if end_msr is None:
            end_msr = self.audio.msr_cnt - 1

        self._validate_measure_range(start_msr, msr_offset, end_msr)

        start = self._bpm_to_secs(start_msr, msr_offset)
        end = self._bpm_to_secs(end_msr, msr_offset=0)
        dur = end - start

        # Load audio
        # audio signal, sample rate
        y, sr = librosa.load(
            self.audio.path, sr=None, offset=start, duration=dur, mono=True
        )

        # Estimate pitch / fundamental frequency
        f0, voiced_flag, voiced_prob = self._extract_f0(y, sr)

        # Time axis for each pitch estimate
        times = librosa.times_like(f0, sr=sr, hop_length=self.HOP_LENGTH) + start

        # Compute rms
        rms = librosa.feature.rms(
            y=y, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Keep only voiced and confident frames
        pitches, pitch_times = self._filter_voiced_frames(
            f0, voiced_flag, voiced_prob, times, rms_db
        )

        if not get_hz:
            notes = librosa.midi_to_note(
                np.round(librosa.hz_to_midi(pitches)).astype(int)
            )

            return notes, pitch_times
        else:
            return pitches, pitch_times

    @property
    def _seconds_per_bar(self) -> float:
        """
        Calculates how much time in seconds takes one bar in sheet music

        Returns:
            How many seconds per bar
        """
        beats_per_bar = self.audio.time_signature[0]
        return beats_per_bar * 60 / self.audio.bpm

    def _bpm_to_secs(self, measure: int, msr_offset: int) -> float:
        """
        Converts concrete beat in score to seconds from start

        Args:
            audio: The audio dataclass to calculate
            measure: The measure number index (start at 0)
            msr_offset: The offset inside the measure (start at 0)

        Returns:
            The time in seconds
        """
        if not 0 <= measure < self.audio.msr_cnt:
            raise ValueError(
                "Argument measure must satisfy: 0 <= measure < self.msr_cnt \n"
                f"got: {measure}"
            )

        beats_per_measure = self.audio.time_signature[0]

        if not 0 <= msr_offset < beats_per_measure:
            raise ValueError(
                "Argument msr_offset must satisfy: 0 <= msr_offset < self.time_signature[0] \n"
                f"got: {msr_offset}"
            )

        offset = self._seconds_per_bar / beats_per_measure * msr_offset
        return self._seconds_per_bar * measure + offset

    def _validate_measure_range(
        self, start_msr: int, msr_offset: float, end_msr: int
    ) -> None:
        if not 0 <= start_msr < self.audio.msr_cnt:
            raise ValueError(
                "Argument start_msr must satisfy: 0 <= start_msr < self.msr_cnt\n"
                f"got: {start_msr}"
            )

        beats_per_measure = self.audio.time_signature[0]
        if not 0 <= msr_offset < beats_per_measure:
            raise ValueError(
                "Argument msr_offset must satisfy: 0 <= msr_offset < self.time_signature[0]\n"
                f"got: {msr_offset}"
            )

        if not 0 <= end_msr <= self.audio.msr_cnt:
            raise ValueError(
                "Argument end_msr must satisfy: 0 <= end_msr <= self.msr_cnt\n"
                f"got: {end_msr}"
            )

        if not start_msr < end_msr:
            raise ValueError(
                "Arguments start_msr and end_msr must satisfy: start_msr < end_msr\n"
                f"got: start_msr={start_msr}, end_msr={end_msr}"
            )

    def _extract_f0(self, y, sr: int):
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("G3"),
            fmax=librosa.note_to_hz("E7"),
            sr=sr,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

        return f0, voiced_flag, voiced_prob

    def _filter_voiced_frames(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        voiced_prob: np.ndarray,
        times: np.ndarray,
        rms_db: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (
            (voiced_flag == True)
            & ~np.isnan(f0)
            & (voiced_prob > self.VOICED_PROB_THRESHOLD)
            & (rms_db > self.RMS_DB_THRESHOLD)
        )

        times_clean = times[mask]
        f0_clean = f0[mask]

        pitches = f0_clean
        pitch_times = times_clean

        return pitches, pitch_times
