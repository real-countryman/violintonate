from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.audio import Audio

FRAME_LENGTH = 2048
HOP_LENGTH = 256


# TODO look into resolution parameter in librosa.pyin() function
@dataclass
class PitchExtractor:
    """
    Extracts pitch estimates and their corresponding timestamps from an audio file.

    The extractor can optionally process only a selected time range of the audio,
    defined by start_sec and end_sec. It uses librosa.pyin to estimate the
    fundamental frequency and filters out frames that are likely to be silence,
    unvoiced, or unreliable.

    Attributes:
        audio:
            Audio object containing at least the path to the audio file.

        start_sec:
            Start time in seconds from which pitch extraction should begin.
            If None, extraction starts at the beginning of the audio.

        end_sec:
            End time in seconds where pitch extraction should stop.
            If None, extraction should continue until the end of the selected audio.
            Currently, this value is converted to a duration internally.

    Class Attributes:
        FRAME_LENGTH:
            Number of samples used in each analysis frame.

        HOP_LENGTH:
            Number of samples between consecutive analysis frames.

        VOICED_PROB_THRESHOLD:
            Minimum voiced probability required for a frame to be kept.

        RMS_DB_THRESHOLD:
            Minimum RMS level in decibels required for a frame to be kept.
            Frames below this threshold are treated as too quiet.
    """

    audio: Audio
    start_sec: float
    end_sec: float

    def __post_init__(self):
        self._validate()

    def extract_pitches_and_times(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """
        Extracts pitches and their timestamps from the audio file.

        The audio is loaded from start_sec to end_sec, converted to mono,
        and analyzed using librosa.pyin. The extracted pitch frames are then
        filtered using three criteria:

            - the frame must be marked as voiced
            - the voiced probability must be above VOICED_PROB_THRESHOLD
            - the RMS level must be above RMS_DB_THRESHOLD

        Args:
            get_midi:
                If True, returns pitch values in fracional MIDI.
                If False, converts the detected pitches to note names,
                for example "A4" instead of 69.0 Hz.

        Returns:
            A tuple containing:

                pitches:
                    A NumPy array of pitch values. These are either MIDI values
                    or note names, depending on get_midi.

                pitch_times:
                    A NumPy array of timestamps in seconds. Each timestamp
                    corresponds to one pitch value.

        Raises:
            ValueError:
                Should be raised by _validate_measure_range if the selected
                time range is invalid.
        """

        dur = self.end_sec - self.start_sec

        # Load audio
        # audio signal, sample rate
        y, sr = librosa.load(
            self.audio.path, sr=None, offset=self.start_sec, duration=dur, mono=True
        )

        # Estimate pitch / fundamental frequency
        f0, voiced_flag, voiced_prob = self._extract_f0(y, sr)

        # Time axis for each pitch estimate
        times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH) + self.start_sec

        # Compute rms
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]

        return (f0, voiced_flag, voiced_prob), rms, times

    def _validate(self):
        if self.start_sec < 0:
            raise ValueError("Start_sec must be bigger than 0")

        if self.end_sec < 0:
            raise ValueError("End_sec must be bigger than 0")

        if self.end_sec - self.start_sec <= 0:
            raise ValueError("Duration cant be less or equal to 0")

    def _extract_f0(self, y, sr: int):
        """
        Estimates the fundamental frequency of the audio signal.

        Uses librosa.pyin to estimate pitch frame by frame. The pitch range is
        limited to the approximate range of the violin, from G3 to E7.

        Args:
            y:
                Audio signal as a NumPy array.

            sr:
                Sample rate of the audio signal.

        Returns:
            A tuple containing:

                f0:
                    Estimated fundamental frequency for each frame in Hertz.
                    Unvoiced frames may contain NaN values.

                voiced_flag:
                    Boolean array indicating whether each frame is considered voiced.

                voiced_prob:
                    Probability array indicating the confidence that each frame is voiced.
        """
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("G3"),
            fmax=librosa.note_to_hz("E7"),
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )

        return f0, voiced_flag, voiced_prob


@dataclass
class VoicedPitchFilter:
    f0: np.ndarray
    voiced_flags: np.ndarray
    voiced_probs: np.ndarray
    times: np.ndarray
    rms: np.ndarray

    # May need tweaking
    VOICED_PROB_THRESHOLD = 0.8

    def __post_init__(self):
        self._validate()

    def filter_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes unreliable pitch frames.

        A frame is kept only if it is voiced, has a valid f0 value, has a voiced
        probability above VOICED_PROB_THRESHOLD, and is loud enough according to
        RMS_THRESHOLD.

        Returns:
            A tuple containing:

                pitches:
                    Filtered pitch values in Hertz.

                pitch_times:
                    Timestamps corresponding to the filtered pitch values.
        """
        mask = (
            (self.voiced_flags == True)
            & ~np.isnan(self.f0)
            & (self.voiced_probs > self.VOICED_PROB_THRESHOLD)
        )

        times_clean = self.times[mask]
        f0_clean = self.f0[mask]

        pitches = f0_clean
        pitch_times = times_clean

        return pitches, pitch_times

    def _validate(self):
        if (
            self.f0.size != self.voiced_flags.size
            or self.voiced_flags.size != self.voiced_probs.size
            or self.voiced_probs.size != self.times.size
            or self.times.size != self.rms.size
        ):
            raise ValueError("Arrays must be the same lenghts")
