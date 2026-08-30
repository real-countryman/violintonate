from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.audio import Audio


# TODO look into resolution parameter in librosa.pyin() function
@dataclass
class PitchExtractor:
    """Extract pitch, voicing, RMS, and timing information from an audio segment.

    The extractor loads a selected time interval from an audio file and analyzes
    it frame by frame. Fundamental frequency and voicing information are estimated
    using ``librosa.pyin``, while RMS values are calculated for the same audio
    segment.

    Attributes:
        audio: Audio object containing the path to the audio file.
        start_sec: Start time of the analyzed segment in seconds.
        end_sec: End time of the analyzed segment in seconds.
    """

    audio: Audio
    start_sec: float
    end_sec: float

    FRAME_LENGTH = 2048
    HOP_LENGTH = 256
    RESOLUTION = 0.05

    def __post_init__(self):
        """Validate the extractor parameters after initialization."""
        self._validate()

    def extract_pitches_and_times(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """Extract pitch, voicing, RMS, and corresponding time values.

        The selected audio segment is loaded in mono and analyzed using
        ``librosa.pyin``. RMS values are calculated using the same frame and hop
        configuration, and timestamps are shifted by ``start_sec`` so that they
        correspond to positions in the original audio.

        Returns: A tuple containing:
            - A tuple of fundamental frequencies in Hertz, voiced flags, and
              voiced probabilities.
            - An array of RMS values.
            - An array of corresponding frame times in seconds.
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
        times = (
            librosa.times_like(f0, sr=sr, hop_length=self.HOP_LENGTH) + self.start_sec
        )

        # Compute rms
        rms = librosa.feature.rms(
            y=y, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]

        return (f0, voiced_flag, voiced_prob), rms, times

    def _validate(self):
        """Validate the selected audio time range.

        Raises:
            ValueError: If ``start_sec`` or ``end_sec`` is negative, or if
            ``end_sec`` is not greater than ``start_sec``.
        """
        if self.start_sec < 0:
            raise ValueError("Start_sec must be bigger than 0")

        if self.end_sec < 0:
            raise ValueError("End_sec must be bigger than 0")

        if self.end_sec - self.start_sec <= 0:
            raise ValueError("Duration cant be less or equal to 0")

    def _extract_f0(self, y, sr: int):
        """Estimate fundamental frequency and voicing information.

        Uses ``librosa.pyin`` to analyze the audio signal within the approximate
        violin pitch range from G3 to E7.

        Args:
            y: Audio signal samples.
            sr: Sample rate of the audio signal.

        Returns: A tuple containing:
            - Estimated fundamental frequencies in Hertz.
            - Boolean voiced flags for individual frames.
            - Voiced probabilities for individual frames.
        """
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("G3"),
            fmax=librosa.note_to_hz("E7"),
            sr=sr,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
            resolution=self.RESOLUTION,
        )

        return f0, voiced_flag, voiced_prob


@dataclass
class VoicedPitchFilter:
    """Filter unreliable pitch frames based on voicing information.

    The filter removes pitch frames that are unvoiced, contain invalid
    fundamental frequency values, or have insufficient voiced probability.

    Attributes:
        f0: Array of estimated fundamental frequencies in Hertz.
        voiced_flags: Boolean array indicating whether individual frames are voiced.
        voiced_probs: Array containing voiced probabilities for individual frames.
        times: Array of timestamps in seconds corresponding to the pitch frames.
        rms: Array of RMS values corresponding to the pitch frames.

    TODO:
        remove rms attribute, not necessary
    """

    f0: np.ndarray
    voiced_flags: np.ndarray
    voiced_probs: np.ndarray
    times: np.ndarray
    rms: np.ndarray

    # May need tweaking
    VOICED_PROB_THRESHOLD = 0.8

    def __post_init__(self):
        """Validate the filter parameters after initialization."""

        self._validate()

    def filter_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Filter unreliable pitch frames.

        A frame is retained only when it is marked as voiced, its fundamental
        frequency is not NaN, and its voiced probability is greater than
        ``VOICED_PROB_THRESHOLD``.

        Returns:
            A tuple containing:
                - Filtered fundamental frequency values in Hertz.
                - Timestamps in seconds corresponding to the filtered values.
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
        """Validate the input arrays.

        Raises:
            ValueError: If ``f0``, ``voiced_flags``, ``voiced_probs``, ``times``,
                and ``rms`` do not have the same number of elements.
        """

        if (
            self.f0.size != self.voiced_flags.size
            or self.voiced_flags.size != self.voiced_probs.size
            or self.voiced_probs.size != self.times.size
            or self.times.size != self.rms.size
        ):
            raise ValueError("Arrays must be the same lenghts")
