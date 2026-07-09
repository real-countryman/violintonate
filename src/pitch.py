from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from bisect import bisect_left

from src.audio import Audio


@dataclass
class ScoreTimeMapper:
    """
    Converts score positions, expressed as measure numbers and quarter-length offsets,
    into audio time in seconds.

    The class uses BPM together with a beat unit to determine how score durations
    map to real time.

    If bpm_beat_ql is not provided, it is chosen automatically from the denominator
    of the time signature. This means the default BPM beat follows the written beat
    unit of the meter.

    Examples:
        4/4 with bpm_beat_ql=None:
            BPM counts quarter notes.
            bpm_beat_ql = 1.0

        6/8 with bpm_beat_ql=None:
            BPM counts eighth notes.
            bpm_beat_ql = 0.5

        6/8 with bpm_beat_ql=1.5:
            BPM counts dotted quarter notes.
            This represents two large dotted-quarter beats per bar.

    Attributes:
        bpm:
            Tempo value in beats per minute.

            The meaning of one BPM beat is defined by bpm_beat_ql.
            For example, bpm=120 with bpm_beat_ql=1.0 means:
            quarter note = 120 BPM.

        time_signature:
            Tuple in the form (numerator, denominator), for example (4, 4),
            (3, 4), (6, 8), or (2, 2).

            The denominator is also used to choose the default bpm_beat_ql
            when bpm_beat_ql is not provided.

        start_msr:
            Zero-based start measure index.

            For example, start_msr=0 means the first measure.

        start_offset:
            Offset inside the start measure, expressed in quarter lengths.

            For example:
                - 0.0 means the beginning of the measure
                - 1.0 means one quarter note after the beginning
                - 0.5 means one eighth note after the beginning

        end_msr:
            Zero-based end measure index.

            For example, end_msr=4 means the fifth measure.

        end_offset:
            Offset inside the end measure, expressed in quarter lengths.

        bpm_beat_ql:
            Optional length of one BPM beat, expressed in quarter lengths.

            If None, the value is calculated automatically as:

                bpm_beat_ql = 4 / denominator

            Examples:
                - quarter note beat: bpm_beat_ql = 1.0
                - eighth note beat: bpm_beat_ql = 0.5
                - dotted quarter beat: bpm_beat_ql = 1.5
                - half note beat: bpm_beat_ql = 2.0
    """

    bpm: float
    time_signature: tuple[int, int]
    start_msr: int
    start_offset: float
    end_msr: int
    end_offset: float
    bpm_beat_ql: float | None = None

    def __post_init__(self):
        """
        Sets the default BPM beat unit if it was not provided, then validates
        the initialized values.

        If bpm_beat_ql is None, the default beat unit follows the denominator
        of the time signature:

            bpm_beat_ql = 4 / denominator

        Examples:
            4/4 -> bpm_beat_ql = 1.0
            6/8 -> bpm_beat_ql = 0.5
            2/2 -> bpm_beat_ql = 2.0
        """
        if self.bpm_beat_ql is None:
            self.bpm_beat_ql = self._default_bpm_beat_ql()

        self._validate()

    def _default_bpm_beat_ql(self) -> float:
        """
        Calculates the default BPM beat unit from the time-signature denominator.

        The default behavior is:

            bpm_beat_ql = 4 / denominator

        This means the BPM beat follows the written denominator of the meter.

        Examples:
            4/4:
                denominator = 4
                bpm_beat_ql = 4 / 4 = 1.0
                BPM counts quarter notes.

            3/4:
                denominator = 4
                bpm_beat_ql = 4 / 4 = 1.0
                BPM counts quarter notes.

            6/8:
                denominator = 8
                bpm_beat_ql = 4 / 8 = 0.5
                BPM counts eighth notes.

            2/2:
                denominator = 2
                bpm_beat_ql = 4 / 2 = 2.0
                BPM counts half notes.

        Returns:
            The default BPM beat unit, expressed in quarter lengths.
        """
        _, denominator = self.time_signature
        return 4 / denominator

    def get_start_end_in_seconds(self) -> tuple[float, float]:
        """
        Calculates the configured start and end score positions in seconds.

        Returns:
            A tuple in the form:

                (start_seconds, end_seconds)

            start_seconds:
                Absolute time from the beginning of the score to the start position.

            end_seconds:
                Absolute time from the beginning of the score to the end position.
        """
        start_ql, end_ql = self.get_start_end_in_quarter_lengths()

        start_sec = self._quarter_length_to_seconds(start_ql)
        end_sec = self._quarter_length_to_seconds(end_ql)

        return start_sec, end_sec

    def get_start_end_in_quarter_lengths(self) -> tuple[float, float]:
        start = self._measure_offset_to_quarter_length(
            self.start_msr,
            self.start_offset,
        )
        end = self._measure_offset_to_quarter_length(
            self.end_msr,
            self.end_offset,
        )

        return start, end

    def crop_score_events(self, score_events: list[dict]) -> list[dict]:
        start_ql, end_ql = self.get_start_end_in_quarter_lengths()

        start_times = [event["start_quarter_length"] for event in score_events]

        start_idx = bisect_left(start_times, start_ql)
        end_idx = bisect_left(start_times, end_ql)

        return score_events[start_idx:end_idx]

    def score_events_add_times(self, score_events: list[dict]) -> list[dict]:
        """
        Adds start_sec and end_sec to each score event in place.

        Args:
            score_events:
                List of score event dictionaries. Each event must contain
                "start_quarter_length" and "end_quarter_length".

        Returns:
            The same mutated list, with each event modified in place.
        """
        for event in score_events:
            event["start_sec"] = self._quarter_length_to_seconds(
                event["start_quarter_length"]
            )
            event["end_sec"] = self._quarter_length_to_seconds(
                event["end_quarter_length"]
            )

        return score_events

    def _quarter_length_to_seconds(self, ql) -> float:
        """
        Converts a duration or offset in quarter lengths into seconds.

        A quarter length is a score-duration unit where:
            - quarter note = 1.0
            - eighth note = 0.5
            - dotted quarter note = 1.5
            - half note = 2.0
            - whole note = 4.0

        The BPM value tells how many beats happen per minute.
        The bpm_beat_ql value tells how long one of those BPM beats is.

        Formula:
            seconds = ql * 60 / bpm / bpm_beat_ql

        Examples:
            If bpm = 60 and bpm_beat_ql = 1.0:
                quarter note = 60 BPM
                1 quarter length = 1 second

            If bpm = 120 and bpm_beat_ql = 0.5:
                eighth note = 120 BPM
                1 eighth note = 0.5 seconds
                1 quarter length = 1 second

            If bpm = 60 and bpm_beat_ql = 1.5:
                dotted quarter note = 60 BPM
                1 dotted quarter note = 1 second
                1 quarter length = 0.666... seconds

        Args:
            ql:
                Duration or offset expressed in quarter lengths.

        Returns:
            The equivalent duration in seconds.
        """
        seconds_per_beat = 60 / self.bpm
        return ql * seconds_per_beat / self.bpm_beat_ql

    def _measure_offset_to_quarter_length(self, measure: int, offset: float) -> float:
        return measure * self._quarter_lengths_per_bar() + offset

    def _quarter_lengths_per_bar(self) -> float:
        numerator, denominator = self.time_signature
        return numerator * (4 / denominator)

    # TODO
    def _validate(self):
        return


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
    start_sec: float | None = None
    end_sec: float | None = None

    FRAME_LENGTH = 2048
    HOP_LENGTH = 256

    def extract_pitches_and_times(
        self,
        get_midi=True,
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
        if self.start_sec is None:
            self.start_sec = 0

        dur = None
        if self.end_sec is None:
            dur = self.end_sec - self.start_sec

        self._validate_measure_range()

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

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        return (f0, voiced_flag, voiced_prob), rms_db, times

    # TODO
    def _validate_measure_range(self):
        return

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
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

        return f0, voiced_flag, voiced_prob


@dataclass
class IntotationAnalyzer:
    # TODO intonation_tolerance_th (in cents)
    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    # 1 semitone = 100 cents, may need tweaking
    INTONATION_TOLERANCE_CENTS = 10

    def get_intonation_bool(self) -> list[tuple[float, float, bool]]:
        result = []

        for pitch, pitch_time in zip(self.pitches, self.pitch_times):
            ok = self._compare_pitch_with_score_event(pitch, pitch_time)
            result.append((pitch, pitch_time, ok))

        return result

    def get_bad_frames(self) -> list[tuple[float, float]]:
        analyzed_frames = self.get_intonation_bool()

        return [
            (pitch_time, pitch) for pitch_time, pitch, ok in analyzed_frames if not ok
        ]

    def _compare_pitch_with_score_event(self, pitch: float, pitch_time: float) -> bool:
        score_event = self._find_score_event_at_time(pitch_time)

        if score_event is None:
            return False

        if score_event["kind"] != "note":
            return False

        expected_midi = score_event["midi"]

        cents_error = abs((pitch - expected_midi) * 100)

        return cents_error <= self.INTONATION_TOLERANCE_CENTS

    def _find_score_event_at_time(self, pitch_time: float) -> dict | None:
        for event in self.score_events:
            if event["kind"] != "note":
                continue

            if event["start_sec"] <= pitch_time < event["end_sec"]:
                return event

        return None
