from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

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
        start = self._calculate_start()
        end = self._calculate_end()

        return start, end

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

    def _calculate_start(self):
        """
        Converts the configured start measure and start offset into seconds.

        The calculation is:

            start_seconds = start_msr * seconds_per_bar + start_offset_seconds

        where start_offset is measured inside the start measure in quarter lengths.

        Returns:
            The absolute start time in seconds from the beginning of the score.
        """
        sec_per_bar = self._seconds_per_bar()
        offset = self._quarter_length_to_seconds(self.start_offset)

        start = self.start_msr * sec_per_bar + offset
        return start

    def _calculate_end(self):
        """
        Converts the configured end measure and end offset into seconds.

        The calculation is:

            end_seconds = end_msr * seconds_per_bar + end_offset_seconds

        where end_offset is measured inside the end measure in quarter lengths.

        Returns:
            The absolute end time in seconds from the beginning of the score.
        """
        sec_per_bar = self._seconds_per_bar()
        offset = self._quarter_length_to_seconds(self.end_offset)

        end = self.end_msr * sec_per_bar + offset
        return end

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
        return ql * 60 / self.bpm / self.bpm_beat_ql

    def _seconds_per_bar(self) -> float:
        """
        Calculates the duration of one full measure in seconds.

        First, the time signature is converted into the number of quarter lengths
        in one bar:

            bar_ql = numerator * (4 / denominator)

        Then that quarter-length duration is converted into seconds using the BPM
        and bpm_beat_ql.

        Examples:
            4/4:
                bar_ql = 4 * (4 / 4) = 4.0 quarter lengths

            3/4:
                bar_ql = 3 * (4 / 4) = 3.0 quarter lengths

            6/8:
                bar_ql = 6 * (4 / 8) = 3.0 quarter lengths

            2/2:
                bar_ql = 2 * (4 / 2) = 4.0 quarter lengths

        Returns:
            The duration of one full measure in seconds.
        """
        numerator, denominator = self.time_signature
        bar_ql = numerator * (4 / denominator)

        return self._quarter_length_to_seconds(bar_ql)

    # TODO
    def _validate(self):
        return


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


# TODO class not functional, ScoreTimeMapper class needed!
@dataclass
class IntotationAnalyzer:
    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    def get_intonation_bool(self):
        results = []

        for event in self.score_events:
            if event["kind"] != "note":
                continue

            note_name = event["pitch_name"][0]
            start = event["start_quarter_length"]
            end = event["end_quarter_length"]

            mask = (self.pitch_times >= start) & (self.pitch_times < end)
            event_pitches = self.pitches[mask]
            event_times = self.pitch_times[mask]

            if len(event_pitches) == 0:
                continue

            ok = True

            results.append((note_name, event_times, ok))

        return results

    def get_bad_frames(self):
        return
