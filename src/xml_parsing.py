from music21 import converter, note, chord

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left


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

        return 0.0, end_sec - start_sec

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
        start_sec starts from zero, relative to the audio and assumes
        the first measure of the audio is a beat countdown.

        Args:
            score_events:
                List of score event dictionaries. Each event must contain
                "start_quarter_length" and "end_quarter_length".

        Returns:
            The same mutated list, with each event modified in place.
        """
        start_ql = score_events[0]["start_quarter_length"]
        countdown = self.get_seconds_per_bar()

        for event in score_events:
            event["start_sec"] = float(
                self._quarter_length_to_seconds(
                    event["start_quarter_length"] - start_ql
                )
                + countdown
            )
            event["end_sec"] = float(
                self._quarter_length_to_seconds(event["end_quarter_length"] - start_ql)
                + countdown
            )

        return score_events

    def get_seconds_per_bar(self):
        ql_per_bar = self._quarter_lengths_per_bar()
        secs_per_bar = self._quarter_length_to_seconds(ql_per_bar)

        return secs_per_bar

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


@dataclass
class MusicxmlParser:
    path: str
    part_idx: int = 0

    def _validate(self):
        path = Path(self.path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        score = converter.parse(path, format="musicxml")

        if not score.parts:
            raise ValueError("No parts found in MusicXML file")

        if self.part_idx >= len(score.parts):
            raise IndexError(
                f"part_idx: {self.part_idx} out of range. "
                f"Score has: {len(score.parts)} parts."
            )

    def extract_score_events(self):
        self._validate()
        score = converter.parse(self.path, format="musicxml")

        part = score.parts[self.part_idx]
        events = []

        for el in part.recurse().notesAndRests:
            start_ql = float(el.getOffsetInHierarchy(part))
            duration_ql = float(el.duration.quarterLength)
            end_ql = start_ql + duration_ql

            if isinstance(el, note.Note):
                pitch_names = [el.pitch.nameWithOctave]
                midi = [el.pitch.midi]
                kind = "note"
            elif isinstance(el, chord.Chord):
                pitch_names = [p.nameWithOctave for p in el.pitches]
                midi = [p.midi for p in el.pitches]
                kind = "chord"
            elif isinstance(el, note.Rest):
                pitch_names = []
                midi = []
                kind = "rest"
            else:
                continue

            events.append(
                {
                    "kind": kind,
                    "pitch_name": pitch_names,
                    "midi": midi,
                    "start_quarter_length": start_ql,
                    "duration_quarter_length": duration_ql,
                    "end_quarter_length": end_ql,
                }
            )

        return events
