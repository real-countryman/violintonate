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

    def __post_init__(self):
        self._validate()

    def get_seconds_start_end_score_events(self) -> tuple[float, float]:
        start_ql, end_ql = self.get_start_end_in_quarter_lengths()

        start_sec = self.get_seconds_per_bar() + self.start_offset

        section_duration_ql = end_ql - start_ql
        end_sec = start_sec + self._quarter_length_to_seconds(section_duration_ql)

        return start_sec, end_sec

    def get_audio_bounds_with_count_in_and_end_padding(self) -> tuple[float, float]:
        start_score_sec, end_sec = self.get_seconds_start_end_score_events()
        start_sec = start_score_sec - self.get_seconds_per_bar()

        if start_sec < 0:
            raise ValueError("Count in is too short!")

        one_beat = self.get_seconds_per_bar() / self.time_signature[0]

        return start_sec, end_sec + one_beat

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

    def get_seconds_per_beat(self):
        return self.get_seconds_per_bar() / self.time_signature[0]

    def _quarter_length_to_seconds(self, ql: float) -> float:
        denominator = self.time_signature[1]
        return ql * 60 / self.bpm * denominator / 4

    def _measure_offset_to_quarter_length(self, measure: int, offset: float) -> float:
        return measure * self._quarter_lengths_per_bar() + offset

    def _quarter_lengths_per_bar(self) -> float:
        numerator, denominator = self.time_signature
        return numerator * (4 / denominator)

    # TODO
    def _validate(self):
        if self.bpm <= 0:
            raise ValueError("Bpm must be bigger than 0")

        if (
            not isinstance(self.time_signature, tuple)
            or len(self.time_signature) != 2
            or not all(isinstance(x, int) for x in self.time_signature)
        ):
            raise ValueError("Time signature must be tuple[int, int]")

        if self.time_signature[0] <= 0 or self.time_signature[1] <= 0:
            raise ValueError("Time signature can't contain zero or negative value")

        if self.start_msr < 0:
            raise ValueError("Start measure can't be negative")

        if self.start_offset < 0:
            raise ValueError("Start offset can't be negative")

        if self.end_msr < 0:
            raise ValueError("End measure can't be negative")

        if self.end_offset < 0:
            raise ValueError("End offset can't be negative")

        # TODO offset bigger than measure beat length
        # TODO end + offset > start + offset


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
                    "msr_number": el.measureNumber,
                    "pitch_name": pitch_names,
                    "midi": midi,
                    "start_quarter_length": start_ql,
                    "duration_quarter_length": duration_ql,
                    "end_quarter_length": end_ql,
                }
            )

        return events
