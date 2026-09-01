from music21 import converter, note, chord

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left


@dataclass
class ScoreTimeMapper:
    """Map score positions in quarter lengths to corresponding audio times.

    The mapper uses tempo and time signature information to convert score
    positions into seconds. It can determine the selected score range, crop
    score events to that range, add timing information to events, and calculate
    audio bounds including a one-bar count-in and end padding.

    Attributes:
        bpm: Tempo in beats per minute.
        time_signature: Time signature represented as a numerator and denominator.
        start_msr: Zero-based index of the first selected measure.
        start_offset: Offset within the start measure in quarter lengths.
        end_msr: Zero-based index of the end measure.
        end_offset: Offset within the end measure in quarter lengths.
    """

    bpm: float
    time_signature: tuple[int, int]
    start_msr: int
    start_offset: float
    end_msr: int
    end_offset: float

    def __post_init__(self):
        """Validate the mapper parameters after initialization."""

        self._validate()

    def get_seconds_start_end_score_events(self) -> tuple[float, float]:
        """Return the start and end times of the selected score section.

        The selected measure positions are first converted to quarter lengths and
        then mapped to time in seconds. The score section is positioned after one
        bar reserved for the count-in.

        Returns:
            A tuple containing the score section start and end times in seconds.
        """

        start_ql, end_ql = self.get_start_end_in_quarter_lengths()

        start_sec = self.get_seconds_per_bar() + self.start_offset

        section_duration_ql = end_ql - start_ql
        end_sec = start_sec + self._quarter_length_to_seconds(section_duration_ql)

        return start_sec, end_sec

    def get_audio_bounds_with_count_in_and_end_padding(self) -> tuple[float, float]:
        """Return audio bounds including count-in and end padding.

        The audio starts one full bar before the selected score section to include
        the count-in. One additional beat is added after the score section.

        Returns:
            A tuple containing the audio start and end times in seconds.

        Raises:
            ValueError: If there is not enough time before the score section for
                a complete one-bar count-in.
        """

        start_sec, end_sec = self.get_seconds_start_end_score_events()

        if start_sec < 0:
            raise ValueError("Count in is too short!")

        one_beat = self.get_seconds_per_bar() / self.time_signature[0]

        return start_sec, end_sec + one_beat

    def get_start_end_in_quarter_lengths(self) -> tuple[float, float]:
        """Convert the selected measure range to quarter-length positions.

        Returns:
            A tuple containing the absolute start and end positions in quarter
            lengths.
        """

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
        """Crop score events to the selected score range.

        Events are selected according to their start positions in quarter lengths.
        The start boundary is inclusive and the end boundary is exclusive.

        Args:
            score_events: Score events containing ``start_quarter_length`` values.

        Returns:
            A list containing the score events within the selected range.
        """

        start_ql, end_ql = self.get_start_end_in_quarter_lengths()

        start_times = [event["start_quarter_length"] for event in score_events]

        start_idx = bisect_left(start_times, start_ql)
        end_idx = bisect_left(start_times, end_ql)

        return score_events[start_idx:end_idx]

    def score_events_add_times(self, score_events: list[dict]) -> list[dict]:
        """Add start and end times in seconds to score events.

        Timing is calculated relative to the first supplied score event. One bar
        is added before the first event to represent the audio count-in. The input
        event dictionaries are modified in place.

        Args:
            score_events: Score events containing ``start_quarter_length`` and
                ``end_quarter_length`` values.

        Returns:
            The same list of score events with ``start_sec`` and ``end_sec`` added.
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
        """Return the duration of one bar in seconds.

        Returns:
            Duration of one complete measure in seconds according to the configured
            tempo and time signature.
        """

        ql_per_bar = self._quarter_lengths_per_bar()
        secs_per_bar = self._quarter_length_to_seconds(ql_per_bar)

        return secs_per_bar

    def get_seconds_per_beat(self):
        """Return the duration of one notated beat in seconds.

        The duration of one bar is divided by the numerator of the time signature.

        Returns:
            Duration of one beat in seconds.
        """

        return self.get_seconds_per_bar() / self.time_signature[0]

    def _quarter_length_to_seconds(self, ql: float) -> float:
        """Convert a duration in quarter lengths to seconds.

        Args:
            ql: Duration or position expressed in quarter lengths.

        Returns:
            Corresponding duration in seconds.
        """

        denominator = self.time_signature[1]
        return ql * 60 / self.bpm * denominator / 4

    def _measure_offset_to_quarter_length(self, measure: int, offset: float) -> float:
        """Convert a measure and offset to an absolute quarter-length position.

        Args:
            measure: Zero-based measure index.
            offset: Offset within the measure in quarter lengths.

        Returns:
            Absolute score position in quarter lengths.
        """

        return measure * self._quarter_lengths_per_bar() + offset

    def _quarter_lengths_per_bar(self) -> float:
        """Return the number of quarter lengths in one bar.

        The value is calculated from the numerator and denominator of the configured
        time signature.

        Returns:
            Length of one complete measure in quarter lengths.
        """

        numerator, denominator = self.time_signature
        return numerator * (4 / denominator)

    # TODO
    def _validate(self):
        """Validate tempo, time signature, and score range parameters.

        TODO:
            Validate that offsets do not exceed the corresponding measure length.
            Validate that the selected end position occurs after the start position.

        Raises:
            ValueError: If BPM is not positive, the time signature has an invalid
                format or contains non-positive values, or any measure index or
                offset is negative.
        """

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
    """Parse score events from a MusicXML file.

    The parser loads a MusicXML score, selects one part, and extracts notes,
    chords, and rests together with their measure, pitch, and timing
    information expressed in quarter lengths.

    Attributes:
        path: Path to the MusicXML file.
        part_idx: Zero-based index of the score part to extract.
    TODO:
        make a __post_init__() function that calls _validate() to make it
        cleaner
    """

    path: str
    part_idx: int = 0

    def _validate(self):
        """Validate the MusicXML file and selected score part.

        Raises:
            FileNotFoundError: If the MusicXML file does not exist.
            ValueError: If the MusicXML score contains no parts.
            IndexError: If ``part_idx`` is outside the available part range.
        """

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
        """Extract notes, chords, and rests from the selected MusicXML part.

        Each score element is converted into a dictionary containing its type,
        measure number, pitch information, and timing values in quarter lengths.

        Returns:
            A list of dictionaries containing:
                - Event type: ``note``, ``chord``, or ``rest``.
                - Measure number.
                - Pitch names with octave information.
                - MIDI pitch values.
                - Start position in quarter lengths.
                - Duration in quarter lengths.
                - End position in quarter lengths.

        Raises:
            FileNotFoundError: If the MusicXML file does not exist.
            ValueError: If the MusicXML score contains no parts.
            IndexError: If ``part_idx`` is outside the available part range.
        """

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
