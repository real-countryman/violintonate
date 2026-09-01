from music21 import converter, note, chord

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left


@dataclass
class ScoreTimeMapper:
    """Map score positions in quarter lengths to audio times in seconds.

    ``ScoreTimeMapper`` converts score positions expressed in quarter lengths
    into seconds using the configured tempo and time signature. It also manages
    a selected score range, crops score events to that range, assigns audio
    timestamps to score events, and calculates audio boundaries including a
    one-measure count-in and end padding.

    Measure indices are zero-based. Offsets are expressed in quarter lengths
    relative to the beginning of their corresponding measure.

    Attributes:
        bpm: Tempo in beats per minute.
        time_signature: Time signature as ``(numerator, denominator)``.
        start_msr: Zero-based index of the first selected measure.
        start_offset: Offset from the beginning of ``start_msr`` in quarter
            lengths.
        end_msr: Zero-based index of the measure containing the end boundary.
        end_offset: Offset from the beginning of ``end_msr`` in quarter lengths.
        score_events: Score events containing score positions in quarter lengths.
        score_start_ql: Absolute start position of the selected score range in
            quarter lengths.
        score_end_ql: Absolute end position of the selected score range in
            quarter lengths.
    """

    bpm: float
    time_signature: tuple[int, int]
    start_msr: int
    start_offset: float
    end_msr: int
    end_offset: float
    score_events: list[dict]

    score_start_ql = -1
    score_end_ql = -1

    def __post_init__(self):
        """Validate mapper configuration after initialization."""
        self._validate()

    def _set_score_start_end_ql(self):
        """Set score range boundaries from the stored score events.

        The start boundary is based on the start position of the first score
        event plus ``start_offset``. The end boundary is based on the end
        position of the last score event plus ``end_offset``.

        This method updates ``score_start_ql`` and ``score_end_ql`` in place.
        """

        self.score_start_ql = (
            self.score_events[0]["start_quarter_length"] + self.start_offset
        )
        self.score_end_ql = (
            self.score_events[-1]["end_quarter_length"] + self.end_offset
        )

    def get_audio_bounds_with_count_in_and_end_padding(self) -> tuple[float, float]:
        """Return audio boundaries including count-in and end padding.

        The selected score range is extended by one complete measure before its
        start to account for the count-in and by one beat after its end as
        trailing padding.

        Returned times are relative to the beginning of this extended audio
        range. Therefore, the start time corresponds to the end of the one-bar
        count-in.

        Returns:
            A tuple ``(start_sec, end_sec)`` containing the selected score start
            time and padded audio end time in seconds.

        Raises:
            ValueError: If the calculated end boundary does not occur after the
                calculated audio start boundary.
        """

        audio_start_ql = self.score_start_ql - self._quarter_lengths_per_bar()
        one_beat = 1
        audio_end_ql = self.score_end_ql + one_beat

        if audio_end_ql <= audio_start_ql:
            raise ValueError("start_sec > end_sec")

        relative_start_ql = self._quarter_lengths_per_bar()
        relative_end_ql = audio_end_ql - audio_start_ql

        return self._quarter_length_to_seconds(
            relative_start_ql
        ), self._quarter_length_to_seconds(relative_end_ql)

    def crop_score_events(self, score_events: list[dict]) -> list[dict]:
        """Crop score events to the configured score range.

        The selected range is determined from ``start_msr``, ``start_offset``,
        ``end_msr``, and ``end_offset``. Events are selected according to their
        ``start_quarter_length`` values.

        The start boundary is inclusive and the end boundary is exclusive.

        This method also updates ``score_start_ql`` and ``score_end_ql``.

        Args:
            score_events: Score events sorted by ``start_quarter_length``.

        Returns:
            Score events whose start positions fall within the selected range.
        """

        self.score_start_ql = self._get_ql_at_msr_and_offset(
            self.start_msr, self.start_offset
        )
        self.score_end_ql = self._get_ql_at_msr_and_offset(
            self.end_msr, self.end_offset
        )

        start_times = [event["start_quarter_length"] for event in score_events]

        start_idx = bisect_left(start_times, self.score_start_ql)
        end_idx = bisect_left(start_times, self.score_end_ql)

        return score_events[start_idx:end_idx]

    def score_events_add_times(self, score_events: list[dict]) -> list[dict]:
        """Add audio start and end times to score events.

        Score positions are converted from quarter lengths to seconds relative
        to the first supplied event. One complete measure is inserted before
        the first event to represent the audio count-in.

        Each event receives:

        - ``start_sec``: event onset time in seconds.
        - ``end_sec``: event offset time in seconds.

        The supplied event dictionaries are modified in place.

        Args:
            score_events: Score events containing ``start_quarter_length`` and
                ``end_quarter_length`` values.

        Returns:
            The same list with ``start_sec`` and ``end_sec`` added to each
            event.
        """

        start_ql = score_events[0]["start_quarter_length"]
        countdown = self._quarter_length_to_seconds(self._quarter_lengths_per_bar())

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

    def get_seconds_per_beat(self):
        """Return the duration of one tempo beat in seconds.

        Returns:
            Duration of one beat in seconds according to ``bpm``.
        """

        return 60 / self.bpm

    def _quarter_length_to_seconds(self, ql: float) -> float:
        """Convert quarter lengths to seconds.

        The conversion accounts for both tempo and the denominator of the
        configured time signature.

        Args:
            ql: Duration or score position expressed in quarter lengths.

        Returns:
            Equivalent duration in seconds.
        """

        denominator = self.time_signature[1]
        return ql * 60 / self.bpm * denominator / 4

    def _quarter_lengths_per_bar(self) -> float:
        """Return the duration of one measure in quarter lengths.

        Returns:
            Number of quarter lengths contained in one complete measure for the
            configured time signature.
        """

        numerator, denominator = self.time_signature
        return numerator * (4 / denominator)

    def _get_ql_at_msr_and_offset(self, msr: int, offset: float):
        """Convert a measure index and offset to an absolute score position.

        Args:
            msr: Zero-based measure index.
            offset: Offset from the beginning of the measure in quarter lengths.

        Returns:
            Absolute score position in quarter lengths.
        """

        return msr * self._quarter_lengths_per_bar() + offset

    def _validate(self):
        """Validate tempo, time signature, and selected score range parameters.

        Raises:
            ValueError: If ``bpm`` is not positive, ``time_signature`` is not a
                tuple containing two integers, either time-signature component
                is not positive, or a measure index or offset is negative.

        Todo:
            Validate that ``start_offset`` and ``end_offset`` do not exceed the
            length of their corresponding measures.
            Validate that the selected end position occurs after the selected
            start position.
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
