import numpy as np
from dataclasses import dataclass
import pandas as pd
from bisect import bisect_left
from statistics import mean

NOTE_BOUNDARY_SEARCH_RADIUS_SEC = 0.15


# TODO make NOTE_BOUNDARY_SEARCH_RADIUS_SEC a parameter
def get_start_end_time_idx(times: np.ndarray, exp_time: float) -> tuple[int, int]:
    """Return the index range around an expected time value.

    The range is defined by NOTE_BOUNDARY_SEARCH_RADIUS_SEC on both sides
    of exp_time. The returned indices can be used directly for NumPy
    slicing as times[left_idx:right_idx].

    Args:
    times: Sorted array of time values in seconds.
    exp_time: Expected time in seconds around which to search.

    Returns:
    A tuple containing the left-inclusive and right-exclusive indices
    delimiting the search window in times.
    """

    max_time = exp_time + NOTE_BOUNDARY_SEARCH_RADIUS_SEC
    min_time = exp_time - NOTE_BOUNDARY_SEARCH_RADIUS_SEC

    left_idx = np.searchsorted(times, min_time, side="left")
    right_idx = np.searchsorted(times, max_time, side="right")

    return left_idx, right_idx


# TODO think of modifiyng times and rms, maybe main should handle it?
# TODO offsets too late
@dataclass
class RmsOnsetOffsetDetector:
    """Detect RMS-based onset and offset positions for score events.

    The detector smooths the RMS signal and corresponding time values by
    averaging them in fixed-size groups. It then searches for local minima
    around the expected onset and offset times of each score event.

    Detected RMS indices, values, and times are stored in the ``rms`` field
    of each score event.

    Attributes:
        rms: Array of RMS values.
        times: Array of time values in seconds corresponding to ``rms``.
        score_events: Score events containing expected start and end times.
    """

    rms: np.ndarray
    times: np.ndarray
    # TODO make an argument, not a class property
    score_events: list[dict]

    GROUP_SIZE = 5

    def get_rms_idxs_vals(self) -> list[dict[np.int64, np.float32]]:
        """Return detected RMS onset and offset information for all score events.

        Returns:
            A list of dictionaries containing the detected onset and offset
            indices, RMS values, and times for each score event.

        Raises:
            ValueError: If any score event is missing ``start_sec`` or ``end_sec``.
        """

        result: list[dict[np.int64, np.float32]] = []

        if any(
            "start_sec" not in event or "end_sec" not in event
            for event in self.score_events
        ):
            raise ValueError("Some events are missing RMS info")

        for event in self.score_events:
            vals = {
                "rms_offset_idx": event["rms"]["rms_offset_idx"],
                "rms_offset_value": event["rms"]["rms_offset_value"],
                "rms_onset_idx": event["rms"]["rms_onset_idx"],
                "rms_onset_value": event["rms"]["rms_onset_value"],
                "rms_onset_time": event["rms"]["rms_onset_time"],
                "rms_offset_time": event["rms"]["rms_offset_time"],
            }

            result.append(vals)

        return result

    def add_rms_offsets_onsets(self) -> None:
        """Detect and add RMS onset and offset information to score events.

        The RMS signal and corresponding time values are first smoothed by
        averaging consecutive groups of samples. Offset and onset positions are
        then detected as local minima near the expected event boundaries.
        """

        mean_rms, mean_times = self._normalize_values()
        self._add_rms_offsets(mean_rms, mean_times)
        self._add_rms_onsets(mean_rms, mean_times)

    def _normalize_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Smooth the RMS signal and corresponding time values.

        Values are divided into consecutive groups of ``GROUP_SIZE`` samples.
        Each group is replaced by its arithmetic mean.

        Returns:
            A tuple containing the smoothed RMS values and their corresponding
            mean time values.
        """

        return (
            np.array(
                [
                    chunk.mean()
                    for chunk in np.array_split(
                        self.rms,
                        np.arange(self.GROUP_SIZE, len(self.rms), self.GROUP_SIZE),
                    )
                ]
            ),
            np.array(
                [
                    chunk.mean()
                    for chunk in np.array_split(
                        self.times,
                        np.arange(self.GROUP_SIZE, len(self.times), self.GROUP_SIZE),
                    )
                ]
            ),
        )

    def _add_rms_offsets(self, rms: np.ndarray, times: np.ndarray) -> None:
        """Detect RMS offsets and add them to score events.

        For each event, a search window is created around its expected end time.
        The rightmost local minimum within that window is selected as the
        detected offset.

        Args:
            rms: Smoothed RMS values.
            times: Time values in seconds corresponding to ``rms``.
        """

        for event in self.score_events:
            left_idx, right_idx = get_start_end_time_idx(
                times,
                event["end_sec"],
            )

            local_idx, rms_value = self._rightmost_local_minimum(
                rms[left_idx:right_idx]
            )

            global_idx = None
            if local_idx is not None:
                global_idx = left_idx + local_idx

            event.setdefault("rms", {}).update(
                {
                    "rms_offset_idx": global_idx,
                    "rms_offset_value": rms_value,
                    "rms_offset_time": None,
                }
            )

            if global_idx is not None:
                event["rms"]["rms_offset_time"] = times[global_idx]

    def _add_rms_onsets(self, rms: np.ndarray, times: np.ndarray) -> None:
        """Detect RMS onsets and add them to score events.

        For each event, a search window is created around its expected start
        time. The leftmost local minimum within that window is selected as the
        detected onset.

        Args:
            rms: Smoothed RMS values.
            times: Time values in seconds corresponding to ``rms``.
        """

        for event in self.score_events:
            left_idx, right_idx = get_start_end_time_idx(
                times,
                event["start_sec"],
            )

            local_idx, rms_value = self._leftmost_local_minimum(rms[left_idx:right_idx])

            global_idx = None
            if local_idx is not None:
                global_idx = left_idx + local_idx

            event.setdefault("rms", {}).update(
                {
                    "rms_onset_idx": global_idx,
                    "rms_onset_value": rms_value,
                    "rms_onset_time": None,
                }
            )

            if global_idx is not None:
                event["rms"]["rms_onset_time"] = times[global_idx]

    def _rightmost_local_minimum(self, y: np.ndarray):
        """Find the rightmost local minimum in an array.

        Plateau minima are supported by allowing equality with neighboring
        values as long as the candidate is strictly smaller than at least one
        neighbor. If the global minimum occurs at an array boundary, that
        boundary is returned.

        Args:
            y: One-dimensional array in which to search.

        Returns:
            A tuple containing the index and value of the rightmost local
            minimum. Returns ``(None, None)`` if no local minimum is found.
        """

        y = np.asarray(y)
        if len(y) == 0:
            return None, None

        # if values always decreasing
        if y[-1] == min(y):
            return len(y) - 1, y[-1]

        # if values always increasing
        if y[0] == min(y):
            return 0, y[0]

        for i in range(len(y) - 2, 0, -1):
            if (
                y[i] <= y[i - 1]
                and y[i] <= y[i + 1]
                and (y[i] < y[i - 1] or y[i] < y[i + 1])  # [3,2,2,3] edge case
            ):
                return i, y[i]

        return None, None

    def _leftmost_local_minimum(self, y: np.ndarray):
        """Find the leftmost local minimum in an array.

        Plateau minima are supported by allowing equality with neighboring
        values as long as the candidate is strictly smaller than at least one
        neighbor. If the global minimum occurs at an array boundary, that
        boundary is returned.

        Args:
            y: One-dimensional array in which to search.

        Returns:
            A tuple containing the index and value of the leftmost local
            minimum. Returns ``(None, None)`` if no local minimum is found.
        """

        y = np.asarray(y)
        if y.size == 0:
            return None, None

        # if values always decreasing
        if y[-1] == min(y):
            return len(y) - 1, y[-1]

        # if values always increasing
        if y[0] == min(y):
            return 0, y[0]

        for i in range(1, len(y) - 1):
            if (
                y[i] <= y[i - 1]
                and y[i] <= y[i + 1]
                and (y[i] < y[i - 1] or y[i] < y[i + 1])  # [3,2,2,3] edge case
            ):
                return i, y[i]

        return None, None

    # TODO
    def _validate(self):
        """TODO:
        Validate the detector input data and configuration.

        Raises:
            ValueError: If the detector contains invalid or inconsistent input
                data.
        """

        return


@dataclass
class PitchChangeDetector:
    """Detect pitch changes and transition times between consecutive score events.

    The detector estimates the performed pitch before and after expected note
    boundaries using rolling medians. For transitions between different notes,
    it also determines when the performed pitch reaches and stabilizes around
    the following note's pitch.

    Detected pitch information is stored in the ``pitch`` field of each score
    event.

    Attributes:
        pitches: Array of detected pitch values in MIDI units.
        pitch_times: Array of time values in seconds corresponding to ``pitches``.
        score_events: Score events containing expected pitch and timing information.
    """

    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    ROLLING_MEDIAN_COUNT = 7
    SAME_PITCH_MIDI_TOLERANCE = 0.5

    def __post_init__(self):
        self._update_score_events()

    def add_tone_transitions_frequencies(self) -> None:
        """Estimate performed pitches before and after note transitions.

        For each pair of consecutive score events, transitions involving rests or
        identical pitches are ignored. Pitch values before and after the expected
        boundary are collected and processed using rolling medians. The resulting
        pitch estimates are stored as ``cur_pitch`` and ``next_pitch`` in the
        current event.

        The final score event is initialized without transition pitch information.
        """

        for cur_event, next_event in zip(self.score_events, self.score_events[1:]):
            # Both must be notes (no rests)
            if cur_event["kind"] != "note" or next_event["kind"] != "note":
                continue
            # Must be different pitches
            if cur_event["pitch_name"] == next_event["pitch_name"]:
                continue

            left_idx, right_idx = get_start_end_time_idx(
                self.pitch_times, cur_event["end_sec"]
            )

            left_values = self.pitches[left_idx - self.ROLLING_MEDIAN_COUNT : left_idx]

            right_values = self.pitches[
                right_idx : right_idx + self.ROLLING_MEDIAN_COUNT
            ]

            cur_pitch, next_pitch = self._get_cur_next_rolling_median_pitch(
                left_values, right_values
            )

            cur_event["pitch"]["cur_pitch"] = cur_pitch
            cur_event["pitch"]["next_pitch"] = next_pitch

        self.score_events[-1]["pitch"] = {
            "cur_pitch": None,
            "next_pitch": None,
        }

        # TODO last note
        if self.score_events[-1]["kind"] == "note":
            ...

    def add_tone_transition_times(self) -> None:
        """Detect and store pitch transition times.

        For each score event except the last one, a search window is created around
        its expected end time. The transition time is defined as the first time at
        which three consecutive detected pitch values are within
        ``SAME_PITCH_MIDI_TOLERANCE`` of the estimated next pitch.

        If no next pitch is available, the transition time is set to ``None``.
        The final score event always has no transition time.
        """

        for event in self.score_events[: len(self.score_events) - 1]:
            left_idx, right_idx = get_start_end_time_idx(
                self.pitch_times, float(event["end_sec"])
            )

            next_pitch = event["pitch"]["next_pitch"]
            if next_pitch is None:
                event["pitch"]["transition_time"] = None
                continue

            idx_shift = 0
            for pitch_1, pitch_2, pitch_3 in zip(
                self.pitches[left_idx:right_idx],
                self.pitches[left_idx + 1 : right_idx + 1],
                self.pitches[left_idx + 2 : right_idx + 2],
            ):
                if (
                    abs(pitch_1 - next_pitch) < self.SAME_PITCH_MIDI_TOLERANCE
                    and abs(pitch_2 - next_pitch) < self.SAME_PITCH_MIDI_TOLERANCE
                    and abs(pitch_3 - next_pitch) < self.SAME_PITCH_MIDI_TOLERANCE
                ):
                    event["pitch"]["transition_time"] = self.pitch_times[
                        left_idx + idx_shift
                    ]
                    break
                else:
                    idx_shift += 1

        self.score_events[-1]["pitch"]["transition_time"] = None

    def _get_rolling_medians_cur_next_values(self, left_values, right_values):
        """Calculate rolling median pitch values on both sides of a transition.

        A centered rolling median with a window size of three samples is applied
        separately to the pitch values before and after the expected transition.

        Args:
            left_values: Pitch values preceding the expected transition.
            right_values: Pitch values following the expected transition.

        Returns:
            A tuple containing the rolling median arrays for the left and right
            pitch values.
        """

        lv = pd.Series(left_values)
        rv = pd.Series(right_values)

        lv_rolling_median = (
            lv.rolling(
                window=3,
                center=True,
                min_periods=2,
            )
            .median()
            .to_numpy()
        )
        rv_rolling_median = (
            rv.rolling(
                window=3,
                center=True,
                min_periods=2,
            )
            .median()
            .to_numpy()
        )

        return lv_rolling_median, rv_rolling_median

    def _get_cur_next_rolling_median_pitch(self, left_values, right_values):
        """Estimate the current and next performed pitches around a transition.

        Rolling median filtering is first applied to pitch values on both sides of
        the expected transition. The representative pitch for each side is then
        calculated using the median of the filtered values while ignoring NaNs.

        Args:
            left_values: Pitch values preceding the expected transition.
            right_values: Pitch values following the expected transition.

        Returns:
            A tuple containing the estimated current pitch and next pitch in MIDI
            units.
        """

        left_pitches, right_pitches = self._get_rolling_medians_cur_next_values(
            left_values, right_values
        )

        left_pitch = np.nanmedian(left_pitches)
        right_pitch = np.nanmedian(right_pitches)

        return left_pitch, right_pitch

    def _update_score_events(self):
        """Initialize pitch analysis fields in all score events.

        Ensures that every score event contains a ``pitch`` dictionary with
        ``cur_pitch``, ``next_pitch``, and ``transition_time`` initialized to
        ``None``.
        """

        for event in self.score_events:
            event.setdefault("pitch", {}).update(
                {
                    "cur_pitch": None,
                    "next_pitch": None,
                    "transition_time": None,
                }
            )

    # TODO
    def _validate(self):
        """TODO:
        Validate the detector input data and configuration.

        Raises:
            ValueError: If the detector contains invalid or inconsistent pitch,
                timing, or score event data.
        """

        return


# TODO tendency
@dataclass
class IntonationAnalyzer:
    """Analyze performed pitch accuracy for individual score events.

    Each note event is divided into start, middle, and end sections based on
    its performed duration. Pitch frames within each section are compared with
    the expected MIDI pitch and classified as ``perfect``, ``okay``, or
    ``wrong``.

    For every section, the analyzer stores pitch differences, pitch times,
    individual intonation flags, flag ratios, tendency information, and an
    overall intonation flag.

    Attributes:
        pitches: Array of detected pitch values in MIDI units.
        pitch_times: Array of time values in seconds corresponding to ``pitches``.
        score_events: Score events containing pitch, timing, and rhythm
            information.
    """

    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    # 1 semitone = 100 cents, may need tweaking
    INTONATION_TOLERANCE_MIDI = 0.10
    START_SECTION_END_RATIO = 0.25
    END_SECTION_START_RATIO = 0.75

    def __post_init__(self):
        self._validate()
        self._update_score_events()

    # TODO event["intonation"]["end"] is always [] (empty)
    def add_intonation(self) -> None:
        """Analyze intonation for all note events.

        The performed start and end times are calculated using the detected onset
        and offset differences. Each note is then divided into start, middle, and
        end sections.

        Every pitch frame is compared with the expected pitch and assigned an
        intonation flag. Flag ratios and an overall intonation flag are subsequently
        calculated for each section.

        TODO:
        Add pitch tendency analysis for each section.
        """

        for event in self.score_events:
            if event["kind"] != "note":
                continue
            if (
                event["rhythm"]["onset_diff_secs"] == None
                or event["rhythm"]["offset_diff_secs"] == None
            ):
                continue

            # TODO exception, no onset / offset, make a helper function
            real_start_sec = event["start_sec"] + event["rhythm"]["onset_diff_secs"]
            real_end_sec = event["end_sec"] + event["rhythm"]["offset_diff_secs"]

            start_section_start_idx, end_section_end_idx = (
                self._get_start_end_idx_at_times(real_start_sec, real_end_sec)
            )

            start_section_end_idx, end_section_start_idx = (
                self._get_start_end_section_bounds_idxs(real_start_sec, real_end_sec)
            )

            expected_pitch = event["midi"][0]
            cnts = {
                "start": [0, 0, 0],
                "middle": [0, 0, 0],
                "end": [0, 0, 0],
            }
            for i in range(start_section_start_idx, end_section_end_idx):
                if i < start_section_end_idx:
                    section = "start"
                elif i < end_section_start_idx:
                    section = "middle"
                else:
                    section = "end"

                pitch = float(self.pitches[i])
                pitch_time = float(self.pitch_times[i])
                event["intonation"][section]["pitch_diffs"].append(
                    pitch - expected_pitch
                )

                event["intonation"][section]["pitch_times"].append(pitch_time)

                flag = self._get_intonation_flag(pitch, expected_pitch)
                event["intonation"][section]["pitch_flags"].append(flag)

                if flag == "perfect":
                    cnts[section][0] += 1
                elif flag == "okay":
                    cnts[section][1] = +1
                elif flag == "wrong":
                    cnts[section][2] += 1

            self._set_ratio_flags(event, cnts)
            self._set_tendency_flag(event)
            self._set_overall_flag(event)

    def _set_ratio_flags(
        self,
        event: dict,
        ratio_cnts: dict[list[int, int, int], list[int, int, int], list[int, int, int]],
    ) -> None:
        """Calculate and store intonation flag ratios for each note section.

        The numbers of ``perfect``, ``okay``, and ``wrong`` pitch frames are
        converted into ratios relative to the total number of analyzed frames in
        each section. Sections containing no analyzed frames are left unchanged.

        Args:
            event: Score event whose intonation information is updated.
            ratio_cnts: Mapping of section names to counts of perfect, okay, and
                wrong pitch frames.
        """

        sections = ratio_cnts.keys()
        for section in sections:
            perfect_cnt, okay_cnt, wrong_cnt = ratio_cnts[section]

            if perfect_cnt > 0 or okay_cnt > 0 or wrong_cnt > 0:
                sum_cnt = perfect_cnt + okay_cnt + wrong_cnt

                event["intonation"][section]["perfect_ratio"] = perfect_cnt / sum_cnt
                event["intonation"][section]["okay_ratio"] = okay_cnt / sum_cnt
                event["intonation"][section]["wrong_ratio"] = wrong_cnt / sum_cnt

    def _set_overall_flag(self, event: dict) -> None:
        """Determine the overall intonation flag for each note section.

        A section is classified as ``wrong`` when at least 25% of its analyzed
        frames are wrong. It is classified as ``perfect`` when at least 75% are
        perfect and at most 5% are wrong. All other valid sections are classified
        as ``okay``.

        Sections without calculated intonation ratios are ignored.

        Args:
            event: Score event whose section-level overall flags are updated.
        """

        for section in ["start", "middle", "end"]:
            event_section = event["intonation"][section]

            if (
                event_section["wrong_ratio"] == None
                or event_section["perfect_ratio"] == None
                or event_section["okay_ratio"] == None
            ):
                continue

            if event_section["wrong_ratio"] >= 0.25:
                event_section["overall_flag"] = "wrong"
            elif (
                event_section["perfect_ratio"] >= 0.75
                and event_section["wrong_ratio"] <= 0.05
            ):
                event_section["overall_flag"] = "perfect"
            else:
                event_section["overall_flag"] = "okay"

    def _set_tendency_flag(self, event: dict) -> None: ...

    """TODO:
    Determine the pitch tendency for each note section.

    The tendency describes the directional behavior of the performed pitch,
    such as being in tune, flat, sharp, or unstable.
    """

    # TODO optimise, so it takes event as an argument and makes
    # real_start_sec and real_end_sec computation inside the function
    def _get_start_end_idx_at_times(
        self, start_sec: float, end_sec: float
    ) -> tuple[int, int]:
        """Return pitch-array indices corresponding to a performed time interval.

        The start and end times are converted to indices in ``pitch_times`` using
        left-side binary insertion points.

        Args:
            start_sec: Start of the performed note interval in seconds.
            end_sec: End of the performed note interval in seconds.

        Returns:
            A tuple containing the start-inclusive and end-exclusive pitch indices.
        """

        return bisect_left(self.pitch_times, start_sec), bisect_left(
            self.pitch_times, end_sec
        )

    def _get_intonation_flag(self, pitch: float, expected_pitch: float) -> str:
        """Classify a performed pitch relative to the expected pitch.

        A pitch is classified as ``perfect`` when its absolute difference from the
        expected pitch is smaller than ``INTONATION_TOLERANCE_MIDI``. Differences
        below twice this tolerance are classified as ``okay``. All larger
        differences are classified as ``wrong``.

        Args:
            pitch: Performed pitch in MIDI units.
            expected_pitch: Expected pitch in MIDI units.

        Returns:
            ``"perfect"``, ``"okay"``, or ``"wrong"`` according to the pitch
            difference.
        """

        if abs(pitch - expected_pitch) < self.INTONATION_TOLERANCE_MIDI:
            return "perfect"
        elif abs(pitch - expected_pitch) < 2 * self.INTONATION_TOLERANCE_MIDI:
            return "okay"
        else:
            return "wrong"

    def _get_start_end_section_bounds_idxs(
        self,
        start_sec: float,
        end_sec: float,
    ) -> tuple[int, int]:
        """Return the indices separating the start, middle, and end note sections.

        The note duration is divided according to ``START_SECTION_END_RATIO`` and
        ``END_SECTION_START_RATIO``. The resulting time boundaries are converted
        to indices in ``pitch_times``.

        Args:
            start_sec: Performed note start time in seconds.
            end_sec: Performed note end time in seconds.

        Returns:
            A tuple containing the index marking the end of the start section and
            the index marking the beginning of the end section.
        """

        duration = end_sec - start_sec

        start_time_bound = start_sec + duration * self.START_SECTION_END_RATIO
        end_time_bound = start_sec + duration * self.END_SECTION_START_RATIO

        return (
            bisect_left(self.pitch_times, start_time_bound),
            bisect_left(self.pitch_times, end_time_bound),
        )

    def _update_score_events(self) -> None:
        """Initialize intonation analysis fields for all score events.

        Each event receives ``start``, ``middle``, and ``end`` intonation sections.
        Every section contains storage for pitch differences, pitch times,
        per-frame flags, flag ratios, tendency, and the overall intonation flag.
        """

        for event in self.score_events:
            event.setdefault("intonation", {}).update(
                {
                    "start": {
                        "pitch_diffs": [],
                        "pitch_times": [],
                        "pitch_flags": [],  # perfect / okay / wrong
                        #
                        "perfect_ratio": None,  # 0.72
                        "okay_ratio": None,  # 0.22
                        "wrong_ratio": None,  # 0.06
                        #
                        "tendency": None,  # okay / flat / sharp / unstable
                        "overall_flag": None,  # perfect / okay / wrong
                    },
                    #
                    "middle": {
                        "pitch_diffs": [],
                        "pitch_times": [],
                        "pitch_flags": [],
                        #
                        "perfect_ratio": None,
                        "okay_ratio": None,
                        "wrong_ratio": None,
                        #
                        "tendency": None,
                        "overall_flag": None,
                    },
                    #
                    "end": {
                        "pitch_diffs": [],
                        "pitch_times": [],
                        "pitch_flags": [],
                        #
                        "perfect_ratio": None,
                        "okay_ratio": None,
                        "wrong_ratio": None,
                        #
                        "tendency": None,
                        "overall_flag": None,
                    },
                }
            )

    def _validate(self):
        """TODO:
        Validate the intonation analyzer input data.
        """
        ...


@dataclass
class RhythmAnalyzer:
    """Analyze rhythmic onset and offset accuracy for score events.

    The analyzer compares detected note boundaries with their expected score
    times. Pitch transition times are preferred when available; otherwise,
    RMS-based onset and offset times are used.

    Timing differences are stored in seconds and beats and classified as
    ``perfect``, ``okay``, or ``wrong`` according to configured tolerances.

    Attributes:
        score_events: Score events containing expected timing, RMS, and pitch
            transition information.
    """

    score_events: list[dict]

    # TODO may need tweaking
    ONSET_TOLERANCE_BEATS = 0.10
    OFFSET_TOLERANCE_BEATS = 0.10

    def __post_init__(self):
        self._validate()

    def add_rhythm_onset_offset_diffs_and_flags(self, beat_secs: float) -> None:
        """Calculate onset and offset timing differences and rhythm flags.

        For each score event, detected note boundaries are compared with their
        expected start and end times. Pitch transition times are used for
        boundaries between notes when available; otherwise, RMS onset and offset
        times are used.

        The resulting differences are stored in seconds and beats, and each
        boundary is assigned a rhythm accuracy flag.

        Args:
            beat_secs: Duration of one beat in seconds.
        """

        self._update_score_events()

        first_event = self.score_events[0]
        # compute by rms offset
        if first_event["rms"]["rms_onset_time"] != None:
            first_event["rhythm"]["onset_diff_secs"] = (
                first_event["rms"]["rms_onset_time"] - first_event["start_sec"]
            )

        for cur_event, next_event in zip(self.score_events, self.score_events[1:]):
            # if there is pitch transition, compute by frequency change
            if cur_event["pitch"]["transition_time"] != None:
                cur_event["rhythm"]["offset_diff_secs"] = (
                    cur_event["pitch"]["transition_time"] - cur_event["end_sec"]
                )
                next_event["rhythm"]["onset_diff_secs"] = (
                    cur_event["pitch"]["transition_time"] - next_event["start_sec"]
                )

            # else compute by rms
            else:
                if cur_event["rms"]["rms_offset_time"] != None:
                    cur_event["rhythm"]["offset_diff_secs"] = (
                        cur_event["rms"]["rms_offset_time"] - cur_event["end_sec"]
                    )

                if next_event["rms"]["rms_onset_time"] != None:
                    next_event["rhythm"]["onset_diff_secs"] = (
                        next_event["rms"]["rms_onset_time"] - next_event["start_sec"]
                    )

            self._set_onset_offset_flags(cur_event)
            self._add_diffs_in_beats(cur_event, beat_secs)

        last_event = self.score_events[-1]

        # if there is no pitch transition from previous event,
        if last_event["rhythm"]["onset_diff_secs"] == None:
            # compute by rms
            if last_event["rms"]["rms_onset_time"] != None:
                last_event["rhythm"]["onset_diff_secs"] = (
                    last_event["rms"]["rms_onset_time"] - last_event["start_sec"]
                )

        # compute last offset by rms:
        if last_event["rms"]["rms_offset_time"] != None:
            last_event["rhythm"]["offset_diff_secs"] = (
                last_event["rms"]["rms_offset_time"] - last_event["end_sec"]
            )

        self._set_onset_offset_flags(last_event)
        self._add_diffs_in_beats(last_event, beat_secs)

    def _add_diffs_in_beats(self, event: dict, beat_secs: float) -> None:
        """Convert onset and offset timing differences from seconds to beats.

        Differences are divided by the duration of one beat. Missing timing
        differences are ignored.

        Args:
            event: Score event whose rhythm information is updated.
            beat_secs: Duration of one beat in seconds.
        """

        if event["rhythm"]["onset_diff_secs"] != None:
            event["rhythm"]["onset_diff_beats"] = (
                event["rhythm"]["onset_diff_secs"] / beat_secs
            )

        if event["rhythm"]["offset_diff_secs"] != None:
            event["rhythm"]["offset_diff_beats"] = (
                event["rhythm"]["offset_diff_secs"] / beat_secs
            )

    def _set_onset_offset_flags(self, event) -> None:
        """Assign rhythm accuracy flags to an event's onset and offset.

        The absolute onset and offset timing differences are compared against
        their respective tolerances. Differences below the tolerance are
        classified as ``perfect``, differences below twice the tolerance as
        ``okay``, and larger differences as ``wrong``.

        Args:
            event: Score event whose onset and offset flags are updated.
        """

        # set onset flag
        if event["rhythm"]["onset_diff_secs"] != None:
            if abs(event["rhythm"]["onset_diff_secs"]) < self.ONSET_TOLERANCE_BEATS:
                event["rhythm"]["onset_diff_flag"] = "perfect"
            elif (
                abs(event["rhythm"]["onset_diff_secs"]) < 2 * self.ONSET_TOLERANCE_BEATS
            ):
                event["rhythm"]["onset_diff_flag"] = "okay"
            else:
                event["rhythm"]["onset_diff_flag"] = "wrong"

        # set offset flag
        if event["rhythm"]["offset_diff_secs"] != None:
            if abs(event["rhythm"]["offset_diff_secs"]) < self.OFFSET_TOLERANCE_BEATS:
                event["rhythm"]["offset_diff_flag"] = "perfect"
            elif (
                abs(event["rhythm"]["offset_diff_secs"])
                < 2 * self.OFFSET_TOLERANCE_BEATS
            ):
                event["rhythm"]["offset_diff_flag"] = "okay"
            else:
                event["rhythm"]["offset_diff_flag"] = "wrong"

    def _update_score_events(self) -> None:
        """Initialize rhythm analysis fields for all score events.

        Each event receives a ``rhythm`` dictionary containing onset and offset
        differences in seconds and beats, together with their corresponding
        accuracy flags.
        """

        for event in self.score_events:
            event.setdefault("rhythm", {}).update(
                {
                    "onset_diff_secs": None,
                    "offset_diff_secs": None,
                    "onset_diff_flag": None,
                    "offset_diff_flag": None,
                    #
                    "onset_diff_beats": None,
                    "offset_diff_beats": None,
                }
            )

    def _validate(self): ...

    """TODO:
    Validate the rhythm analyzer input data.
    """
