import numpy as np
from dataclasses import dataclass
import pandas as pd
from bisect import bisect_left
from statistics import mean

NOTE_BOUNDARY_SEARCH_RADIUS_SEC = 0.15


# TODO make NOTE_BOUNDARY_SEARCH_RADIUS_SEC a parameter
def get_start_end_time_idx(times: np.ndarray, exp_time: float) -> tuple[int, int]:

    max_time = exp_time + NOTE_BOUNDARY_SEARCH_RADIUS_SEC
    min_time = exp_time - NOTE_BOUNDARY_SEARCH_RADIUS_SEC

    left_idx = np.searchsorted(times, min_time, side="left")
    right_idx = np.searchsorted(times, max_time, side="right")

    return left_idx, right_idx


# TODO think of modifiyng times and rms, maybe main should handle it?
# TODO offsets too late
@dataclass
class RmsThresholdEstimator:
    rms: np.ndarray
    times: np.ndarray
    # TODO make an argument, not a class property
    score_events: list[dict]

    GROUP_SIZE = 5

    def get_rms_idxs_vals(self) -> list[dict[np.int64, np.float32]]:
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
        mean_rms, mean_times = self._normalize_values()
        self._add_rms_offsets(mean_rms, mean_times)
        self._add_rms_onsets(mean_rms, mean_times)

    def _normalize_values(self) -> tuple[np.ndarray, np.ndarray]:
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
        y = np.asarray(y)

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
        y = np.asarray(y)

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
        return


@dataclass
class PitchChangeDetector:
    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    ROLLING_MEDIAN_COUNT = 7
    SAME_PITCH_MIDI_TOLERANCE = 0.5

    def add_tone_transitions_frequencies(self) -> None:
        for cur_event, next_event in zip(self.score_events, self.score_events[1:]):
            cur_event["pitch"] = {
                "cur_pitch": None,
                "next_pitch": None,
            }

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

            cur_event["pitch"] = {
                "cur_pitch": cur_pitch,
                "next_pitch": next_pitch,
            }

        self.score_events[-1]["pitch"] = {
            "cur_pitch": None,
            "next_pitch": None,
        }

        # TODO last note
        if self.score_events[-1]["kind"] == "note":
            ...

    def add_tone_transition_times(self) -> None:
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
        left_pitches, right_pitches = self._get_rolling_medians_cur_next_values(
            left_values, right_values
        )

        left_pitch = np.nanmedian(left_pitches)
        right_pitch = np.nanmedian(right_pitches)

        return left_pitch, right_pitch

    # TODO
    def _validate(self):
        return


# TODO tendency
@dataclass
class IntonationAnalyzer:
    pitches: np.ndarray
    pitch_times: np.ndarray
    score_events: list[dict]

    # 1 semitone = 100 cents, may need tweaking
    INTONATION_TOLERANCE_MIDI = 0.10
    START_SECTION_END_RATIO = 0.25
    END_SECTION_START_RATIO = 0.75

    def __post_init__(self):
        self._update_score_events()

    # TODO event["intonation"]["end"] is always [] (empty)
    def add_intonation(self) -> None:
        for event in self.score_events:
            if event["kind"] != "note":
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
        sections = ratio_cnts.keys()
        for section in sections:
            perfect_cnt, okay_cnt, wrong_cnt = ratio_cnts[section]

            if perfect_cnt > 0 or okay_cnt > 0 or wrong_cnt > 0:
                sum_cnt = perfect_cnt + okay_cnt + wrong_cnt

                event["intonation"][section]["perfect_ratio"] = perfect_cnt / sum_cnt
                event["intonation"][section]["okay_ratio"] = okay_cnt / sum_cnt
                event["intonation"][section]["wrong_ratio"] = wrong_cnt / sum_cnt

    def _set_overall_flag(self, event: dict) -> None:
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

    # TODO optimise, so it takes event as an argument and makes
    # real_start_sec and real_end_sec computation inside the function
    def _get_start_end_idx_at_times(
        self, start_sec: float, end_sec: float
    ) -> tuple[int, int]:
        return bisect_left(self.pitch_times, start_sec), bisect_left(
            self.pitch_times, end_sec
        )

    def _get_intonation_flag(self, pitch: float, expected_pitch: float) -> str:
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
        duration = end_sec - start_sec

        start_time_bound = start_sec + duration * self.START_SECTION_END_RATIO
        end_time_bound = start_sec + duration * self.END_SECTION_START_RATIO

        return (
            bisect_left(self.pitch_times, start_time_bound),
            bisect_left(self.pitch_times, end_time_bound),
        )

    def _update_score_events(self) -> None:
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


@dataclass
class RhythmAnalyzer:
    score_events: list[dict]

    # TODO may need tweaking
    ONSET_TOLERANCE_BEATS = 0.10
    OFFSET_TOLERANCE_BEATS = 0.10

    def __post_init__(self):
        self._validate()

    def add_rhythm_onset_offset_diffs_and_flags(self, beat_secs: float) -> None:
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

    def _add_diffs_in_beats(self, event: dict, beat_secs: float) -> None:
        if event["rhythm"]["onset_diff_secs"] != None:
            event["rhythm"]["onset_diff_beats"] = (
                event["rhythm"]["onset_diff_secs"] / beat_secs
            )

        if event["rhythm"]["offset_diff_secs"] != None:
            event["rhythm"]["offset_diff_beats"] = (
                event["rhythm"]["offset_diff_secs"] / beat_secs
            )

    def _set_onset_offset_flags(self, event) -> None:
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
        if abs(event["rhythm"]["offset_diff_secs"]) != None:
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
