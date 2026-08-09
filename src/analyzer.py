import numpy as np
from dataclasses import dataclass
import pandas as pd

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

    def add_tone_transition_times(self) -> None:
        for event in self.score_events[: len(self.score_events) - 2]:
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


@dataclass
class IntonationAnalyzer:
    # TODO intonation_tolerance_th (in cents)
    pitches: np.ndarray
    pitch_times: np.ndarray
    # TODO make it an argument not a property
    score_events: list[dict]

    # 1 semitone = 100 cents, may need tweaking
    INTONATION_TOLERANCE_CENTS = 10.0

    def get_intonation(self) -> list[tuple[float, float, float | None]]:
        result = []

        for pitch, pitch_time in zip(self.pitches, self.pitch_times):
            cent_deviation = self._compare_pitch_with_score_event(
                float(pitch),
                float(pitch_time),
            )
            result.append((float(pitch), float(pitch_time), cent_deviation))

        return result

    def get_bad_frames(self) -> list[tuple[float, float]]:
        analyzed_frames = self.get_intonation()

        return [
            (pitch_time, pitch, cent_deviation)
            for pitch_time, pitch, cent_deviation in analyzed_frames
            if cent_deviation is not None
            and abs(cent_deviation) > self.INTONATION_TOLERANCE_CENTS
        ]

    def _compare_pitch_with_score_event(
        self,
        pitch: float,
        pitch_time: float,
    ) -> float | None:
        score_event = self._find_score_event_at_time(pitch_time)

        if score_event is None:
            return False

        expected_midi = score_event["midi"]
        expected_midi_value = float(expected_midi[0])

        cents_error = (pitch - expected_midi_value) * 100.0

        return cents_error

    def _find_score_event_at_time(
        self,
        pitch_time: float,
    ) -> dict | None:
        for event in self.score_events:
            if event["kind"] != "note":
                continue

            if event["start_sec"] <= pitch_time < event["end_sec"]:
                return event

        return None


@dataclass
class RhythmAnalyzer:
    score_events: list[dict]

    def __post_init__(self):
        self._validate()

    def add_rhythm_onset_offset_diffs(self):
        for cur_event, next_event in zip(self.score_events, self.score_events[1:]):
            cur_event.setdefault("rhythm", {}).update(
                {
                    "onset_diff_secs": None,
                    "offset_diff_secs": None,
                }
            )

            next_event.setdefault("rhythm", {}).update(
                {
                    "onset_diff_secs": None,
                    "offset_diff_secs": None,
                }
            )

            print("There I choked:")
            print(cur_event)
            print("-----------------------------")
            # if there is pitch transition, compute by frequency change
            if cur_event["pitch"]["transition_time"] != None:
                print("I AM ALIVE!")
                print(cur_event)
                print("--------------------------------------")
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
                    next_event["rhythm"]["rms_onset_time"] = (
                        next_event["rms"]["rms_onset_time"] - next_event["start_sec"]
                    )

    def _validate(self): ...
