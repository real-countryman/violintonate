from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from bisect import bisect_left

from src.audio import Audio

FRAME_LENGTH = 2048
HOP_LENGTH = 256
NOTE_BOUNDARY_SEARCH_RADIUS_SEC = 0.15


def hz_to_midi(pitches_hz: np.ndarray) -> np.ndarray:
    return 69 + 12 * np.log2(pitches_hz / 440.0)


# TODO make NOTE_BOUNDARY_SEARCH_RADIUS_SEC a parameter
def get_start_end_time_idx(times: np.ndarray, exp_time: float) -> tuple[int, int]:

    max_time = exp_time + NOTE_BOUNDARY_SEARCH_RADIUS_SEC
    min_time = exp_time - NOTE_BOUNDARY_SEARCH_RADIUS_SEC

    left_idx = np.searchsorted(times, min_time, side="left")
    right_idx = np.searchsorted(times, max_time, side="right")

    return left_idx, right_idx


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
        times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH) + self.start_sec

        # Compute rms
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]

        return (f0, voiced_flag, voiced_prob), rms, times

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
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )

        return f0, voiced_flag, voiced_prob


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
class VoicedPitchFilter:
    f0: np.ndarray
    voiced_flags: np.ndarray
    voiced_probs: np.ndarray
    times: np.ndarray
    rms: np.ndarray
    msr_time_secs: float

    # May need tweaking
    VOICED_PROB_THRESHOLD = 0.8
    RMS_THRESHOLD = -1

    def __post_init__(self):
        self._set_rms_db_threshold()

    def filter_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes unreliable pitch frames.

        A frame is kept only if it is voiced, has a valid f0 value, has a voiced
        probability above VOICED_PROB_THRESHOLD, and is loud enough according to
        RMS_THRESHOLD.

        Returns:
            A tuple containing:

                pitches:
                    Filtered pitch values in Hertz.

                pitch_times:
                    Timestamps corresponding to the filtered pitch values.
        """
        mask = (
            (self.voiced_flags == True)
            & ~np.isnan(self.f0)
            & (self.voiced_probs > self.VOICED_PROB_THRESHOLD)
            & (self.rms > self.RMS_THRESHOLD)
        )

        times_clean = self.times[mask]
        f0_clean = self.f0[mask]

        pitches = f0_clean
        pitch_times = times_clean

        return pitches, pitch_times

    def _set_rms_db_threshold(self):
        """
        Sets RMS_THRESHOLD based on the average RMS value
        during the first measure, which is assumed to be quiet/count-in.
        """

        first_measure_end_idx = np.searchsorted(self.times, self.msr_time_secs)

        quiet_rms = self.rms[:first_measure_end_idx]

        if quiet_rms.size == 0:
            raise ValueError("No RMS frames found in the first measure.")

        quiet_avg_db = np.average(quiet_rms)

        self.RMS_THRESHOLD = 2 * quiet_avg_db


# TODO think of modifiyng times and rms, maybe main should handle it?
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

        for i in range(len(y) - 2, 0, -1):
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
                return i, y[i]

        return None, None

    def _leftmost_local_minimum(self, y: np.ndarray):
        y = np.asarray(y)

        for i in range(1, len(y) - 1):
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
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
            cur_event["tone_transition"] = {
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

            cur_event["tone_transition"] = {
                "cur_pitch": cur_pitch,
                "next_pitch": next_pitch,
            }

        self.score_events[-1]["tone_transition"] = {
            "cur_pitch": None,
            "next_pitch": None,
        }

    def add_tone_transition_times(self) -> None:
        for event in self.score_events[: len(self.score_events) - 2]:
            left_idx, right_idx = get_start_end_time_idx(
                self.pitch_times, float(event["end_sec"])
            )

            next_pitch = event["tone_transition"]["next_pitch"]
            if next_pitch is None:
                event["tone_transition"]["transition_time"] = None
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
                    event["tone_transition"]["transition_time"] = self.pitch_times[
                        left_idx + idx_shift
                    ]
                    break
                else:
                    idx_shift += 1

        self.score_events[-1]["tone_transition"]["transition_time"] = None

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
