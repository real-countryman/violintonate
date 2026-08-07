import pytest
import math

from src.xml_parsing import *


def get_seconds_per_bar(time_signature: tuple[int, int], bpm: float):
    SECONDS_PER_MINUTE = 60

    return SECONDS_PER_MINUTE / bpm * time_signature[0]


# TODO offset bigger than measure beat length
# TODO end + offset > start + offset
class TestScoreTimeMapperValidation:
    def test_validation_ok(self):
        bpm = 78.3
        time_signature = (4, 4)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        time_mapper = ScoreTimeMapper(
            bpm,
            time_signature,
            start_msr,
            start_offset,
            end_msr,
            end_offset,
        )

    def test_negative_bpm(self):
        bpm = -1
        time_signature = (4, 4)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

    def test_wrong_time_signature(self):
        bpm = 78.3
        time_signature = (-1, 4)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

        bpm = 78.3
        time_signature = (4, -1)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

    def test_negative_start_msr(self):
        bpm = 78.3
        time_signature = (4, 4)
        start_msr = -1
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

    def test_negative_start_offset(self):
        bpm = 78.3
        time_signature = (4, 4)
        start_msr = 3
        start_offset = -1
        end_msr = 7
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

    def test_negative_end_msr(self):
        bpm = 78.3
        time_signature = (4, 4)
        start_msr = 3
        start_offset = 0.5
        end_msr = -1
        end_offset = 0.75

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )

    def test_negative_end_offset(self):
        bpm = 78.3
        time_signature = (4, 4)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = -1

        with pytest.raises(ValueError):
            time_mapper = ScoreTimeMapper(
                bpm,
                time_signature,
                start_msr,
                start_offset,
                end_msr,
                end_offset,
            )


class TestScoreTimeMapperFunctions:
    @pytest.fixture(autouse=True)
    def setup_score_time_mapper(self):
        bpm = 78.0
        time_signature = (6, 8)
        start_msr = 3
        start_offset = 0.5
        end_msr = 7
        end_offset = 0.75

        self.time_mapper = ScoreTimeMapper(
            bpm,
            time_signature,
            start_msr,
            start_offset,
            end_msr,
            end_offset,
        )

    def test_get_seconds_per_bar(self):
        secs_per_bar = get_seconds_per_bar(
            self.time_mapper.time_signature, self.time_mapper.bpm
        )

        pytest.approx(self.time_mapper.get_seconds_per_bar(), secs_per_bar)

    def test_score_events_add_times(self): ...

    def test_crop_score_events(self): ...

    def test_get_start_end_in_quarter_lengths(self): ...
