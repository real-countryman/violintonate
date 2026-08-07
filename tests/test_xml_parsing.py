import pytest

from src.xml_parsing import *


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
