import pytest
import tempfile
from pathlib import Path

from src.pitch import *


class TestPitchExtractor:
    @pytest.fixture(autouse=True)
    def setup_audio(self, tmp_path):
        mp3_path = tmp_path / "audio.mp3"
        mp3_path.write_bytes(b"fake mp3 data")

        self.audio = Audio(
            path=mp3_path,
            bpm=78,
            time_signature=(4, 4),
            msr_cnt=20,
        )

    def test_validation_ok(self):
        pitch_extractor = PitchExtractor(
            audio=self.audio,
            start_sec=3,
            end_sec=6,
        )

        assert pitch_extractor.audio is self.audio
        assert pitch_extractor.start_sec == 3
        assert pitch_extractor.end_sec == 6

    def test_negative_start_sec(self):
        with pytest.raises(ValueError):
            pitch_extractor = PitchExtractor(
                audio=self.audio,
                start_sec=-1,
                end_sec=5,
            )

    def test_negative_end_sec(self):
        with pytest.raises(ValueError):
            pitch_extractor = PitchExtractor(
                audio=self.audio,
                start_sec=2,
                end_sec=-1,
            )

    def test_measure_range_not_ok(self):
        with pytest.raises(ValueError):
            pitch_extractor = PitchExtractor(
                audio=self.audio,
                start_sec=2,
                end_sec=1,
            )
