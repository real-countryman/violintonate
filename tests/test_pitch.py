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


class TestVoicedPitchFilter:
    def test_validate_ok(self):
        f0 = np.array([78.9, 90.3, 111.1])
        voiced_flags = np.array([True, False, True])
        voiced_probs = np.array([0.8, 0.3, 0.9])
        times = np.array([1, 1.2, 1.4])
        rms = np.array([50, 70, 50])
        msr_time_secs = 0.3

        pitch_filter = VoicedPitchFilter(
            f0=f0,
            voiced_flags=voiced_flags,
            voiced_probs=voiced_probs,
            times=times,
            rms=rms,
            msr_time_secs=msr_time_secs,
        )

    def test_validate_arr_not_same_len(self):
        f0 = np.array([90.3, 111.1])
        voiced_flags = np.array([True, False, True])
        voiced_probs = np.array([0.8, 0.3, 0.9])
        times = np.array([1, 1.2, 1.4])
        rms = np.array([50, 70, 50])
        msr_time_secs = 0.3

        with pytest.raises(ValueError):
            pitch_filter = VoicedPitchFilter(
                f0=f0,
                voiced_flags=voiced_flags,
                voiced_probs=voiced_probs,
                times=times,
                rms=rms,
                msr_time_secs=msr_time_secs,
            )

        f0 = np.array([1, 2, 3])
        voiced_flags = np.array([True, False])

        with pytest.raises(ValueError):
            pitch_filter = VoicedPitchFilter(
                f0=f0,
                voiced_flags=voiced_flags,
                voiced_probs=voiced_probs,
                times=times,
                rms=rms,
                msr_time_secs=msr_time_secs,
            )

        voiced_flags = np.array([True, False, True])
        voiced_probs = np.array([0.8, 0.3])

        with pytest.raises(ValueError):
            pitch_filter = VoicedPitchFilter(
                f0=f0,
                voiced_flags=voiced_flags,
                voiced_probs=voiced_probs,
                times=times,
                rms=rms,
                msr_time_secs=msr_time_secs,
            )

        voiced_probs = np.array([0.8, 0.3, 0.9])
        times = np.array([1, 2])

        with pytest.raises(ValueError):
            pitch_filter = VoicedPitchFilter(
                f0=f0,
                voiced_flags=voiced_flags,
                voiced_probs=voiced_probs,
                times=times,
                rms=rms,
                msr_time_secs=msr_time_secs,
            )

        times = np.array([1, 2, 3])
        rms = np.array([50, 70])

        with pytest.raises(ValueError):
            pitch_filter = VoicedPitchFilter(
                f0=f0,
                voiced_flags=voiced_flags,
                voiced_probs=voiced_probs,
                times=times,
                rms=rms,
                msr_time_secs=msr_time_secs,
            )
