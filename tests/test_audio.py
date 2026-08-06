import unittest
import tempfile
from pathlib import Path
import numpy as np
from math import log2

from src.audio import *


class TestHzToMidiMethod(unittest.TestCase):
    def test_ok_values(self):
        hz_values = np.array([20, 30, 40])
        midi_values = hz_to_midi(hz_values)

        for i in range(len(midi_values)):
            self.assertAlmostEqual(midi_values[i], 69 + 12 * log2(hz_values[i] / 440.0))

    def test_wrong_values(self):
        hz_values = np.array([20, 0, -10])

        with self.assertRaises(ValueError):
            hz_to_midi(hz_values)


class TestAudio(unittest.TestCase):
    def test_valid_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "audio.mp3"
            mp3_path.write_bytes(b"fake mp3 data")

            audio = Audio(
                path=mp3_path,
                bpm=78,
                time_signature=(4, 4),
                msr_cnt=20,
            )

            self.assertTrue(audio.path.exists())
            self.assertEqual(audio.path, mp3_path)

            self.assertEqual(audio.bpm, 78)
            self.assertEqual(audio.time_signature, (4, 4))
            self.assertEqual(audio.msr_cnt, 20)

    def test_file_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_file = Path(temp_dir) / "missing.mp3"

            self.assertFalse(missing_file.exists())

            with self.assertRaises(FileNotFoundError):
                Audio(
                    path=missing_file,
                    bpm=78,
                    time_signature=(4, 4),
                    msr_cnt=20,
                )

    def test_float_bpm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "audio.mp3"
            mp3_path.write_bytes(b"fake mp3 data")

            audio = Audio(
                path=mp3_path,
                bpm=78.5,
                time_signature=(4, 4),
                msr_cnt=20,
            )

            self.assertAlmostEqual(audio.bpm, 78.5)

    def test_negative_bpm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "audio.mp3"
            mp3_path.write_bytes(b"fake mp3 data")

            with self.assertRaises(ValueError):
                audio = Audio(
                    path=mp3_path,
                    bpm=-1,
                    time_signature=(4, 4),
                    msr_cnt=20,
                )

    def test_wrong_float_msr_cnt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "audio.mp3"
            mp3_path.write_bytes(b"fake mp3 data")

            with self.assertRaises(ValueError):
                audio = Audio(
                    path=mp3_path,
                    bpm=78,
                    time_signature=(4, 4),
                    msr_cnt=20.5,
                )

    def test_negative_msr_cnt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "audio.mp3"
            mp3_path.write_bytes(b"fake mp3 data")

            with self.assertRaises(ValueError):
                audio = Audio(
                    path=mp3_path,
                    bpm=78,
                    time_signature=(4, 4),
                    msr_cnt=-1,
                )


if __name__ == "__main__":
    unittest.main()
