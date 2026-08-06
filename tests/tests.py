import unittest
import tempfile
from pathlib import Path

from src.audio import Audio


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
