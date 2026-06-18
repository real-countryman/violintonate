import unittest
from unittest.mock import patch

import numpy as np
from audio import *


class TestAudioMethods(unittest.TestCase):

    def test_seconds_per_bar(self):
        audio = Audio(
            path="audio.mp3",
            bpm=120,
            time_signature=(4, 4),
            msr_cnt=8,
        )

        # 4 beats per bar * 60 seconds / 120 bpm = 2 seconds per bar
        self.assertAlmostEqual(audio._seconds_per_bar, 2.0)

    def test_bpm_to_secs(self):
        audio = Audio(
            path="audio.mp3",
            bpm=120,
            time_signature=(4, 4),
            msr_cnt=8,
        )

        # At 120 bpm, each beat is 0.5 seconds.
        # Measure 0, offset 0 -> 0 seconds
        self.assertAlmostEqual(audio._bpm_to_secs(0, 0), 0.0)

        # Measure 1 starts after one 4/4 bar -> 2 seconds
        self.assertAlmostEqual(audio._bpm_to_secs(1, 0), 2.0)

        # Measure 2 starts at 4 seconds.
        # Offset 2 beats adds 1 second.
        self.assertAlmostEqual(audio._bpm_to_secs(2, 2), 5.0)

        #TODO get_pitches_and_times(...)

if __name__ == "__main__":
    unittest.main()