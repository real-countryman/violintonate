import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

@dataclass
class Audio:
    path: str
    bpm: float
    time_signature: tuple[int, int]
    msr_cnt: int

    def _validate(self):
        """
        Helper function for validation of parameters
        """
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if self.bpm <= 0:
            raise ValueError("Bpm must be > 0")

        if self.time_signature[0] <= 0 or self.time_signature[1] <= 0:
            raise ValueError("Time signature must be at least (1/1) or more")
        
        if self.msr_cnt <= 0:
            raise ValueError("Msr_cnt must be > 0")