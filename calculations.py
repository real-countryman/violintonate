import math
from dataclasses import dataclass

@dataclass
class Audio:
    path: str
    bpm: float
    time_signature: tuple[int, int]
    msr_cnt: int

def hz_to_notes(hz: float):
    """Converts hertz to notes"""
    return 12 * math.log2(hz/440)

def seconds_per_bar(audio: Audio) -> float:
    """
    Calculates how many seconds per bar
    
    Args:
        audio: The audio dataclass to calculate

    Returns:
        How many seconds per bar
    """
    return audio.time_signature[0] * 60 / audio.bpm

def bpm_to_secs(audio: Audio, measure: int, msr_offset: int) -> float:
    """
    Converts bpm to seconds taking to consideration measure number and measure offset

    Args:
        audio: The audio dataclass to calculate
        measure: The measure number index (start at 0)
        msr_offset: The offset inside the measure (start at 0)

    Returns:
        The time in seconds
    """
    offset = seconds_per_bar(audio) / audio.time_signature[0] * msr_offset
    return seconds_per_bar(audio) * measure + offset