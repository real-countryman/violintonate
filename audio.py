import librosa
import numpy as np
from calculations import *

def get_pitches_and_times(audio: Audio, start_msr: int, msr_offset: int, end_msr: int):
    start = bpm_to_secs(audio, start_msr, msr_offset)
    end = bpm_to_secs(audio, end_msr, msr_offset=0)
    dur = end - start
    # Load audio
    # audio signal, sample rate
    y, sr = librosa.load(audio.path, sr=None, offset=start, duration=dur)

    # Estimate pitch / fundamental frequency
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("G3"),
        fmax=librosa.note_to_hz("E7"),
        sr=sr
    )

    # Time axis for each pitch estimate
    times = librosa.times_like(f0, sr=sr)

    # Keep only voiced frames
    pitches = f0[voiced_flag]
    pitch_times = times[voiced_flag]

    return pitches, pitch_times # Hz values, times in seconds