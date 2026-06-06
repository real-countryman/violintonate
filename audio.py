import librosa
import numpy as np

# Load audio
y, sr = librosa.load("audio.mp3", sr=None)

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

print(pitches[:100])       # Hz values
print(pitch_times[:100])   # times in seconds