import sys
from calculations import *
from audio import *

def main():
    argc = len(sys.argv)
    argv = sys.argv

    if argc != 2:
        print("Usage: python3 main.py <audio_file>")
        return

    audio = Audio(argv[1], 60, 30, 10)

    pitches, pitch_times = get_pitches_and_times(audio)
    
    print(f"{pitches} \n ---------------------- \n {pitch_times}")

if __name__ == "__main__":
    main()