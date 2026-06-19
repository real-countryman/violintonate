import sys
from audio import *

def main():
    argc = len(sys.argv)
    argv = sys.argv

    if argc != 2:
        print("Usage: python3 main.py <audio_file>")
        return

    audio = Audio(argv[1], 110, (4,4), 6)

    pitches, times = audio.get_pitches_and_times(0, 0, 1)
    
    np.set_printoptions(threshold=np.inf)
    print(pitches)
    print(times)

if __name__ == "__main__":
    main()