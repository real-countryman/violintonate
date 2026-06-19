import sys
from src.audio import *
from src.xml_parsing import *

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

    xml_file = Musicxml_parser("./input/Canon_in_D.mxl")
    score_events = xml_file.extract_score_events()

    print(score_events)

if __name__ == "__main__":
    main()