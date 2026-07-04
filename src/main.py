import sys
from src.audio import *
from src.pitch import *
from src.xml_parsing import *


def main():
    argc = len(sys.argv)
    argv = sys.argv

    if argc != 2:
        print("Usage: python3 main.py <audio_file>")
        return

    audio = Audio(argv[1], 76.0, (4, 4), 96)

    pitches, times = PitchExtractor(audio).extract_pitches_and_times(
        end_msr=10, get_hz=False
    )

    np.set_printoptions(threshold=np.inf)
    for pitch, time in zip(pitches, times):
        print((pitch, time))

    xml_file = MusicxmlParser("./input/xml_files/Quantum Occasu.xml", part_idx=0)
    score_events = xml_file.extract_score_events()

    for score_event in score_events:
        print(score_event)


if __name__ == "__main__":
    main()
