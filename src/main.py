import sys
from src.audio import *
from src.pitch import *
from src.xml_parsing import *


def main():
    argc = len(sys.argv)
    argv = sys.argv

    np.set_printoptions(threshold=np.inf)

    if argc != 2:
        print("Usage: python3 main.py <audio_file>")
        return

    audio = Audio(argv[1], 76.0, (4, 4), 96)

    xml_file = MusicxmlParser("./input/xml_files/Quantum Occasu.xml", part_idx=0)
    score_events = xml_file.extract_score_events()

    time_mapper = ScoreTimeMapper(
        audio.bpm,
        audio.time_signature,
        start_msr=0,
        start_offset=0,
        end_msr=10,
        end_offset=0,
    )

    for event in score_events:
        print(event)

    start_sec, end_sec = time_mapper.get_start_end_in_seconds()

    pitches, pitch_times = PitchExtractor(
        audio, start_sec, end_sec
    ).extract_pitches_and_times(get_hz=True)

    for pitch, time in zip(pitches, pitch_times):
        print(f"pitch: {pitch}, time: {time}")


if __name__ == "__main__":
    main()
