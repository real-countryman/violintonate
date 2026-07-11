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

    score_events = time_mapper.crop_score_events(score_events)
    score_events = time_mapper.score_events_add_times(score_events)

    for event in score_events:
        print(event)

    start_sec, end_sec = time_mapper.get_start_end_in_seconds()

    (f0, voiced_flags, voiced_probs), rms, times = PitchExtractor(
        audio, start_sec, end_sec
    ).extract_pitches_and_times()

    """
    for f0_value, voiced_flag_value, voiced_prob_value, rms_value, time in zip(
        f0, voiced_flags, voiced_probs, rms, times
    ):
        print(
            f"audio_properties: {(f0_value, voiced_flag_value, voiced_prob_value)} | "
            f"rms: {rms_value} | "
            f"time: {time}"
        )
    """
    """
    analyzer = IntotationAnalyzer(pitches, pitch_times, score_events)
    events_ok = analyzer.get_intonation_bool()
    bad_frames = analyzer.get_bad_frames()

    for event in events_ok:
        print(f"all: {event}")

    for frame in bad_frames:
        print(f"bad: {frame}")
    """

    first_msr_time = time_mapper.get_seconds_per_bar()

    pitch_filter = PitchFilter(
        f0, voiced_flags, voiced_probs, times, rms, msr_time_secs=first_msr_time
    )

    filtered_pitches, filtered_times = pitch_filter.filter_frames()

    for filtered_pitch, filtered_time in zip(filtered_pitches, filtered_times):
        print(f"filtered_pitch: {filtered_pitch} | filtered_time: {filtered_time}")


if __name__ == "__main__":
    main()
