import sys

from src.audio import *
from src.pitch import *
from src.xml_parsing import *
from src.graphs import *


def main():
    argc = len(sys.argv)
    argv = sys.argv

    np.set_printoptions(threshold=np.inf)

    if argc != 2:
        print("Usage: python3 main.py <audio_file>")
        return

    audio = Audio(argv[1], 78.0, (4, 4), 96)

    xml_file = MusicxmlParser("./input/xml_files/Quantum Occasu.xml", part_idx=0)
    score_events = xml_file.extract_score_events()

    time_mapper = ScoreTimeMapper(
        audio.bpm,
        audio.time_signature,
        start_msr=9,
        start_offset=0,
        end_msr=17,
        end_offset=3.0,
    )

    score_events = time_mapper.crop_score_events(score_events)
    score_events = time_mapper.score_events_add_times(score_events)

    score_event_start_sec, score_event_end_sec = (
        time_mapper.get_seconds_start_end_score_events()
    )

    audio_start_sec, audio_end_sec = (
        time_mapper.get_seconds_start_end_audio_with_count_in()
    )

    print(
        f"score_event_start_sec: {score_event_end_sec}, score_event_end_sec: {score_event_end_sec}\n"
        f"audio_start_sec: {audio_start_sec}, audio_end_sec: {audio_end_sec}"
    )

    return

    (f0, voiced_flags, voiced_probs), rms, times = PitchExtractor(
        audio, start_sec, end_sec
    ).extract_pitches_and_times()

    first_msr_time = time_mapper.get_seconds_per_bar()

    pitch_filter = VoicedPitchFilter(
        f0, voiced_flags, voiced_probs, times, rms, msr_time_secs=first_msr_time
    )

    filtered_pitches, filtered_times = pitch_filter.filter_frames()

    filtered_pitches = hz_to_midi(filtered_pitches)

    intonation_analyzer = IntonationAnalyzer(
        filtered_pitches, filtered_times, score_events
    )

    bad_frames = intonation_analyzer.get_bad_frames()

    for pitch, pitch_time, cent_deviation in bad_frames:
        print(
            f"pitch: {pitch}, pitch_time: {pitch_time}, cent_deviation: {cent_deviation}"
        )
    return
    th_estimator = RmsThresholdEstimator(rms, times, score_events)
    th_estimator.add_rms_offsets_onsets()

    onsets_offsets = th_estimator.get_rms_idxs_vals()
    """
    for val in onsets_offsets:
        print(val)
    """
    # show_and_save_graph(times, rms, onsets_offsets, "Times", "RMS", "rms_graph_new")

    pitch_change_detector = PitchChangeDetector(
        filtered_pitches, filtered_times, score_events
    )

    pitch_change_detector.add_tone_transitions()

    for event in score_events:
        print(event)


if __name__ == "__main__":
    main()
