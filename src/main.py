import sys

from src.audio import *
from src.pitch import *
from src.xml_parsing import *
from src.graphs import *
from src.analyzer import *


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

    audio_start_sec, audio_end_sec = (
        time_mapper.get_seconds_start_end_audio_with_count_in()
    )

    (f0, voiced_flags, voiced_probs), rms, times = PitchExtractor(
        audio, audio_start_sec, audio_end_sec
    ).extract_pitches_and_times()

    pitch_filter = VoicedPitchFilter(
        f0,
        voiced_flags,
        voiced_probs,
        times,
        rms,
    )

    filtered_pitches, filtered_times = pitch_filter.filter_frames()
    filtered_pitches = hz_to_midi(filtered_pitches)

    th_estimator = RmsThresholdEstimator(rms, times, score_events)
    th_estimator.add_rms_offsets_onsets()

    onsets_offsets = th_estimator.get_rms_idxs_vals()

    normalized_rms, normalized_times = th_estimator._normalize_values()
    show_and_save_graph(
        normalized_times,
        normalized_rms,
        onsets_offsets,
        "Times",
        "RMS",
        "rms_graph_new",
    )

    pitch_change_detector = PitchChangeDetector(
        filtered_pitches, filtered_times, score_events
    )

    pitch_change_detector.add_tone_transitions_frequencies()
    pitch_change_detector.add_tone_transition_times()

    for event in score_events:
        print(event)


if __name__ == "__main__":
    main()
