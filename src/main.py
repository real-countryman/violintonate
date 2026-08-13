from src.audio import *
from src.pitch import *
from src.xml_parsing import *
from src.graphs import *
from src.analyzer import *
from src.parser import *
from src.output import *


def main():
    args = parse_args()

    np.set_printoptions(threshold=np.inf)

    audio = Audio(args.audio, args.bpm, tuple(args.time_signature))

    xml_file = MusicxmlParser(args.musicxml, part_idx=0)
    score_events = xml_file.extract_score_events()

    start_msr = args.start_msr - 1  # one based idx -> zero based idx
    end_msr = args.end_msr - 1  # one based idx -> zero based idx

    time_mapper = ScoreTimeMapper(
        audio.bpm,
        audio.time_signature,
        start_msr=start_msr,
        start_offset=args.start_offset,
        end_msr=end_msr,
        end_offset=args.end_offset,
    )

    score_events = time_mapper.crop_score_events(score_events)
    score_events = time_mapper.score_events_add_times(score_events)

    audio_start_sec, audio_end_sec = (
        time_mapper.get_audio_bounds_with_count_in_and_end_padding()
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

    """
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
    """

    pitch_change_detector = PitchChangeDetector(
        filtered_pitches, filtered_times, score_events
    )

    pitch_change_detector.add_tone_transitions_frequencies()
    pitch_change_detector.add_tone_transition_times()

    rhythm_analyzer = RhythmAnalyzer(score_events)
    rhythm_analyzer.add_rhythm_onset_offset_diffs_and_flags()

    intonation_analyzer = IntonationAnalyzer(
        filtered_pitches,
        filtered_times,
        score_events,
    )

    intonation_analyzer.add_intonation()

    cli_output = Output(score_events)
    cli_output.print_cli_output()


if __name__ == "__main__":
    main()
