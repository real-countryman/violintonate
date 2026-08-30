from src.audio import *
from src.pitch import *
from src.xml_parsing import *
from src.analyzer import *
from src.parser import *
from src.output import *
from src.graphs import show_and_save_rms_graph


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

    rms_onset_offset_detector = RmsOnsetOffsetDetector(rms, times, score_events)
    rms_onset_offset_detector.add_rms_offsets_onsets()

    if args.rms_graph:
        onsets_offsets = rms_onset_offset_detector.get_rms_idxs_vals()
        mean_rms, mean_times = rms_onset_offset_detector._normalize_values()

        show_and_save_rms_graph(
            mean_times,
            mean_rms,
            onsets_offsets,
            "Times",
            "RMS",
            "RMS Onsets, Offsets",
        )

    pitch_change_detector = PitchChangeDetector(
        filtered_pitches, filtered_times, score_events
    )

    pitch_change_detector.add_tone_transitions_frequencies()
    pitch_change_detector.add_tone_transition_times()

    rhythm_analyzer = RhythmAnalyzer(score_events)
    rhythm_analyzer.add_rhythm_onset_offset_diffs_and_flags(
        time_mapper.get_seconds_per_beat()
    )

    intonation_analyzer = IntonationAnalyzer(
        filtered_pitches,
        filtered_times,
        score_events,
    )

    intonation_analyzer.add_intonation()

    output = Output(score_events)
    if args.audacity_json:
        output.print_rhythm_json()
    if not args.no_cli_output:
        output.print_cli_output()

    show_graph_bool = True
    save_graph_bool = True
    if args.no_show_graph:
        show_graph_bool = False
    if args.no_save_graph:
        save_graph_bool = False

    if show_graph_bool or save_graph_bool:
        output.render_and_save_graph_output(show_graph_bool, save_graph_bool)


if __name__ == "__main__":
    main()
