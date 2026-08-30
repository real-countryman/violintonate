import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio recording file")
    parser.add_argument("musicxml", help="Path to MusicXML sheet music file")
    parser.add_argument(
        "--part_idx",
        required=False,
        type=int,
        default=0,
        help="Zero based index of part to use if there are more parts in sheet music",
    )
    parser.add_argument(
        "--bpm", required=True, type=float, help="Tempo of the recording"
    )
    parser.add_argument(
        "--time_signature",
        required=True,
        type=int,
        nargs=2,
        help="Numerator and denominator of the time signature:\n\t--time_signature 4 4",
    )
    parser.add_argument(
        "--start_msr",
        required=False,
        default=1,
        type=int,
        help="One based index of start measure",
    )
    parser.add_argument(
        "--end_msr",
        required=True,
        type=int,
        help="One based inex of end measure",
    )
    parser.add_argument(
        "--start_offset",
        required=False,
        type=float,
        default=0.0,
        help="Start measure decimal offset measured in beats. Default is 0.",
    )
    parser.add_argument(
        "--end_offset",
        required=False,
        type=float,
        default=0.0,
        help="End measure decimal offset measured in beats. Default is 0.",
    )
    parser.add_argument(
        "--audacity_json",
        action="store_true",
        help="Prints json with onset secs and offset secs for audacity script.",
    )
    parser.add_argument(
        "--no_show_graph",
        action="store_true",
        help="Disables showing of a graph.",
    )
    parser.add_argument(
        "--no_save_graph",
        action="store_true",
        help="Disables saving of the graph.",
    )
    parser.add_argument(
        "--no_cli_output",
        action="store_true",
        help="Disables CLI output with playing info.",
    )
    parser.add_argument(
        "--rms_graph",
        action="store_true",
        help="Shows and saves RMS graph of the interpretation. Used mainly for debugging.",
    )

    return parser.parse_args()
