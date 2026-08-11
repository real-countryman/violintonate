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

    return parser.parse_args()
