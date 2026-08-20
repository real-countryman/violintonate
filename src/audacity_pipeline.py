#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

SOURCE_TRACK = 0


def read_input():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON on stdin: {e}")

    if not isinstance(data, list):
        raise SystemExit(
            "Expected a JSON array:\n" '[{"start_sec": 1.0, "end_sec": 2.0}, ...]'
        )

    segments = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"Item {index} must be a JSON object")

        try:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except KeyError as e:
            raise SystemExit(f"Item {index}: missing field '{e.args[0]}'")
        except (TypeError, ValueError):
            raise SystemExit(f"Item {index}: start_sec and end_sec must be numbers")

        if start_sec < 0:
            raise SystemExit(f"Item {index}: start_sec cannot be negative")

        if end_sec <= start_sec:
            raise SystemExit(f"Item {index}: end_sec must be greater than start_sec")

        segments.append((start_sec, end_sec))

    if not segments:
        raise SystemExit("Input JSON contains no segments")

    return segments


def get_arguments():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python3 -m src.audacity_pipeline "
            "AUDIO_FILE OUTPUT_PROJECT.aup3 < timestamps.json"
        )

    audio_file = Path(sys.argv[1]).expanduser().resolve()
    output_project = Path(sys.argv[2]).expanduser().resolve()

    if not audio_file.is_file():
        raise SystemExit(f"Audio file does not exist:\n{audio_file}")

    if output_project.suffix.lower() != ".aup3":
        output_project = output_project.with_suffix(".aup3")

    output_project.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return audio_file, output_project


def connect_to_audacity():
    uid = os.getuid()

    to_pipe = f"/tmp/audacity_script_pipe.to.{uid}"
    from_pipe = f"/tmp/audacity_script_pipe.from.{uid}"

    if not os.path.exists(to_pipe):
        raise SystemExit(
            f"Audacity pipe not found:\n{to_pipe}\n\n"
            "Make sure Audacity is running and "
            "mod-script-pipe is enabled."
        )

    if not os.path.exists(from_pipe):
        raise SystemExit(
            f"Audacity pipe not found:\n{from_pipe}\n\n"
            "Make sure Audacity is running and "
            "mod-script-pipe is enabled."
        )

    to_file = open(to_pipe, "w")
    from_file = open(from_pipe, "r")

    return to_file, from_file


def do_command(to_file, from_file, command):
    print(f">>> {command}", file=sys.stderr)

    to_file.write(command + "\n")
    to_file.flush()

    response = []

    while True:
        line = from_file.readline()

        if not line:
            break

        if line.strip() == "":
            break

        response.append(line.rstrip())

    result = "\n".join(response)

    if result:
        print(result, file=sys.stderr)

    return result


def escape_audacity_string(value):
    value = str(value)

    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')

    return value


def import_audio(
    to_file,
    from_file,
    audio_file,
):
    filename = escape_audacity_string(audio_file)

    do_command(
        to_file,
        from_file,
        f'Import2: Filename="{filename}"',
    )


def select_segment(
    to_file,
    from_file,
    source_track,
    start_sec,
    end_sec,
):
    do_command(
        to_file,
        from_file,
        (
            "Select: "
            f"Track={source_track} "
            "TrackCount=1 "
            f"Start={start_sec:.9f} "
            f"End={end_sec:.9f} "
            "RelativeTo=ProjectStart "
            "Mode=Set"
        ),
    )


def duplicate_segment(
    to_file,
    from_file,
    source_track,
    start_sec,
    end_sec,
):
    select_segment(
        to_file,
        from_file,
        source_track,
        start_sec,
        end_sec,
    )

    do_command(
        to_file,
        from_file,
        "Duplicate:",
    )


def save_project(
    to_file,
    from_file,
    output_project,
):
    filename = escape_audacity_string(output_project)

    do_command(
        to_file,
        from_file,
        ("SaveProject2: " f'Filename="{filename}" ' "AddToHistory=1 " "Compress=0"),
    )


def main():
    audio_file, output_project = get_arguments()
    segments = read_input()

    print(f"Audio: {audio_file}", file=sys.stderr)
    print(f"Project: {output_project}", file=sys.stderr)
    print(
        f"Segments: {len(segments)}",
        file=sys.stderr,
    )

    to_file, from_file = connect_to_audacity()

    processed = 0

    try:
        # Import the original audio as track 0.
        print(
            "\nImporting audio...",
            file=sys.stderr,
        )

        import_audio(
            to_file,
            from_file,
            audio_file,
        )

        # Each timestamp becomes one new track.
        for index, (start_sec, end_sec) in enumerate(
            segments,
            start=1,
        ):
            print(
                f"\n[{index}/{len(segments)}] "
                f"{start_sec:.6f}s -> "
                f"{end_sec:.6f}s",
                file=sys.stderr,
            )

            duplicate_segment(
                to_file=to_file,
                from_file=from_file,
                source_track=SOURCE_TRACK,
                start_sec=start_sec,
                end_sec=end_sec,
            )

            processed += 1

        print(
            "\nSaving project...",
            file=sys.stderr,
        )

        save_project(
            to_file,
            from_file,
            output_project,
        )

    finally:
        to_file.close()
        from_file.close()

    print(
        json.dumps(
            {
                "ok": True,
                "audio_file": str(audio_file),
                "project": str(output_project),
                "segments_processed": processed,
                "tracks_created": processed,
            }
        )
    )


if __name__ == "__main__":
    main()
