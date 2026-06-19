from music21 import converter, note, chord
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Musicxml_parser():
    path: str

    def extract_score_events(self):
        path = Path(self.path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        score = converter.parse(path, format="musicxml")
        events = []

        for el in score.recurse().notesAndRests:
            start_ql = float(el.getOffsetInHierarchy(score))
            duration_ql = float(el.duration.quarterLength)
            end_ql = start_ql + duration_ql

            if isinstance(el, note.Note):
                pitch_names = [el.pitch.nameWithOctave]
                midi = [el.pitch.midi]
                kind = "note"
            elif isinstance(el, chord.Chord):
                pitch_names = [p.nameWithOctave for p in el.pitches]
                midi = [p.midi for p in el.pitches]
                kind = "chord"
            elif isinstance(el, note.Rest):
                pitch_names = []
                midi = []
                kind = "rest"
            else:
                continue

            events.append({
                "kind": kind,
                "pitch_name": pitch_names,
                "midi": midi,
                "start_quarter_length": start_ql,
                "duration_quarter_length": duration_ql,
                "end_quarter_length": end_ql,
            })

        return events