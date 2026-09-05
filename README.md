# Violintonate

**Violintonate** is a prototype violin-practice analysis tool developed for **Booster Challenge 2026** at **VUT FIT**.

It compares a homemade violin recording with sheet music in **MusicXML** format and estimates:

- **Rhythmic accuracy**
- **Intonation accuracy**

The current prototype runs as a **command-line interface (CLI)** and produces a text table or JSON output together with an optional interactive/saved intonation graph.

## Overview

A user provides:

1. a violin audio recording,
2. a MusicXML score,
3. score and timing parameters such as tempo, time signature, measure range, and offsets.

The program parses the score, maps expected score events to recording time, extracts pitch and RMS features from the audio, detects rhythmic boundaries and pitch transitions, and evaluates the performance against the score.

## Features

The user can configure:

- score part number for multi-instrument scores,
- time signature (for example `4/4`, `3/4`, or `6/8`),
- start and end measures,
- start and end offsets in beats,
- tempo in beats per minute,
- text-table or JSON output,
- interactive graph display or graph saving.

The JSON output can include detected onset and offset times.

## Processing Pipeline

### 1. MusicXML Parser

The MusicXML score is parsed with the Python library **music21**.

Score events are stored with information such as:

- event type (`note` / `rest`),
- measure number,
- pitch as MIDI,
- start position,
- end position.

### 2. Score Time Mapper

The score is mapped to recording time using:

- BPM,
- time signature,
- selected measure range,
- beat offsets.

The mapper crops unused score events and calculates expected start and end times in seconds relative to the audio recording, assuming a one-bar count-in.

### 3. Pitch Extractor

Audio analysis is performed with **librosa**.

The extractor calculates:

- fundamental frequency using the **pYIN** algorithm,
- voiced/unvoiced flags,
- voiced probabilities,
- RMS values,
- frame timestamps.

Pitch values are converted to fractional MIDI values for later analysis.

### 4. Voiced Pitch Filter

Frames are removed when their pitch estimate is not considered reliable.

A frame is kept only when:

- it is marked as voiced, and
- its voiced probability is at least `0.5`.

### 5. RMS Onset / Offset Detection

RMS values are first smoothed using block-average smoothing.

Expected score-event boundaries are searched within a window of:

```text
±0.15 seconds
```

The detector selects:

- the rightmost local RMS minimum near an expected onset,
- the leftmost local RMS minimum near an expected offset.

### 6. Pitch Change Detection

For transitions from one note to another, pitch changes are used to locate the boundary more accurately.

The algorithm:

1. analyzes pitch frames on both sides of the expected transition,
2. smooths them with a rolling median,
3. estimates previous and next pitches,
4. searches for three consecutive frames matching the next pitch.

### 7. Rhythm Analysis

Two methods are used depending on the score event:

- **pitch-transition detection** for consecutive notes with different pitches,
- **RMS onset/offset detection** for cases such as repeated notes.

Timing differences are reported in seconds for analysis and in beats for user-facing output.

Rhythm evaluation uses a beat-difference tolerance of:

```text
BEAT_DIFF_TOLERANCE = 0.10
```

Classification:

| Difference | Result |
|---|---|
| `< 0.10` | perfect |
| `0.10 – < 0.20` | okay |
| `>= 0.20` | wrong |

### 8. Intonation Analysis

Each note is divided into three regions:

- **start:** `[0.00, 0.25)`
- **middle:** `[0.25, 0.75)`
- **end:** `[0.75, 1.00]`

Each analyzed pitch frame is compared with the expected score pitch.

The MIDI tolerance is:

```text
INTONATION_TOLERANCE_MIDI = 0.10
```

Classification:

| Pitch difference | Result |
|---|---|
| `< 0.10` | perfect |
| `0.10 – < 0.20` | okay |
| `>= 0.20` | wrong |

For each note section, the program reports the percentage of frames classified as perfect, okay, or wrong.

The overall flag is determined as follows:

- **wrong** if the wrong-frame ratio is at least `25%`,
- **perfect** if the perfect-frame ratio is at least `75%` and the wrong-frame ratio is at most `5%`,
- **okay** otherwise.

## Dependencies

Software:
- [music21](https://music21.org/music21docs/)
- [librosa](https://librosa.org/)
- [pandas] (https://pandas.pydata.org/)
- [pytest] (https://docs.pytest.org/en/stable/)

Documentation (work in progress):
- [sphinx] (https://www.sphinx-doc.org/en/master/)

> Use Make help for guidance with instalation and usage

## Input

The prototype expects:

- a homemade violin audio recording,
- sheet music in **MusicXML** format.

PDF sheet-music input is not currently supported and is listed as future work.

## Output

The program can produce:

- a text table,
- JSON containing timing information,
- an interactive intonation graph,
- a saved graph.

The feedback describes how accurately the performed rhythm and intonation match the score.

## Validation

The prototype was tested by cropping recordings in **Audacity** according to detected onset and offset times and checking the resulting notes by listening.

Testing included:

- scales,
- legato,
- détaché,
- different tempi,
- one practical repertoire recording.

The reported prototype successfully detects rhythm and intonation differences when the performer follows the metronome closely enough to remain inside the analysis windows.

The report notes that additional work is needed for fast playing; testing included sixteenth notes at up to approximately **120 BPM**.

## Current Limitations

The current implementation assumes relatively stable timing around the expected score positions. Known areas requiring further work include:

- fast-tempo pitch recognition,
- rubato / unstable BPM,
- vibrato detection,
- ornament detection,
- real-time feedback,
- PDF sheet-music input,
- graphical sheet-music feedback.

## Future Work

### Backend

- Real-time feedback
- Vibrato detection
- Rubato handling
- Improved fast-tempo pitch recognition
- Ornamentation detection and feedback
  - trills
  - turns
  - glissando
  - accents
  - etc.
- PDF sheet-music support

### GUI

- Sheet-music rendering with intuitive rhythm and intonation feedback
- User-friendly graphical interface
- Potential desktop or smartphone application

## Intended Use

Violintonate is intended as a violin practice tool for:

- students practicing rhythm and intonation,
- self-taught violinists,
- teachers supporting students' home practice.

A future GUI could combine the analysis backend with rendered sheet music and visual feedback.

## Project Status

**Prototype / experimental.**

The current version is a CLI tool that compares a violin performance with a MusicXML score. It is not yet a complete end-user desktop or mobile application.

## Author

**Dávid Novák**  
VUT FIT

## References

- music21 Development Team, *music21.converter — File and data conversion*
- music21 Development Team, *music21.note — Notes, rests, and lyrics*
- librosa Development Team, *librosa.pyin — Fundamental frequency estimation using probabilistic YIN*
- librosa Development Team, *librosa.feature.rms — Root-mean-square (RMS) computation*
- librosa Development Team, *librosa.times_like — Generate time values corresponding to a feature matrix*
- Che-Yuan Liang, Li Su, Yi-Hsuan Yang, and Hsin-Ming Lin, *Musical Offset Detection of Pitched Instruments: The Case of Violin*, ISMIR 2015
- Audacity Team, *Audacity*
