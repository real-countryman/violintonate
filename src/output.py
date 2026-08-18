from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


# TODO intonation tendency flag
@dataclass
class Output:
    score_events: list[dict]

    def print_cli_output(self):
        for i in range(len(self.score_events)):
            if self.score_events[i]["kind"] != "note":
                continue

            print(
                "----------------------------------------------------------------------------"
            )

            note_info = [
                (
                    "Measure",
                    self.score_events[i]["msr_number"],
                ),
                ("Note", self.score_events[i]["pitch_name"][0]),
            ]

            for label, value in note_info:
                print(f"{label:<18}: {str(value):>13}")

            print(
                "----------------------------------------------------------------------------"
            )

            onset_diff_secs = self.score_events[i]["rhythm"]["onset_diff_secs"]
            offset_diff_secs = self.score_events[i]["rhythm"]["offset_diff_secs"]

            onset_diff_beats = self.score_events[i]["rhythm"]["onset_diff_beats"]
            offset_diff_beats = self.score_events[i]["rhythm"]["offset_diff_beats"]

            onset_diff_flag = self.score_events[i]["rhythm"]["onset_diff_flag"]
            offset_diff_flag = self.score_events[i]["rhythm"]["offset_diff_flag"]

            rhythm_info = [
                ("Rhythm", "Onset", "Offset"),
                ("Time difference", onset_diff_secs, offset_diff_secs),
                ("Beats difference:", onset_diff_beats, offset_diff_beats),
                ("Evaluation:", onset_diff_flag, offset_diff_flag),
            ]

            for label, onset, offset in rhythm_info:
                if isinstance(onset, (int, float)):
                    onset = f"{onset:+.2f}"
                if isinstance(offset, float):
                    offset = f"{offset:+.2f}"

                print(f"{label:<18}: {str(onset):>13} {str(offset):>13}")

            print(
                "----------------------------------------------------------------------------"
            )

            intonation = self.score_events[i]["intonation"]

            intonation_info = [("Intonation", "Perfect", "Okay", "Wrong", "Overall")]

            for position in ("start", "middle", "end"):
                row = (
                    position.capitalize(),
                    (
                        format(intonation[position]["perfect_ratio"], ".0%")
                        if intonation[position]["perfect_ratio"] != None
                        else "Undetected"
                    ),
                    (
                        format(intonation[position]["okay_ratio"], ".0%")
                        if intonation[position]["perfect_ratio"] != None
                        else "Undetected"
                    ),
                    (
                        format(intonation[position]["wrong_ratio"], ".0%")
                        if intonation[position]["perfect_ratio"] != None
                        else "Undetected"
                    ),
                    (
                        intonation[position]["overall_flag"].capitalize()
                        if intonation[position]["overall_flag"] != None
                        else "Undetected"
                    ),
                )

                intonation_info.append(row)

            for label, perfect, okay, wrong, flag in intonation_info:
                print(
                    f"{label:<18}: {str(perfect):>13} {str(okay):>13} {str(wrong):>13} {str(flag):>13}"
                )

            print(
                "----------------------------------------------------------------------------"
            )
            print() if i < len(self.score_events) - 1 else None

    def render_and_save_graph_output(self, save=True, show=True):
        y_vals_intonation = []
        y_vals_reference = []
        x_vals_beats = []
        colors = []

        color_map = {
            "wrong": "red",
            "okay": "yellow",
            "perfect": "green",
        }

        for event in self.score_events:
            all_diffs = (
                event["intonation"]["start"]["pitch_diffs"]
                + event["intonation"]["middle"]["pitch_diffs"]
                + event["intonation"]["end"]["pitch_diffs"]
            )

            all_flags = (
                event["intonation"]["start"]["pitch_flags"]
                + event["intonation"]["middle"]["pitch_flags"]
                + event["intonation"]["end"]["pitch_flags"]
            )

            for diff, flag in zip(all_diffs, all_flags):
                y_vals_intonation.append(event["midi"][0] + diff)
                y_vals_reference.append(event["midi"][0])

                colors.append(color_map.get(flag, "gray"))

            x_vals_beats.extend(
                np.linspace(
                    float(event["start_quarter_length"]),
                    float(event["end_quarter_length"]),
                    len(all_diffs),
                )
            )

        x_vals_beats = np.array(x_vals_beats)
        y_vals_intonation = np.array(y_vals_intonation)
        y_vals_reference = np.array(y_vals_reference)

        fig, ax = plt.subplots()

        # Reference pitch
        ax.plot(
            x_vals_beats,
            y_vals_reference,
            color="black",
            linewidth=1.5,
            label="Reference Pitch",
        )

        # Build line segments for player's intonation
        points = np.column_stack((x_vals_beats, y_vals_intonation)).reshape(-1, 1, 2)

        segments = np.concatenate(
            [points[:-1], points[1:]],
            axis=1,
        )

        # There are N-1 segments for N points
        line_collection = LineCollection(
            segments,
            colors=colors[:-1],
            linewidth=2,
        )

        ax.add_collection(line_collection)

        # Make sure matplotlib includes the LineCollection in the axis limits
        ax.autoscale()

        ax.set_title("Intonation Difference Graph")
        ax.set_xlabel("Beats")
        ax.set_ylabel("Pitch")
        ax.grid(True)

        # Custom legend because LineCollection doesn't automatically
        # produce the legend entries we want.
        legend_items = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                label="Reference Pitch",
            ),
            Line2D(
                [0],
                [0],
                color="green",
                linewidth=2,
                label="Perfect",
            ),
            Line2D(
                [0],
                [0],
                color="yellow",
                linewidth=2,
                label="Okay",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=2,
                label="Wrong",
            ),
        ]

        ax.legend(handles=legend_items)

        if save:
            plt.savefig(
                "graph.png",
                dpi=300,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close()
