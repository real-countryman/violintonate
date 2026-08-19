from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


# TODO intonation tendency flag
@dataclass
class Output:
    score_events: list[dict]

    def print_score_events(self):
        for event in self.score_events:
            print(event)

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

        x_onset_rhythm_points = []
        y_onset_rhythm_points = []
        x_offset_rhythm_points = []
        y_offset_rhythm_points = []

        intonation_colors = []
        rhythm_onset_colors = []
        rhythm_offset_colors = []

        color_map = {
            "wrong": "red",
            "okay": "yellow",
            "perfect": "green",
            None: "blue",
        }

        for event in self.score_events:
            if event["kind"] != "note":
                continue

            all_diffs = (
                event["intonation"]["start"]["pitch_diffs"]
                + event["intonation"]["middle"]["pitch_diffs"]
                + event["intonation"]["end"]["pitch_diffs"]
            )

            add_intonation_flags = (
                event["intonation"]["start"]["pitch_flags"]
                + event["intonation"]["middle"]["pitch_flags"]
                + event["intonation"]["end"]["pitch_flags"]
            )

            # if None, append perfect default onset and offset to the graph
            onset_point, offset_point = self._get_rhythm_points(event)
            if onset_point != None:
                x_onset_rhythm_points.append(onset_point[0])
                y_onset_rhythm_points.append(onset_point[1])
            else:
                x_onset_rhythm_points.append(event["start_quarter_length"])
                y_onset_rhythm_points.append(event["midi"][0])

            if offset_point != None:
                x_offset_rhythm_points.append(offset_point[0])
                y_offset_rhythm_points.append(offset_point[1])
            else:
                x_offset_rhythm_points.append(event["end_quarter_length"])
                y_offset_rhythm_points.append(event["midi"][0])

            # appending None: not a bug, makes color blue!
            rhythm_onset_flag = event["rhythm"]["onset_diff_flag"]
            rhythm_offset_flag = event["rhythm"]["offset_diff_flag"]

            rhythm_onset_colors.append(color_map.get(rhythm_onset_flag, "gray"))
            rhythm_offset_colors.append(color_map.get(rhythm_offset_flag, "gray"))

            for diff, flag in zip(all_diffs, add_intonation_flags):
                y_vals_intonation.append(event["midi"][0] + diff)
                y_vals_reference.append(event["midi"][0])

                intonation_colors.append(color_map.get(flag, "gray"))

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

        fig, ax = plt.subplots(figsize=(20, 10))

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
            colors=intonation_colors[:-1],
            linewidth=2,
        )

        ax.add_collection(line_collection)

        # rhythm onset points
        ax.scatter(
            x_onset_rhythm_points,
            y_onset_rhythm_points,
            c=rhythm_onset_colors,
            s=50,
            zorder=10,
            marker="^",
            edgecolors="black",
        )

        # rhythm offset points
        ax.scatter(
            x_offset_rhythm_points,
            y_offset_rhythm_points,
            c=rhythm_offset_colors,
            s=50,
            zorder=10,
            marker="o",
            edgecolors="black",
        )

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
            Line2D(
                [0],
                [0],
                color="blue",
                linewidth=2,
                label="Undetected",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="gray",
                markersize=7,
                label="Onset",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="gray",
                markersize=7,
                label="Offset",
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

    def _get_rhythm_points(
        self,
        event,
    ) -> tuple[tuple[float, float], tuple[float, float]] | tuple[None, None]:
        print(event)

        if (
            event["rhythm"]["onset_diff_beats"] == None
            or event["rhythm"]["offset_diff_beats"] == None
            or event["start_quarter_length"] == None
            or event["end_quarter_length"] == None
        ):
            return None, None

        return (
            (
                event["start_quarter_length"] + event["rhythm"]["onset_diff_beats"],
                event["midi"][0],
            ),
            (
                event["end_quarter_length"] + event["rhythm"]["offset_diff_beats"],
                event["midi"][0],
            ),
        )

    def _validate(self): ...
