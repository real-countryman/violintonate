from dataclasses import dataclass


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

            onset_diff_flag = self.score_events[i]["rhythm"]["onset_diff_flag"]
            offset_diff_flag = self.score_events[i]["rhythm"]["offset_diff_flag"]

            rhythm_info = [
                ("Rhythm", "Onset", "Offset"),
                ("Time difference:", onset_diff_secs, offset_diff_secs),
                ("Evaluation:", onset_diff_flag, offset_diff_flag),
            ]

            for label, onset, offset in rhythm_info:
                if isinstance(onset, float):
                    onset = f"{onset:.2f}"
                if isinstance(offset, float):
                    offset = f"{offset:.2f}"

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

    def render_and_save_graph_output(self): ...
