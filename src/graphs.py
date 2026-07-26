import matplotlib.pyplot as plt
import numpy as np


def show_and_save_graph(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    points: list[dict[str, np.int64 | np.float32]],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    plt.figure()

    plt.plot(x_vals, y_vals, label="RMS")

    onset_label_added = False
    offset_label_added = False

    for point in points:
        onset_idx = point.get("rms_onset_idx")
        onset_value = point.get("rms_onset_value")

        if onset_idx is not None and onset_value is not None:
            plt.scatter(
                x_vals[onset_idx],
                onset_value,
                marker="^",
                zorder=3,
                label="RMS onset" if not onset_label_added else None,
            )
            onset_label_added = True

        offset_idx = point.get("rms_offset_idx")
        offset_value = point.get("rms_offset_value")

        if offset_idx is not None and offset_value is not None:
            plt.scatter(
                x_vals[offset_idx],
                offset_value,
                marker="o",
                zorder=3,
                label="RMS offset" if not offset_label_added else None,
            )
            offset_label_added = True

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
