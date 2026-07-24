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

    for point in points:
        idx = point["rms_offset_idx"]
        value = point["rms_offset_value"]

        if idx is None or value is None:
            continue

        plt.scatter(
            x_vals[idx],
            value,
            marker="o",
            zorder=3,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
