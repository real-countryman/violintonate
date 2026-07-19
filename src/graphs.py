import matplotlib.pyplot as plt
import numpy as np


def show_and_save_graph(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
):

    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()
