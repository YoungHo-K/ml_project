import os
import numpy as np
import matplotlib.pyplot as plt

from plot.default import set_default_figure


def plot_histogram(histogram, bins, dst_dir_path=None):
    fig = plt.figure(figsize=(11, 7))

    ax = fig.add_subplot(1, 1, 1)
    ax = set_default_figure(ax)

    ax.set_xlabel("Bin", fontdict={"fontsize": 12})
    ax.set_ylabel("Frequency", fontdict={"fontsize": 12})

    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_xticks(range(0, len(bins)))
    ax.set_xticklabels([str(x) for x in bins], fontsize=11)

    ax.bar(np.arange(0.5, len(bins) - 1, 1), histogram, width=1.0, color="royalblue", edgecolor="white")

    plt.tight_layout(pad=1.0)
    plt.show() if dst_dir_path is None else plt.savefig(os.path.join(dst_dir_path, "plot_histogram.png"))

