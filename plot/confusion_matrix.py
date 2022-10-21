import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from plot.default import set_default_figure


def plot_confusion_matrix(confusion_matrix, target_names=None, normalize=True, dst_dir_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = set_default_figure(ax)

    ax.set_xlabel('Predicted label', fontdict={'fontsize': 18}, labelpad=15.0)
    ax.set_ylabel('True label', fontdict={'fontsize': 18})

    if target_names is not None:
        ticks = range(0, len(target_names))

        ax.set_xticks(ticks, target_names, fontsize=15)
        ax.set_yticks(ticks, target_names, fontsize=15)

    if normalize is True:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    threshold = confusion_matrix.max() / 1.5 if normalize is True else confusion_matrix.max() / 2
    for y, x in itertools.product(range(0, len(confusion_matrix)), range(0, len(confusion_matrix))):
        ax.text(x, y, f"{confusion_matrix[y, x] * 100:.1f}%",
                horizontalalignment="center",
                color="white" if confusion_matrix[y, x] > threshold else "black", fontdict={'fontsize': 12})

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.get_cmap("Greys"))

    plt.tight_layout(pad=2.0)
    plt.show() if dst_dir_path is None else plt.savefig(os.path.join(dst_dir_path, "plot_confusion_matrix.png"))


if __name__ == '__main__':

    confusion_matrix = np.array([[81, 271, 55, 28],
                                 [42, 6474, 274, 27],
                                 [4, 160, 4454, 40],
                                 [0, 6, 52, 545]])

    plot_confusion_matrix(confusion_matrix, target_names=['Tsunami', 'Gafgyt', 'Mirai', 'Dofloo'])